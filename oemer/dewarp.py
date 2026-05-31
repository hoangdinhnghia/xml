import os
import pickle
import typing
from typing import List, Tuple, Any, Union
from typing_extensions import Self

import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.interpolate import interp1d, griddata
from sklearn.linear_model import LinearRegression

from oemer.symbol_extraction import morph_open
from oemer.utils import get_logger
from oemer.bbox import BBox
from oemer.utils import _save


logger = get_logger(__name__)

"""
dewarp.py

Mục đích:
- Khử biến dạng hình học của ảnh bản nhạc dựa trên bản đồ dự đoán các dòng khuông
    (`staff_pred`). Thuật toán tách ảnh thành các đoạn lưới dọc, gom các đoạn liên thông
    thành nhóm, nội suy các đoạn bị đứt và tạo ánh xạ tọa độ `coords_x, coords_y` dùng
    để remap ảnh về không gian đã chuẩn hoá.

Ghi chú cho người học:
- File này tập trung vào các bước hình học (quantization -> grouping -> interpolation).
- Tôi chỉ thêm chú thích để giải thích ý đồ mỗi hàm và các biến trung gian. Không thay
    đổi bất kỳ logic xử lý nào để đảm bảo hành vi chạy không đổi.
"""


class Grid:
    def __init__(self) -> None:
        self.id: Union[int, None] = None
        self.bbox: BBox = None  # type: ignore
        self.y_shift: int = 0
    """Đại diện cho một đoạn dọc (mảnh) của dòng khuông.

    Thuộc tính:
    - `id`: chỉ số nguyên nhận dạng mảnh trong danh sách `grids`.
    - `bbox`: hộp bao dạng (x0, y0, x1, y1) cho mảnh.
    - `y_shift`: giá trị dịch dọc (thường dùng khi nội suy/điều chỉnh vị trí).
    - `y_center`/`height`: thuộc tính tính toán tiện lợi cho tâm và chiều cao mảnh.
    """

    @property
    def y_center(self) -> float:
        return (self.bbox[1]+self.bbox[3]) / 2

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


class GridGroup:
    def __init__(self) -> None:
        self.id: Union[int, None] = None
        self.reg_id: Union[int, None] = None
        self.bbox: BBox = None  # type: ignore
        self.gids: List[int] = []
        self.split_unit: int = None  # type: ignore
    """Tập hợp các `Grid` liên tiếp tạo thành một dải ngang (một 'line band').

    Thuộc tính:
    - `id`: chỉ số nhóm sau khi sắp xếp/tái đánh số.
    - `reg_id`: id vùng tạm thời từ `scipy.ndimage.label` trước khi tái chỉ số.
    - `bbox`: hộp bao chung của toàn bộ nhóm.
    - `gids`: danh sách các `Grid.id` thành viên.
    - `split_unit`: kích thước bước ngang dùng khi phân đoạn (bằng chiều rộng Grid).
    """

    @property
    def y_center(self):
        return round((self.bbox[1]+self.bbox[3]) / 2)

    def __lt__(self, tar: Self) -> bool:
        # Sắp xếp theo bề rộng nhóm
        w = self.bbox[2] - self.bbox[0]
        tw = tar.bbox[2] - tar.bbox[0]
        return w < tw

    def __repr__(self):
        return f"Grid Group {self.id} / Width: {self.bbox[2]-self.bbox[0]} / BBox: {self.bbox}" \
            f" / Y-center: {self.y_center} / Reg. ID: {self.reg_id}"


def build_grid(st_pred: ndarray, split_unit: int = 11) -> Tuple[ndarray, List[Grid]]:
    """Chuyển bản đồ nhị phân của dòng khuông thành các đoạn dọc ngắn (Grid).

        Thao tác:
        - Duyệt ảnh theo từng khối dọc (mỗi khối có bề ngang `split_unit`).
        - Trong mỗi khối dọc, quét theo trục y để phát hiện các đoạn liên tiếp có
            pixel "on" (thể hiện phần của dòng khuông). Mỗi đoạn ngắn thỏa điều kiện
            chiều cao được lưu dưới dạng một `Grid` với `bbox` và `id`.

        Trả về:
        - `grid_map`: ma trận cùng kích thước với `st_pred`, mỗi ô chứa id của Grid
            tương ứng hoặc -1 nếu không thuộc Grid nào.
        - `grids`: danh sách các đối tượng `Grid` đã phát hiện, giữ `bbox` (x0,y0,x1,y1)
            và `id` tương ứng.

        Lý do dùng chiến lược này:
        - Giảm nhiễu đơn pixel bằng cách quyết định trạng thái trong một cửa sổ
            ngang (`is_on`) thay vì dùng từng pixel riêng lẻ.
        - Chia vấn đề lớn (dòng khuông trên toàn ảnh) thành các mảnh nhỏ dễ xử lý,
            thuận tiện cho bước gom và nội suy sau này.
        """
    grid_map = np.zeros(st_pred.shape) - 1
    h, w = st_pred.shape

    # Hàm phụ: quyết định một cửa sổ ngang nhỏ có thuộc dòng khuông hay không
    # bằng quy tắc đa số, giúp giảm nhạy với nhiễu đơn pixel.
    is_on = lambda data: np.sum(data) > split_unit // 2

    grids: List[Grid] = []
    # Trượt theo trục x với bước split_unit rồi quét theo trục y
    for i in range(0, w, split_unit):
        cur_y = 0
        last_y = 0
        # Trạng thái ban đầu tại hàng đầu tiên của cột khối hiện tại
        cur_stat = is_on(st_pred[cur_y, i:i+split_unit])
        while cur_y < h:
            # Tiến đến khi trạng thái (on/off) thay đổi
            while cur_y < h and cur_stat == is_on(st_pred[cur_y, i:i+split_unit]):
                cur_y += 1
            # Nếu vừa kết thúc một đoạn "on" có chiều cao không quá lớn,
            # ghi lại thành một mảnh Grid
            if cur_stat and (cur_y - last_y < split_unit):
                grid_map[last_y:cur_y, i:i+split_unit] = len(grids)
                gg = Grid()
                gg.bbox = (i, last_y, i + split_unit, cur_y)
                gg.id = len(grids)
                grids.append(gg)
            # Đảo trạng thái và tiếp tục quét
            cur_stat = not cur_stat
            last_y = cur_y
    return grid_map, grids


def build_grid_group(grid_map: ndarray, grids: List[Grid]) -> Tuple[ndarray, List[GridGroup]]:
    """Gom các đoạn Grid lân cận thành các GridGroup (dải ngang liền mạch).

        Thao tác:
        - Dùng `scipy.ndimage.label` trên `grid_map+1` để đánh nhãn vùng liên thông
            (đổi -1 thành 0 trước khi label để tránh tính background).
        - Với mỗi vùng, thu tập id của các Grid thành viên, tính `bbox` tổng hợp của
            toàn vùng và khởi tạo một `GridGroup` chứa thông tin này.
        - Sắp xếp các GridGroup theo bề rộng giảm dần, và tái đánh số các nhóm để
            nhóm lớn hơn có chỉ số nhỏ hơn — điều này tiện cho việc tham chiếu sau này.

        Trả về:
        - `gg_map`: ma trận cùng kích thước với `grid_map`, mỗi ô chứa id GridGroup
            (hoặc -1 nếu không thuộc nhóm nào).
        - `grid_groups`: danh sách các đối tượng `GridGroup` đã được sắp xếp và tái chỉ số.
        """
    regions, feat_num = scipy.ndimage.label(grid_map + 1)
    grid_groups = []
    for i in range(feat_num):
        region = grid_map[regions == i + 1]
        gids = list(np.unique(region).astype(int))
        gids = sorted(gids)
        # Mảnh trái nhất và phải nhất xác định biên ngang của nhóm
        lbox = grids[gids[0]].bbox
        rbox = grids[gids[-1]].bbox
        box = (
            min(lbox[0], rbox[0]),
            min(lbox[1], rbox[1]),
            max(lbox[2], rbox[2]),
            max(lbox[3], rbox[3]),
        )
        gg = GridGroup()
        gg.reg_id = i + 1
        gg.gids = gids
        gg.bbox = box
        gg.split_unit = lbox[2] - lbox[0]
        grid_groups.append(gg)

    # Sắp xếp nhóm theo bề rộng (lớn trước) và dựng lại gg_map với id mới liên tiếp
    grid_groups = sorted(grid_groups, reverse=True)
    gg_map = np.zeros_like(regions) - 1
    for idx, gg in enumerate(grid_groups):
        gg.id = idx
        gg_map[regions == gg.reg_id] = idx
        gg.reg_id = idx

    return gg_map, grid_groups


def connect_nearby_grid_group(
    gg_map: ndarray, 
    grid_groups: List[GridGroup], 
    grid_map: ndarray, 
    grids: List[Grid], 
    ref_count: int = 8, 
    max_step: int = 20
) -> ndarray:
    """Kết nối và lấp đầy các khoảng trống ngắn giữa các nhóm lưới.
    Đối với mỗi Nhóm Lưới, chúng ta cố gắng ngoại suy quỹ đạo tâm dọc của nó bằng cách sử dụng
    phép nội suy tuyến tính trên một số lưới tham chiếu. Sau đó, chúng ta bước ra ngoài (sang trái trong
    cách triển khai) và kiểm tra xem dải ngoại suy có gặp nhóm khác hay không. Nếu có,
    chúng ta nội suy các mục Lưới trung gian và chèn chúng để lấp đầy khoảng trống.
    Lưu ý quan trọng dành cho người học:
    - Vòng lặp sẽ duyệt qua các nhóm và duy trì một tập hợp `còn lại` để tránh duyệt lại.
    - Chúng ta không giới hạn chỉ số lát cắt ở đây; trong mã sản xuất, bạn có thể muốn
    đảm bảo `y` và `end_x-step_size` nằm trong giới hạn hình ảnh để tránh hành vi không mong muốn
    trên các đầu vào cực đoan.

    Tham số:
    - `gg_map`: bản đồ id nhóm lưới hiện tại (mỗi pixel là id nhóm hoặc -1).
    - `grid_groups`: danh sách các nhóm lưới đã gom.
    - `grid_map`: bản đồ id lưới con ban đầu.
    - `grids`: danh sách các lưới con (`Grid`) hiện có.
    - `ref_count`: số lưới tham chiếu dùng để fit đường xu hướng.
    - `max_step`: số bước ngoại suy tối đa cho mỗi lần tìm kết nối.

    Trả về:
    - `new_gg_map`: bản đồ nhóm lưới sau khi đã chèn các lưới nội suy để nối các khoảng đứt.

    """
    new_gg_map = np.copy(gg_map)
    # Bắt đầu từ nhóm lớn nhất (id 0 sau khi đã sắp xếp ở build_grid_group)
    ref_gids = grid_groups[0].gids[:ref_count]
    idx = 0
    gg = grid_groups[idx]
    remaining = set(range(len(grid_groups)))
    while remaining:
        # Chọn nhóm chưa duyệt tiếp theo
        if gg.id not in remaining:
            if remaining:
                gid = remaining.pop()
                gg = grid_groups[gid]
                ref_gids = gg.gids[:ref_count]
            else:
                break
        else:
            remaining.remove(gg.id)

        if len(ref_gids) < 2:
            # Không đủ điểm tham chiếu để fit một đường xu hướng tin cậy
            continue

        # Fit mô hình tuyến tính trên tâm y của các lưới tham chiếu
        step_size = gg.split_unit
        centers = [grids[gid].y_center for gid in ref_gids]
        x = np.arange(len(centers)).reshape(-1, 1) * step_size
        model = LinearRegression().fit(x, centers)
        ref_box = grids[ref_gids[0]].bbox

        end_x = ref_box[0]
        h = ref_box[3] - ref_box[1]
        cands_box = []  # Lưu các box ứng viên dọc theo quỹ đạo ngoại suy
        for i in range(max_step):
            tar_x = (-i - 1) * step_size
            cen_y = model.predict([[tar_x]])[0]  # Tâm y dự đoán tại vị trí lệch hiện tại
            y = int(round(cen_y - h / 2))
            # LƯU Ý: lát cắt vùng có thể vượt biên ở các trường hợp biên.
            # Khi cần độ bền cao hơn, có thể thêm clamp chỉ số ở đây.
            region = new_gg_map[y:y + h, end_x - step_size:end_x]  # Vùng kiểm tra va chạm
            unique, counts = np.unique(region, return_counts=True)
            labels = set(unique)
            if -1 in labels:
                # Loại id nền (-1) ra khỏi tập xét
                labels.remove(-1)
                no_id_idx = np.where(unique == -1)[0][0]
                unique = np.delete(unique, no_id_idx)
                counts = np.delete(counts, no_id_idx)

            cands_box.append((end_x - step_size, y, end_x, y + h))
            if len(labels) == 0:
                # Vùng trống, tiếp tục ngoại suy
                end_x -= step_size
            else:
                # Đã chạm nhóm khác; nếu nhiều nhãn thì chọn nhãn chồng lấp nhiều nhất
                cands_box = cands_box[:-1]
                if len(labels) > 1:
                    overlapped_size = sorted(zip(unique, counts), key=lambda it: it[1], reverse=True)
                    label = overlapped_size[0][0]
                else:
                    label = labels.pop()

                # Đảm bảo phần chồng lấp hợp lý về hình học
                tar_box = grid_groups[label].bbox
                if tar_box[2] > end_x:
                    break

                # Tìm grid đại diện trong vùng chồng lấp
                yidx, xidx = np.where(region == label)
                yidx += y
                xidx += end_x - step_size
                reg = grid_map[yidx, xidx]
                grid_id, counts = np.unique(reg, return_counts=True)
                if len(grid_id) > 1:
                    logger.warn("Detected multiple possible overlapping grids: %s. Reg. count: %s", str(grid_id), str(counts))
                grid_id = int(grid_id[np.argmax(counts)])
                assert grid_id in grid_groups[label].gids, f"{grid_id}, {label}"
                grid = grids[grid_id]

                # Nội suy giữa grid vừa tìm thấy và grid tham chiếu đầu của nhóm hiện tại
                centers = [grid.y_center, centers[0]]
                x = [-i - 1, 0]  # type: ignore
                inter_func = interp1d(x, centers, kind='linear')

                # Chèn các grid nội suy vào cấu trúc dữ liệu
                cands_ids = []
                for bi, box in enumerate(cands_box):
                    interp_y = round(inter_func(-bi - 1) - h / 2)
                    grid = Grid()
                    box = (box[0], interp_y, box[2], interp_y + h)
                    grid.bbox = box
                    grid.id = len(grids)
                    cands_ids.append(len(grids))
                    gg.gids.append(len(grids))
                    gg.bbox = (
                        min(gg.bbox[0], box[0]),
                        min(gg.bbox[1], box[1]),
                        max(gg.bbox[2], box[2]),
                        max(gg.bbox[3], box[3])
                    )
                    gg.bbox = typing.cast(BBox, [int(bb) for bb in gg.bbox])
                    box = [int(bb) for bb in box]  # type: ignore
                    grids.append(grid)
                    new_gg_map[box[1]:box[3], box[0]:box[2]] = gg.id

                # Tiếp tục đi từ nhóm vừa chạm
                gg = grid_groups[label]
                gids = gg.gids + cands_ids[::-1]
                ref_gids = gids[:ref_count]

                break

    return new_gg_map


def build_mapping(gg_map: ndarray, min_width_ratio: float = 0.4) -> Tuple[ndarray, ndarray]:

    """Lấy mẫu các điểm kiểm soát từ `gg_map` và xây một bản đồ rời rạc `coords_y`.

        Mô tả chi tiết:
        - Với mỗi vùng liên thông (connected region) trong `gg_map`, ta tính tâm dọc
            (vertical center) của vùng tại các cột khác nhau. Để giảm số điểm, ta chỉ
            lấy mẫu mỗi `period` cột.
        - Bỏ qua những vùng có bề ngang nhỏ hơn `min_width_ratio * width_image` vì
            chúng có thể là nhiễu.

        Lưu ý:
        - Thêm hai hàng biên (top/bottom) vào `points` để đảm bảo nội suy có phủ
            rìa ảnh, tránh giá trị NaN tại biên.
    """
    regions, num = scipy.ndimage.label(gg_map + 1)
    min_width = gg_map.shape[1] * min_width_ratio

    points = []
    coords_y = np.zeros_like(gg_map)
    period = 10
    for i in range(num):
        y, x = np.where(regions == i + 1)
        w = np.max(x) - np.min(x)
        if w < min_width:
            # ignore very small regions (likely noise)
            continue

        target_y = round(np.mean(y))

        uniq_x = np.unique(x)
        # sample every `period` columns to reduce number of control points
        for ii, ux in enumerate(uniq_x):
            if ii % period == 0:
                meta_idx = np.where(x == ux)[0]
                sub_y = y[meta_idx]
                cen_y = round(np.mean(sub_y))
                coords_y[int(target_y), int(ux)] = cen_y
                points.append((target_y, ux))

    # Add corner rows to guarantee boundary coverage for interpolation
    coords_y[0] = 0
    coords_y[-1] = len(coords_y) - 1
    for i in range(coords_y.shape[1]):
        points.append((0, i))
        points.append((coords_y.shape[0] - 1, i))

    return coords_y, np.array(points)


def estimate_coords(staff_pred: ndarray) -> Tuple[ndarray, ndarray]:
    """Ước lượng bản đồ tọa độ remap từ mask dự đoán dòng khuông.

    Quy trình:
    - Làm dày dòng khuông bằng giãn ảnh và mở hình thái để ổn định hình học.
    - Chia thành các mảnh `Grid`, gom thành `GridGroup`, rồi nối các khoảng đứt ngắn.
    - Tạo các điểm điều khiển thưa (`points`, `vals`) và nội suy `coords_y` liên tục.
    - `coords_x` giữ theo lưới cột gốc; `coords_y` biểu diễn ánh xạ dọc sau dewarp.

    Trả về:
    - `coords_x`: lưới tọa độ x đích (float32).
    - `coords_y`: lưới tọa độ y đích (float32) để truyền vào `cv2.remap`.
    """
    ker = np.ones((6, 1), dtype=np.uint8)
    pred = cv2.dilate(staff_pred.astype(np.uint8), ker)
    pred = morph_open(pred, (1, 6))

    logger.debug("Building grids")
    grid_map, grids = build_grid(pred)

    logger.debug("Labeling areas")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    logger.debug("Connecting lines")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    logger.debug("Building mapping")
    coords_y, points = build_mapping(new_gg_map)

    logger.debug("Dewarping")
    # Lấy giá trị điều khiển từ coords_y tại các điểm mẫu; ép kiểu chỉ số sang int.
    vals = coords_y[points[:, 0].astype(int), points[:, 1].astype(int)]
    # Dựng lưới đều để nội suy ra bản đồ đầy đủ trên toàn ảnh
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1]
    # Nội suy bản đồ coords_y liên tục từ tập điểm điều khiển thưa
    coords_y = griddata(points, vals, (grid_x, grid_y), method='linear')

    coords_x = grid_y.astype(np.float32)
    coords_y = coords_y.astype(np.float32)
    return coords_x, coords_y


def dewarp(img: ndarray, coords_x: ndarray, coords_y: ndarray) -> ndarray:
    """Khử biến dạng ảnh bằng ánh xạ tọa độ đã ước lượng.

    - Mỗi điểm (x, y) ở ảnh đầu ra sẽ lấy mẫu từ vị trí (`coords_x`, `coords_y`)
      tương ứng trên ảnh đầu vào.
    - Dùng `cv2.INTER_CUBIC` để nội suy mượt và `BORDER_REPLICATE` để xử lý biên.
    """
    return cv2.remap(img.astype(np.float32), coords_x, coords_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def save_dewarp_viz(out_dir: str, before_img: ndarray, after_img: ndarray, coords_x: ndarray, coords_y: ndarray) -> None:
    """Lưu ảnh trước/sau dewarp và ảnh trực quan hóa vector ánh xạ."""
    try:
        _save(out_dir, 'step0_before_dewarp', before_img)
        _save(out_dir, 'step0_after_dewarp', after_img)
        vis = before_img.copy()
        h, w = coords_x.shape
        step = max(20, min(h, w) // 30)
        for yy in range(0, h, step):
            for xx in range(0, w, step):
                tx = int(np.clip(coords_x[yy, xx], 0, w - 1))
                ty = int(np.clip(coords_y[yy, xx], 0, h - 1))
                cv2.arrowedLine(vis, (xx, yy), (tx, ty), (0, 255, 255), 1, tipLength=0.3)
        _save(out_dir, 'step0_dewarp_map', vis)
    except Exception:
        logger.exception('Không thể lưu ảnh trực quan hóa dewarp')


if __name__ == "__main__":
    f_name = "jion_the_fun"
    #f_name = "last"
    #f_name = "tabi"
    img_path = f"images/de/{f_name}.png"

    #img_path = "../test_imgs/Chihiro/7.jpg"
    #img_path = "../test_imgs/Gym/2.jpg"

    ori_img = cv2.imread(img_path)
    f_name, ext = os.path.splitext(os.path.basename(img_path))
    parent_dir = os.path.dirname(img_path)
    pkl_path = os.path.join(parent_dir, f_name+".pkl")
    ff = pickle.load(open(pkl_path, "rb"))
    st_pred = ff['staff']
    ori_img = cv2.resize(ori_img, (st_pred.shape[1], st_pred.shape[0]))

    ker = np.ones((6, 1), dtype=np.uint8)
    pred = cv2.dilate(st_pred.astype(np.uint8), ker)
    pred = morph_open(pred, (1, 6))

    print("Đang xây dựng các mảnh lưới")
    grid_map, grids = build_grid(pred)

    print("Đang gán nhãn các vùng")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    print("Đang kết nối các đoạn bị đứt")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    print("Đang ước lượng ánh xạ")
    coords_y, points = build_mapping(new_gg_map)

    print("Đang khử biến dạng")
    out = np.copy(ori_img)
    vals = coords_y[points[:, 0], points[:, 1]]
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1]
    mapping = griddata(points, vals, (grid_x, grid_y), method='linear')
    for i in range(out.shape[-1]):
        out[..., i] = cv2.remap(out[..., i].astype(np.float32), grid_y.astype(np.float32), mapping.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    mix = np.hstack([ori_img, out])


import random
def teaser():
    plt.clf()
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.subplot(231)
    plt.title("Dự đoán")
    plt.axis('off')
    plt.imshow(st_pred, cmap="Greys")

    plt.subplot(232)
    plt.title("Hình thái học")
    plt.axis('off')
    plt.imshow(pred, cmap='Greys')

    plt.subplot(233)
    plt.title("Lượng tử lưới")
    plt.axis('off')
    plt.imshow(grid_map>0, cmap='Greys')

    plt.subplot(234)
    plt.title("Gom nhóm")
    plt.axis('off')
    ggs = set(np.unique(gg_map))
    ggs.remove(-1)
    _gg_map = np.ones(gg_map.shape+(3,), dtype=np.uint8) * 255
    for i in ggs:
        ys, xs = np.where(gg_map==i)
        for c in range(3):
            v = random.randint(0, 255)
            _gg_map[ys, xs, c] = v
    plt.imshow(_gg_map)

    plt.subplot(235)
    plt.title("Kết nối")
    plt.axis('off')
    plt.imshow(new_gg_map>0, cmap='Greys')

    plt.subplot(236)
    plt.title("Khử biến dạng")
    plt.axis('off')
    plt.imshow(out)