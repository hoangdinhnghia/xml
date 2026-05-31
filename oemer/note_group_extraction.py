"""
Mô tả quy trình gom nhóm đầu nốt (Note grouping)

Module này gom nhóm các `notehead` thành `NoteGroup` thông qua các bước chính sau:

- Mở rộng vùng của cọng nốt (`stems_rests_pred`) bằng phép nhân chập hình thái (dilation)
    với kernel cố định `(3, 2)` để tăng khả năng kết nối giữa đầu nốt và cọng.
- Kết hợp mặt nạ `notehead_pred` và vùng cọng đang được mở rộng, sau đó chạy
    connected-component labeling bằng `scipy.ndimage.label` để tạo các cụm vùng liên thông
    ứng viên (component labels).
- Dựa trên giao nhau giữa `note_id` map và các thành phần liên thông này, các đầu nốt
    được gom vào cùng một nhóm; có logic gộp nhãn khi một nốt nằm trên nhiều component.
- Phân tích hướng cọng (`parse_stem_direction`) so sánh bounding box vùng cọng (group box)
    và bounding box của các đầu nốt trong nhóm để gán `stem_up=True/False`. Với trường hợp mơ hồ
    (ví dụ nhóm chỉ có một nốt), thuật toán tìm nhóm lân cận (`get_possible_nearby_gid`) và
    có thể chuyển nốt đó sang nhóm hợp lý nếu phù hợp.
- `gen_groups` khởi tạo các đối tượng `NoteGroup`, gán `id`, `bbox`, `note_ids`, `stem_up`,
    `has_stem`, `all_same_type`, và cập nhật `track`/`group` dựa trên staff gần nhất.

Ghi chú kỹ thuật:
- Dilation dùng kernel `(3,2)` (không phải các phép hình thái học phức tạp khác).
- Connected-component labeling dùng `scipy.ndimage.label`.
- Việc hiệu chỉnh nhóm lân cận hiện chỉ áp dụng chủ yếu cho trường hợp nhóm có một
    nốt mơ hồ; các kiểm tra nâng cao khác (ví dụ tách/ghép cho nhóm nhiều nốt) là heuristic
    hoặc còn ở dạng placeholder (ví dụ `post_check_groups`).
- Module còn chứa một số helper nhẹ (ví dụ `group_notes`, `advanced_group_notes`,
    `apply_music_constraints`) để gom theo tâm nốt hoặc luật đơn giản, nhưng chúng không
    phải là phần chính của luồng `extract()`.

Đoạn mô tả của bạn phù hợp về mặt ý tưởng; docstring này làm rõ các chi tiết
thực thi quan trọng để tránh nhầm lẫn.
"""

from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import scipy.ndimage
import numpy as np
from numpy import ndarray

from oemer import layers
# Tránh vòng lặp import: hàm `predict` được import trễ bên trong `predict_symbols()`.
from oemer.utils import find_closest_staffs, get_global_unit_size, get_unit_size
from oemer.utils import get_logger
from oemer.bbox import (
    BBox,
    get_center,
    merge_nearby_bbox,
    get_bbox,
    rm_merge_overlap_bbox,
    to_rgb_img,
    draw_bounding_boxes
)

# Biến toàn cục phục vụ trực quan hóa gỡ lỗi
grp_img: ndarray

logger = get_logger(__name__)
from oemer.utils import C, _overlay, _rect, _text, _legend, _save


class NoteGroup:
    def __init__(self) -> None:
        self.id: Union[int, None] = None
        self.bbox: BBox = None  # type: ignore
        self.note_ids: List[int] = []
        self.top_note_ids: List[int] = []  # Dùng cho trường hợp nhiều bè (multi-melody)
        self.bottom_note_ids: List[int] = []  # Dùng cho trường hợp nhiều bè (multi-melody)
        self.stem_up: Union[bool, None] = None
        self.has_stem: Union[bool, None] = None
        self.all_same_type: Union[bool, None] = None  # Tất cả nốt đều cùng loại (đặc hoặc rỗng)
        self.group: Union[int, None] = None
        self.track: Union[int, None] = None

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __len__(self):
        return len(self.note_ids)

    def __repr__(self):
        return f"Note Group No. {self.id} / Group: {self.group} / Track: {self.track} :(\n" \
            f"\tNote count: {len(self.note_ids)}\n" \
            f"\tStem up: {self.stem_up}\n" \
            f"\tHas stem: {self.has_stem}\n" \
            ")\n"


def group_noteheads() -> Tuple[Dict[int, List[int]], ndarray]:
    """
    Gom các đầu nốt vào các nhóm tạm thời dựa trên thành phần liên thông.

    Đầu vào:
    - `note_id_map`: bản đồ id của từng đầu nốt đã tách.
    - `notehead_pred`: mặt nạ đầu nốt.
    - `stems_rests_pred`: mặt nạ thân/cọng nốt và dấu lặng.

    Đầu ra:
    - `groups`: ánh xạ nhãn thành phần liên thông -> danh sách `note_id`.
    - `nh_label`: bản đồ nhãn thành phần liên thông sau khi đã hợp nhất.

    Thuật toán:
    1. Giãn nở vùng cọng để tăng khả năng kết nối đầu nốt với thân.
    2. Trộn head + stem rồi chạy connected-components.
    3. Với mỗi `note_id`, lấy các nhãn liên thông phủ lên vùng lân cận của nốt.
    4. Nếu một nốt chạm nhiều nhãn thì hợp nhất các nhãn đó để đảm bảo cùng nhóm.
    """
    # Lấy dữ liệu cần thiết
    note_id_map = layers.get_layer('note_id')
    notehead = layers.get_layer('notehead_pred')
    stems = layers.get_layer('stems_rests_pred')

    # Mở rộng vùng cọng
    ker = np.ones((3, 2), dtype=np.uint8)
    ext_stems = cv2.dilate(stems.astype(np.uint8), ker)

    # Gán nhãn cho từng vùng ứng viên (connected components)
    mix = notehead + ext_stems
    mix[mix>1] = 1
    nh_label, _ = scipy.ndimage.label(mix)
    nids = set(np.unique(note_id_map))
    if -1 in nids:
        nids.remove(-1)

    groups: Dict = {}
    for nid in nids:
        nys, nxs = np.where(note_id_map==nid)

        offset = 3
        top = np.min(nys) - offset #unit_size // 2
        bt = np.max(nys) + offset #unit_size // 2
        left = np.min(nxs)  # - unit_size // 3
        right = np.max(nxs)  # + unit_size // 3
        covered_region = nh_label[top:bt, left:right]
        labels = set(np.unique(covered_region))
        if 0 in labels:
            labels.remove(0)

        if len(labels) == 0:
            continue
        elif len(labels) > 1:
            keys = set(groups.keys())
            inter = labels.intersection(keys)
            if len(inter) == 0:
                # Thành viên đầu tiên của nhóm mới.
                label = labels.pop()
            elif len(inter) == 1:
                # Đã có nhóm tương ứng, dùng lại nhãn cũ.
                label = inter.pop()
            else:
                # Nốt nằm ở vùng nối giữa nhiều nhóm đã đăng ký.
                # Cần gộp các nhóm chồng lấp vào một nhãn đại diện.
                label = inter.pop()
                tmp_g = groups[label]
                for k in inter:
                    tmp_g.extend(groups[k])
                    del groups[k]
                groups[label] = tmp_g

            for ll in labels:
                # Cập nhật lại bản đồ nhãn sau khi hợp nhất.
                nh_label[nh_label==ll] = label
        else:
            label = labels.pop()

        if label not in groups:
            groups[label] = []
        groups[label].append(nid)

    # Loại bỏ các nhãn (vùng) mà không có đầu nốt đính kèm
    lls = set(np.unique(nh_label))
    diff = lls.difference(groups.keys())
    for ll in diff:
        nh_label[nh_label==ll] = 0

    return groups, nh_label


def get_possible_nearby_gid(cur_note, group_map, scan_range_ratio=5):
    """
    Tìm nhãn nhóm lân cận theo phương dọc cho một note đơn lẻ.

    Tham số:
        - cur_note: đối tượng NoteHead đang xét
        - group_map: mảng nhãn nhóm tạm thời
        - scan_range_ratio: bán kính tìm kiếm theo tỉ lệ unit_size

    Trả về nhãn nhóm phù hợp hoặc None nếu không tìm thấy.
    """
    bbox = cur_note.bbox
    cen_x, cen_y = get_center(bbox)
    cur_gid = group_map[cen_y, cen_x]
    unit_size = get_unit_size(cen_x, cen_y)

    w = bbox[2] - bbox[0] + 4
    start_x = bbox[0] - round(w / 2)
    end_x = min(start_x+w, group_map.shape[1])
    def search(cur_y, y_bound, step):
        while True:
            if step > 0 and cur_y >= y_bound:
                break
            elif step < 0 and cur_y < y_bound:
                break
            pxs = group_map[int(cur_y), int(start_x):int(end_x)]
            gids = set(np.unique(pxs))
            if 0 in gids:
                gids.remove(0)
            if cur_gid in gids:
                gids.remove(cur_gid)

            if len(gids) > 0:
                if len(gids) > 1:
                    # Lấy gid có vùng chồng lấp lớn nhất
                    reg = []
                    for gg in gids:
                        reg.append((gg, pxs[pxs==gg].size))
                    gid = sorted(reg, key=lambda it: it[1])[-1][0]
                else:
                    gid = gids.pop()
                return gid, cur_y
            cur_y += step
        return None, None

    st1, st2 = find_closest_staffs(cen_x, cen_y)
    y_upper = min(st1.y_upper, st2.y_upper)
    y_lower = max(st1.y_lower, st2.y_lower)

    # Quét theo lưới theo hướng lên trên.
    cur_y = bbox[1] - 1
    y_bound = max(cur_y - scan_range_ratio * unit_size, y_upper)
    gid_top, gty = search(cur_y, y_bound, -1)

    # Tìm theo lưới (grid) hướng xuống
    cur_y = bbox[3] + 1
    y_bound = min(cur_y + scan_range_ratio * unit_size, y_lower)
    gid_bt, gby = search(cur_y, y_bound, 1)

    if gid_top is not None and gid_bt is not None:
        diff_top = abs(cen_y - gty)
        diff_bt = abs(cen_y - gby)
        return gid_top if diff_top < diff_bt else gid_bt
    elif gid_top is not None:
        return gid_top
    elif gid_bt is not None:
        return gid_bt
    return None


def check_valid_new_group(ori_grp, tar_grp, group_map, max_x_diff_ratio=0.5):
    """
    Kiểm tra xem việc chuyển từ nhóm gốc sang nhóm đích có hợp lệ theo lệch ngang.
    Trả về True nếu nhóm đích là hợp lệ (hoặc None => luôn hợp lệ).
    """
    if tar_grp is None:
        return True

    def _get_box(gid):
        ys, xs = np.where(group_map==gid)
        return (np.min(xs), np.min(ys), np.max(xs), np.max(ys))

    ori_box = _get_box(ori_grp)
    tar_box = _get_box(tar_grp)
    ori_x_cen, ori_y_cen = get_center(ori_box)
    tar_x_cen, _ = get_center(tar_box)
    unit_size = get_unit_size(ori_x_cen, ori_y_cen)
    max_x_diff = unit_size * max_x_diff_ratio
    diff = abs(tar_x_cen - ori_x_cen)
    return diff < max_x_diff


def parse_stem_direction(
    groups: Dict[int, List[int]], 
    group_map: ndarray, 
    tolerance_ratio: float = 0.2, 
    max_x_diff_ratio: float = 0.5
) -> Tuple[Dict[int, List[int]], ndarray]:
    """
    Suy luận hướng cọng cho từng nhóm dựa trên so sánh hình học.

    Ý tưởng:
    - So sánh `group bbox` (vùng liên thông head+stem) với `notes bbox`.
    - Nếu vùng liên thông dư đáng kể phía trên thì gán `stem_up=True`.
    - Nếu dư đáng kể phía dưới thì gán `stem_up=False`.
    - Nếu mơ hồ và nhóm chỉ có một nốt, thử tìm nhóm lân cận để hợp nhất.

    Trả về:
    - `groups` đã tinh chỉnh.
    - `group_map` đã cập nhật nhãn sau các thao tác hợp nhất.
    """
    # Lấy các tham số/đối tượng cần thiết
    notes = layers.get_layer('notes')

    temp_result = {}
    for gp, nids in groups.items():
        # Lấy hộp bao của vùng liên thông (pixel) và hộp bao của các đầu nốt.
        gy, gx = np.where(group_map==gp)
        gbox = (np.min(gx), np.min(gy), np.max(gx), np.max(gy))
        nbox = np.array([notes[nid].bbox for nid in nids])
        nbox = (np.min(nbox[:, 0]), np.min(nbox[:, 1]), np.max(nbox[:, 2]), np.max(nbox[:, 3]))  # type: ignore
        # Chiều cao trung bình của đầu nốt trong nhóm.
        nh = np.mean([notes[nid].bbox[3]-notes[nid].bbox[1] for nid in nids])  # Chiều cao trung bình của head trong nhóm
        tolerance = nh * tolerance_ratio

        # Kiểm tra phần mở rộng phía trên/dưới so với hộp đầu nốt.
        gp_higher = (gbox[1] < nbox[1] - tolerance)
        gp_lower = (gbox[3] > nbox[3] + tolerance)

        if gp_higher and not gp_lower:
            # Cọng hướng lên: phần thân/nối kéo dài phía trên.
            temp_result[gp] = True
            for nid in nids:
                notes[nid].stem_up = True
            continue
        elif not gp_higher and gp_lower:
            # Cọng hướng xuống: phần thân/nối kéo dài phía dưới.
            temp_result[gp] = False
            for nid in nids:
                notes[nid].stem_up = False
            continue

        # Trường hợp mơ hồ: có thể là nhiều bè hoặc nốt không có cọng rõ ràng.
        if len(nids) == 1:
            nid = nids[0]
            new_group = get_possible_nearby_gid(notes[nid], group_map)
            if (new_group is not None) and check_valid_new_group(gp, new_group, group_map, max_x_diff_ratio):
                if new_group in temp_result:
                    # Nếu nhóm đích đã biết hướng, gán cho note hiện tại
                    notes[nid].stem_up = temp_result[new_group]

                # Hợp nhất nhãn trên bản đồ: đổi các pixel nhãn gp thành new_group
                group_map = np.where(group_map==gp, new_group, group_map)
                groups[new_group].append(nid)
                old_gp_nidx = groups[gp].index(nid)
                del groups[gp][old_gp_nidx]

    # Loại bỏ các nhãn trống (không còn note nào)
    groups = {gp: nids for gp, nids in groups.items() if len(nids) > 0}
    return groups, group_map


def check_group(group):
    notes = layers.get_layer('notes')

    if group.has_stem and group.stem_up is not None:
        # Kiểm tra chiều cao phần thân/stem so với bbox nhóm
        box = group.bbox
        ny_bound = np.array([(notes[nid].bbox[1], notes[nid].bbox[3]) for nid in group.note_ids])
        if group.stem_up:
            diff = abs(box[1] - np.min(ny_bound[:, 0]))
        else:
            diff = abs(box[3] - np.max(ny_bound[:, 1]))
        unit_size = get_unit_size(*get_center(box))
        if diff < unit_size:
            for nid in group.note_ids:
                notes[nid].invalid = True
            return False
    return True


def gen_groups(groups: Dict[int, List[int]], group_map: ndarray) -> Tuple[List[NoteGroup], ndarray]:
    # Lấy tham số/đối tượng cần thiết
    notes = layers.get_layer('notes')

    global grp_img
    grp_img = np.copy(group_map)
    grp_img = to_rgb_img(grp_img)

    ngs = []
    new_map = np.zeros_like(group_map) - 1
    idx = 0
    for gid, nids in groups.items():
        ng = NoteGroup()
        ng.id = idx
        ng.note_ids = nids
        gy, gx = np.where(group_map==gid)
        gbox = (int(np.min(gx)), int(np.min(gy)), int(np.max(gx)), int(np.max(gy)))
        nbox = np.array([notes[nid].bbox for nid in nids])
        nbox = (np.min(nbox[:, 0]), np.min(nbox[:, 1]), np.max(nbox[:, 2]), np.max(nbox[:, 3]))  # type: ignore

        cv2.rectangle(grp_img, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (255, 0, 0), 2)
        cv2.rectangle(grp_img, (nbox[0], nbox[1]), (nbox[2], nbox[3]), (0, 0, 255), 2)

        ng.bbox = gbox
        for nid in nids:
            notes[nid].note_group_id = idx

        if notes[nids[0]].stem_up is None:
            # Trường hợp: cọng có thể ở cả hai bên hoặc không có cọng
            nh = np.mean([notes[nid].bbox[3]-notes[nid].bbox[1] for nid in nids])  # Chiều cao nốt trung bình trong nhóm
            g_height = gbox[3] - gbox[1]
            n_height = nbox[3] - nbox[1]
                # Nếu chiều cao vùng liên thông lớn hơn đáng kể so với chiều cao head
            if abs(g_height-n_height) > nh // 5:
                #assert len(nids) > 1, nids
                ng.has_stem = True
            else:
                ng.has_stem = False
        elif notes[nids[0]].stem_up:
            ng.stem_up = True
            ng.has_stem = True
        else:
            ng.stem_up = False
            ng.has_stem = True

        n_types = [notes[nid].label for nid in nids]
        ng.all_same_type = all(nt==n_types[0] for nt in n_types)

        # Kiểm tra bổ sung sau khi tạo nhóm
        tar_track = notes[nids[0]].track
        tar_group = notes[nids[0]].group
        same_track = all(notes[nid].track==tar_track for nid in nids)
        same_group = all(notes[nid].group==tar_group for nid in nids)
        if not (same_track and same_group):
            y_mass_center = (gbox[1] + gbox[3]) / 2
            x_mass_center = (gbox[0] + gbox[2]) / 2
            st, _ = find_closest_staffs(x_mass_center, y_mass_center)  # type: ignore
            tar_track = st.track
            tar_group = st.group
            for nid in nids:
                notes[nid].track = st.track
                notes[nid].group = st.group

        ng.track = tar_track
        ng.group = tar_group

        new_map[group_map==gid] = idx
        ngs.append(ng)
        idx += 1
    return ngs, new_map


def save_note_groups_viz(out_dir: str) -> None:
    """Lưu ảnh trực quan hóa nhóm nốt vào thư mục đầu ra."""
    try:
        img = layers.get_layer('original_image')
        if img is None:
            return
        canvas = img.copy()
    except Exception:
        return

    groups = layers.get_layer('note_groups')
    notes = layers.get_layer('notes')
    if groups is None:
        _save(out_dir, 'step3_note_groups', canvas)
        return

    GROUP_COLORS = [
        (255, 80,  80),  (80, 255,  80),  (80, 100, 255),
        (255, 220,  0),  (0,  220, 220),  (220,   0, 220),
        (255, 160,  40), (40, 200, 180),  (180, 100, 255),
        (80, 255, 180),  (255, 100, 180), (180, 255,  60),
    ]

    for g_idx, grp in enumerate(groups):
        col = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
        if grp.bbox is not None:
            gx1, gy1, gx2, gy2 = grp.bbox
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), col, 2)
            gcx = int((gx1 + gx2) / 2); gcy = int((gy1 + gy2) / 2)
            stem_tag = '^' if grp.stem_up else ('v' if grp.stem_up is False else '?')
            has_s = 'S' if grp.has_stem else 'noS'
            same_t = '≡' if grp.all_same_type else '≠'
            tag = f"G{grp.id} T{grp.track}g{grp.group} {stem_tag}{has_s}{same_t} n={len(grp.note_ids)}"
            _text(canvas, tag, (gx1 + 2, gy1 - 5), col, scale=0.35)

        for nid in grp.note_ids:
            note = notes[nid]
            if note.bbox is None:
                continue
            nx1, ny1, nx2, ny2 = note.bbox
            _rect(canvas, note.bbox, col, thickness=1)
            if grp.bbox is not None:
                ncx = int((nx1 + nx2) / 2); ncy = int((ny1 + ny2) / 2)
                cv2.line(canvas, (gcx, gcy), (ncx, ncy), col, 1, cv2.LINE_AA)
            _text(canvas, f"n{nid}", (nx1, ny1 - 2), col, scale=0.28)

    total_groups = len(groups)
    total_notes = sum(len(g.note_ids) for g in groups)
    info = f"Groups: {total_groups}  Notes in groups: {total_notes}"
    _text(canvas, info, (6, canvas.shape[0] - 8), C['white'], scale=0.45)
    _save(out_dir, 'step3_note_groups', canvas)


def post_check_groups(groups):
    # Lấy tham số/đối tượng cần thiết
    notes = layers.get_layer('notes')

    for grp in groups:
        if len(grp.note_ids) != 2:
            # Hiện chỉ hỗ trợ tách trường hợp gom nhầm khi nhóm có đúng 2 nốt.
            continue


def extract() -> Tuple[List[NoteGroup], ndarray]:
    # Bắt đầu xử lý
    logger.debug("Grouping noteheads")
    groups, group_map = group_noteheads()

    logger.debug("Phân tích hướng cọng")
    groups, group_map = parse_stem_direction(groups, group_map)

    logger.debug("Khởi tạo các nhóm nốt")
    groups, group_map = gen_groups(groups, group_map)  # type: ignore

    logger.debug("Post check notes in groups")

    return groups, group_map  # type: ignore


def predict_symbols():
    pred = layers.get_layer('celfs_keys_pred')  # sfn -> dấu thăng, dấu giáng, dấu hóa
    #pred = layers.get_layer('stems_rests_pred')
    bboxes = get_bbox(pred)
    bboxes = merge_nearby_bbox(bboxes, 15)
    bboxes = rm_merge_overlap_bbox(bboxes)

    img = np.ones(pred.shape+(3,), dtype=np.uint8) * 255
    idx = np.where(pred>0)
    img[idx[0], idx[1]] = 0
    # Import `predict` theo kiểu import trễ để tránh vòng lặp import khi nạp module.
    from oemer.inference import predict
    for box in bboxes:
        region = pred[box[1]:box[3], box[0]:box[2]]
        region[region>0] = 255
        pp = predict(region, "sfn")
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img, pp, (box[2]+2, box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    return img


def draw_notes(notes, ori_img):
    img = ori_img.copy()
    img = np.array(img)
    for note in notes:
        x1, y1, x2, y2 = note.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if getattr(note, 'has_dot', False):
            # Đánh dấu nốt có chấm (dotted) để trực quan
            cv2.putText(img, "DOT", (x2 + 2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    return img


# --- Lightweight grouping helpers moved from postproc ---
def _connected_components(points: List[Tuple[int, int]], threshold: float) -> List[List[int]]:
    if not points:
        return []
    pts = np.array(points)
    n = len(points)
    visited = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            du = pts[u][None, :] - pts
            dists = np.linalg.norm(du, axis=1)
            for j in range(n):
                if not visited[j] and dists[j] <= threshold:
                    visited[j] = True
                    stack.append(j)
        comps.append(comp)
    return comps


def group_notes(centers: List[Tuple[int, int]], unit_size: int = 10) -> List[List[int]]:
    """Gom các tâm nốt thành hợp âm/nhóm bằng ngưỡng khoảng cách đơn giản.

    Trả về danh sách nhóm, mỗi nhóm là danh sách các chỉ số vào `centers`.
    """
    if not centers:
        return []
    # Ngưỡng gom: ưu tiên độ gần theo phương dọc, dùng hệ số theo unit_size.
    threshold = unit_size * 1.6
    comps = _connected_components(centers, threshold)
    # Chuẩn hóa kết quả về danh sách nhóm, mỗi nhóm là danh sách chỉ số.
    groups: List[List[int]] = []
    for comp in comps:
        groups.append(comp)
    return groups


def extract_note_centers_from_classmap(class_map: np.ndarray, class_value: int = 2) -> List[Tuple[int, int]]:
    """Trích xuất tâm nốt đơn giản từ bản đồ lớp, khi giá trị lớp tương ứng là notehead.

    Trả về danh sách toạ độ (x, y) của tâm nốt (pixel centers).
    """
    centers: List[Tuple[int, int]] = []
    try:
        if class_map.ndim == 2:
            mask = (class_map == class_value).astype(np.uint8)
        else:
            # Nếu là bản đồ đa kênh thì lấy kênh có xác suất cao nhất.
            mask = (np.argmax(class_map, axis=-1) == class_value).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            centers.append((int(round(cx)), int(round(cy))))
    except Exception:
        pass
    return centers


def advanced_group_notes(centers: List[Tuple[int, int]], unit_size: int = 10) -> List[Dict[str, Any]]:
    """Gom tâm nốt thành ứng viên hợp âm và tách các voice khi có sự phân tách theo chiều dọc.

    Trả về danh sách dict nhóm: {indices: [...], centroid: (x,y), voices: {0:[idx],1:[idx]}}
    """
    groups_idx = group_notes(centers, unit_size=unit_size)
    results: List[Dict[str, Any]] = []
    pts = np.array(centers) if centers else np.zeros((0, 2))
    for comp in groups_idx:
        comp_pts = pts[comp] if len(comp) > 0 else np.zeros((0, 2))
        centroid = (int(np.mean(comp_pts[:, 0])) if len(comp_pts) else 0, int(np.mean(comp_pts[:, 1])) if len(comp_pts) else 0)
        # Nếu độ trải dọc lớn, tách thành 2 bè theo trung vị trục y.
        voices = {0: comp, 1: []}
        if len(comp_pts) > 1:
            ys = comp_pts[:, 1]
            spread = ys.max() - ys.min()
            if spread > unit_size * 0.6:
                median_y = np.median(ys)
                up = [comp[i] for i, y in enumerate(ys) if y <= median_y]
                down = [comp[i] for i, y in enumerate(ys) if y > median_y]
                voices = {0: up, 1: down}
        results.append({"indices": comp, "centroid": centroid, "voices": voices})
    return results


def apply_music_constraints(groups: List[List[int]], centers: Optional[List[Tuple[int, int]]] = None, clefs: Optional[List[Any]] = None, key_signature: Optional[Any] = None) -> List[List[int]]:
    """Áp dụng một số ràng buộc âm nhạc đơn giản lên các nhóm nốt.

    Đây là hàm nhẹ, chủ yếu là heuristic; hiện tại chỉ trả về nhóm không đổi
    nhưng đánh dấu những nơi mà ràng buộc có thể áp dụng (placeholder).
    """
    if centers is None:
        return groups
    grouped: List[List[int]] = []
    # Cố gắng dùng thông tin staff nếu có để gán chỉ số khuông
    staff_centers = None
    try:
        from oemer import layers as _layers
        staffs = None
        try:
            staffs = _layers.get_layer('staffs')
        except Exception:
            staffs = None
        if staffs is not None and len(staffs) > 0:
            staff_centers = [getattr(s, 'Center', None) or getattr(s, 'center', None) or getattr(s, 'center_y', None) for s in staffs]
    except Exception:
        staff_centers = None

    for grp in groups:
        if len(grp) <= 1:
            grouped.append(grp)
            continue
        xs = [centers[i][0] for i in grp]
        ys = [centers[i][1] for i in grp]
        # Nếu có tâm khuông, tách theo khuông gần nhất.
        if staff_centers:
            assignment = {}
            for idx in grp:
                y = centers[idx][1]
                nearest = min(range(len(staff_centers)), key=lambda k: abs(staff_centers[k] - y))
                assignment.setdefault(nearest, []).append(idx)
            for k in sorted(assignment.keys()):
                grouped.append(assignment[k])
            continue

        span = max(ys) - min(ys)
        median_gap = np.median(np.diff(sorted(ys))) if len(ys) > 1 else 0
        if span > 3 * (median_gap if median_gap > 0 else 1):
            median_y = int(np.median(ys))
            low = [i for i in grp if centers[i][1] <= median_y]
            high = [i for i in grp if centers[i][1] > median_y]
            if low:
                grouped.append(low)
            if high:
                grouped.append(high)
        else:
            grouped.append(grp)

    grouped.sort(key=lambda g: np.mean([centers[i][0] for i in g]) if g else 0)
    return grouped

