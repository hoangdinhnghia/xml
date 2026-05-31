"""
Module trích xuất và căn chỉnh staffline.

Chứa các lớp `Line` và `Staff` cùng pipeline trích xuất:
- Xác định vùng chứa staff (init_zones)
- Phát hiện các hàng chứa line bằng histogram và find_peaks
- Lọc peak, ánh xạ pixel vào `Line`
- Gom `Line` thành `Staff` và căn chỉnh giữa các vùng (align_staffs)
- Suy luận số track (further_infer_track_nums)

Ghi chú: các bước được triển khai bằng các heuristic hình học và tối ưu hóa vector hóa (NumPy).
"""

import enum
import pickle
import typing
from typing import List, Any, Tuple, Union
from typing_extensions import Self
from functools import cached_property  # Hiện đại hóa cơ chế cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import bool_, ndarray
from scipy.signal import find_peaks

from oemer import layers
from oemer import exceptions as E
from oemer.utils import get_logger, C, _overlay, _rect, _text, _legend, _save
from oemer.bbox import BBox, find_lines, get_bbox, get_center

logger = get_logger(__name__)


class LineLabel(enum.Enum):
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4


class Line:
    """Đại diện cho một dòng staffline đơn lẻ.

    Lưu trữ tập điểm ảnh thuộc dòng và cung cấp các thuộc tính hình học
    (tâm, biên, slope) dưới dạng `cached_property` để tái sử dụng.
    """
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []
        self.label: Union[LineLabel, None] = None

    def add_point(self, y: int, x: int) -> None:
        self.points.append((y, x))
        # Tự động xóa các bộ nhớ đệm đã tính khi có điểm mới.
        self.__dict__.pop('y_center', None)
        self.__dict__.pop('y_upper', None)
        self.__dict__.pop('y_lower', None)
        self.__dict__.pop('x_center', None)
        self.__dict__.pop('x_left', None)
        self.__dict__.pop('x_right', None)
        self.__dict__.pop('slope', None)

    @cached_property
    def y_center(self) -> float:
        if not self.points:
            return 0.0
        return float(np.mean([point[0] for point in self.points]))

    @cached_property
    def y_upper(self) -> float:
        if not self.points:
            return 0.0
        return float(np.min([point[0] for point in self.points]))

    @cached_property
    def y_lower(self) -> float:
        if not self.points:
            return 0.0
        return float(np.max([point[0] for point in self.points]))

    @cached_property
    def x_center(self) -> float:
        if not self.points:
            return 0.0
        return float(np.mean([point[1] for point in self.points]))

    @cached_property
    def x_left(self) -> float:
        if not self.points:
            return 0.0
        return float(np.min([point[1] for point in self.points]))  

    @cached_property
    def x_right(self) -> float:
        if not self.points:
            return 0.0
        return float(np.max([point[1] for point in self.points]))

    @cached_property
    def slope(self) -> float:
        """Ước lượng độ nghiêng của dòng bằng xấp xỉ tuyến tính.

        Sử dụng `np.polyfit(xs, ys, 1)` trên tọa độ điểm; với trường hợp suy biến
        (ví dụ tất cả xs trùng nhau) trả về giá trị đại diện.
        """
        if len(self.points) < 2:
            return 0.0
        
        points = np.array(self.points)
        xs = points[:, 1]
        ys = points[:, 0]
        
        # Bảo vệ chống lỗi suy biến ma trận khi đường thẳng đứng (xs trùng nhau)
        if np.all(xs == xs[0]):
            return 999.0  # Đại diện cho đường thẳng đứng vô hạn độ dốc
            
        try:
            m, _ = np.polyfit(xs, ys, 1)
            return float(m)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def __lt__(self, line: Self) -> bool:
        return self.y_center < line.y_center

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return "Line(\n" \
            f"\tPoint count: {len(self.points)}\n" \
            f"\tCenter: {self.y_center}\n" \
            f"\tUpper bound: {self.y_upper}\n" \
            f"\tLower bound: {self.y_lower}\n" \
            f"\tLabel: {self.label}\n" \
            f"\tSlope: {self.slope}\n" \
            ")\n"


class Staff:
    """Đại diện cho một khuông nhạc (gồm tối đa 5 dòng).

    Lưu các `Line` thành phần và tính các thuộc tính tổng hợp như `unit_size`,
    tâm, biên và slope trung bình. Hỗ trợ nhân bản (`duplicate`) để nội suy khi
    một phần khuông bị thiếu trong một vùng.
    """
    def __init__(self) -> None:
        self.lines: List[Line] = []
        self.track: Union[int, None] = None
        self.group: Union[int, None]  = None
        self.is_interp: bool = False

    def add_line(self, line: Line) -> None:
        self.lines.append(line)
        # Đặt lại bộ nhớ đệm khi cấu trúc dòng kẻ thay đổi.
        self.__dict__.pop('y_center', None)
        self.__dict__.pop('y_upper', None)
        self.__dict__.pop('y_lower', None)
        self.__dict__.pop('x_center', None)
        self.__dict__.pop('x_left', None)
        self.__dict__.pop('x_right', None)
        self.__dict__.pop('unit_size', None)
        self.__dict__.pop('slope', None)

    @property
    def y_center(self) -> float:
        if 'y_center' in self.__dict__:
            return self.__dict__['y_center']
        if not self.lines:
            return 0.0
        self.__dict__['y_center'] = float(np.mean([line.y_center for line in self.lines]))
        return self.__dict__['y_center']

    @y_center.setter
    def y_center(self, val):
        self.__dict__['y_center'] = val

    @property
    def y_upper(self) -> float:
        if 'y_upper' in self.__dict__:
            return self.__dict__['y_upper']
        if not self.lines:
            return 0.0
        self.__dict__['y_upper'] = float(np.min([line.y_upper for line in self.lines]))
        return self.__dict__['y_upper']

    @y_upper.setter
    def y_upper(self, val):
        self.__dict__['y_upper'] = val

    @property
    def y_lower(self) -> float:
        if 'y_lower' in self.__dict__:
            return self.__dict__['y_lower']
        if not self.lines:
            return 0.0
        self.__dict__['y_lower'] = float(np.max([line.y_lower for line in self.lines]))
        return self.__dict__['y_lower']

    @y_lower.setter
    def y_lower(self, val):
        self.__dict__['y_lower'] = val

    @property
    def x_center(self) -> float:
        if 'x_center' in self.__dict__:
            return self.__dict__['x_center']
        if not self.lines:
            return 0.0
        self.__dict__['x_center'] = float(np.mean([line.x_center for line in self.lines]))
        return self.__dict__['x_center']

    @x_center.setter
    def x_center(self, val):
        self.__dict__['x_center'] = val

    @property
    def x_left(self) -> float:
        if 'x_left' in self.__dict__:
            return self.__dict__['x_left']
        if not self.lines:
            return 0.0
        self.__dict__['x_left'] = float(np.min([line.x_left for line in self.lines]))
        return self.__dict__['x_left']

    @x_left.setter
    def x_left(self, val):
        self.__dict__['x_left'] = val

    @property
    def x_right(self) -> float:
        if 'x_right' in self.__dict__:
            return self.__dict__['x_right']
        if not self.lines:
            return 0.0
        self.__dict__['x_right'] = float(np.max([line.x_right for line in self.lines]))
        return self.__dict__['x_right']

    @x_right.setter
    def x_right(self, val):
        self.__dict__['x_right'] = val

    @cached_property
    def unit_size(self) -> float:
        """Khoảng cách chuẩn giữa các dòng trong một `Staff`.

        Tính trung bình các khoảng cách giữa các tâm dòng liên tiếp.
        """
        if len(self.lines) < 2:
            return 0.0
        centers = [line.y_center for line in self.lines]
        gaps = [centers[i] - centers[i-1] for i in range(1, len(self.lines))]
        return float(np.mean(gaps))

    @property
    def incomplete(self) -> bool:
        return len(self.lines) != 5

    @cached_property
    def slope(self) -> float:
        if not self.lines:
            return 0.0
        return float(np.mean([l.slope for l in self.lines]))

    def duplicate(self, x_offset=0, y_offset=0):
        st = Staff()
        for line in self.lines:
            new_l = Line()
            for y, x in line.points:
                new_l.add_point(y+y_offset, x+x_offset)
            st.add_line(new_l)
        return st

    def __lt__(self, st):
        return self.y_center < st.y_center

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        return "Staff(\n" \
            f"\tLines: {len(self.lines)}\n" \
            f"\tCenter: {self.y_center}\n" \
            f"\tUpper bound: {self.y_upper}\n" \
            f"\tLower bound: {self.y_lower}\n" \
            f"\tUnit size: {self.unit_size}\n" \
            f"\tTrack: {self.track}\n" \
            f"\tGroup: {self.group}\n" \
            f"\tIs interpolation: {self.is_interp}\n" \
            f"\tSlope: {self.slope}\n" \
            ")\n"

    def __sub__(self, st: Union[List[int], 'Staff']) -> float:
        if isinstance(st, Staff):
            x, y = st.x_center, st.y_center
        else:
            x, y = st
        x_dist = (x - self.x_center) ** 2
        y_dist = (y - self.y_center) ** 2
        return (x_dist + y_dist) ** 0.5


def init_zones(staff_pred: ndarray, splits: int) -> Tuple[ndarray, int, int, int]:
    """Xác định các vùng cột (zones) chứa staff để xử lý cục bộ.

    Trả về mảng zone (mỗi zone là range cột), left_bound, right_bound, bottom_bound.

    Ý tưởng:
    - Dùng histogram theo trục x để tìm biên thực sự có staff.
    - Mở rộng biên bằng khoảng đệm để tránh cắt cụt nét.
    - Chia đều miền cột thành nhiều zone để phát hiện ổn định hơn theo cục bộ.
    """
    ys, xs = np.where(staff_pred > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([], dtype=object), 0, staff_pred.shape[1], staff_pred.shape[0]

    accum_x = np.sum(staff_pred, axis=0)
    mean_accum = np.mean(accum_x)
    if mean_accum > 0:
        accum_x = accum_x / mean_accum
        
    half = round(len(accum_x) / 2)
    right_bound = min(max(xs) + 50, staff_pred.shape[1])
    left_bound = max(min(xs) - 50, 0)
    
    for i in range(half+10, len(accum_x)):
        if np.mean(accum_x[i-10:i]) < 0.1:
            right_bound = i
            break
    for i in range(half-10, 0, -1):
        if np.mean(accum_x[i:i+10]) < 0.1:
            left_bound = i
            break

    bottom_bound = min(max(ys) + 100, len(staff_pred))
    
    # Chia mảng đều bằng np.array_split để tránh phần dư dồn vào đoạn cuối.
    x_coords = np.arange(left_bound, right_bound)
    split_arrays = np.array_split(x_coords, splits)
    zones = [range(arr[0], arr[-1] + 1) for arr in split_arrays if len(arr) > 0]
    
    return np.array(zones, dtype=object), left_bound, right_bound, bottom_bound


def extract(
    splits: int = 8, 
    line_threshold: float = 0.8, 
    horizontal_diff_th: float = 0.1, 
    unit_size_diff_th: float = 0.1, 
    barline_min_degree: int = 75
) -> Tuple[ndarray, ndarray]:
    """Điều phối toàn bộ quy trình trích xuất staff trên ảnh hiện tại.

    - Chia ảnh theo `splits` vùng bằng `init_zones`.
    - Với mỗi zone gọi `extract_part` để lấy các Staff cục bộ.
    - Căn chỉnh các staff giữa các zone bằng `align_staffs`.
    - Lọc các staff không thống nhất về `unit_size` hoặc tâm.
    Trả về mảng staff đã lọc và các zone.

    Chiến lược tổng thể:
    - Ưu tiên độ ổn định toàn trang: phát hiện cục bộ, sau đó hợp nhất liên-zone.
    - Chỉ giữ các hàng staff có hình học đồng nhất giữa các zone.
    """
    staff_pred = layers.get_layer('staff_pred')

    zones, *_ = init_zones(staff_pred, splits=splits)
    all_staffs: List[List[Staff]] = []
    for rr in zones:
        logger.debug(f"Processing zone: {rr[0]} to {rr[-1]}")
        rr = np.array(rr, dtype=np.int64)
        staffs = extract_part(staff_pred[:, rr], x_offset=rr[0], line_threshold=line_threshold)
        if staffs is not None:
            all_staffs.append(staffs)
            
    if not all_staffs:
        return np.array([]), zones
        
    aligned_staffs: np.ndarray = align_staffs(all_staffs)

    num_track = further_infer_track_nums(aligned_staffs, min_degree=barline_min_degree)
    logger.debug(f"Tracks: {num_track}")
    for col_sts in aligned_staffs:
        for idx, st in enumerate(col_sts):
            st.track = idx % num_track
            st.group = idx // num_track

    if not all([len(staff) == len(aligned_staffs[0]) for staff in aligned_staffs]):
        raise E.InvalidStaffException("Staff alignment count mismatch across zones.")

    norm = lambda data: np.abs(np.array(data) / (np.mean(data) if np.mean(data) > 0 else 1) - 1)
    valid_staffs: list[list[Staff]] = []
    for staffs in aligned_staffs.T:
        line_num = [len(staff.lines) for staff in staffs]
        if len(set(line_num)) != 1:
            logger.warning(f"Some stafflines contain mismatched lines: {line_num}")
            continue

        centers = np.array([staff.y_center for staff in staffs])
        if not np.all(norm(centers) < horizontal_diff_th):
            logger.warning(f"Centers of staff parts at the same row not aligned.")
            continue

        unit_size = np.array([staff.unit_size for staff in staffs])
        if not np.all(norm(unit_size) < unit_size_diff_th):          
            logger.warning(f"Unit sizes not consistent.")
            continue
        valid_staffs.append(staffs)

    return np.array(valid_staffs).T, zones


def extract_part(pred: ndarray, x_offset: int, line_threshold: float = 0.8) -> List[Staff]:
    """Trích xuất danh sách `Staff` trong một vùng cột nhỏ.

    Hàm gọi `extract_line` để lấy các `Line` trong vùng, sau đó gom mỗi 5
    `Line` liên tiếp thành một `Staff`. Nếu vùng có ít hơn 5 line trả về None.

    Quy ước: thứ tự dòng trong mỗi staff là FIRST..FIFTH.
    """
    lines, _ = extract_line(pred, x_offset=x_offset, line_threshold=line_threshold)

    if len(lines) < 5:
        return None  # type: ignore

    staffs = []
    line_buffer: Any = []
    for idx, line in enumerate(lines):
        lid = idx % 5
        assert line.label == LineLabel(lid), f"{line}, {lid}, {idx}"
        if lid == 0 and line_buffer:
            staff = Staff()
            assert len(line_buffer) == 5, len(line_buffer)
            for l in line_buffer:
                staff.add_line(l)
            staffs.append(staff)
            line_buffer = []
        line_buffer.append(line)

    staff = Staff()
    for l in line_buffer:
        staff.add_line(l)
    staffs.append(staff)

    return staffs


def extract_line(pred: ndarray, x_offset: int, line_threshold: float = 0.8) -> Tuple[ndarray, ndarray]:
    """Phát hiện các `Line` trong một vùng cột.

    - Tạo histogram theo hàng (số pixel staff trên mỗi hàng).
    - Chuẩn hoá bằng z-score và tìm peak bằng `find_peaks`.
    - Lọc peak bằng `filter_line_peaks`.
    - Vector hoá việc ánh xạ pixel vào center để tạo danh sách `Line`.
    Trả về mảng `Line` hợp lệ và vectơ chuẩn hoá `norm`.

    Bộ lọc nhiễu gồm 3 lớp:
    - Ngưỡng peak theo histogram chuẩn hóa.
    - Ràng buộc khoảng cách điểm tới peak gần nhất (max_gap).
    - Chỉ giữ các peak được xác thực theo nhóm 5 dòng.
    """
    count = np.zeros(len(pred), dtype=np.uint16)
    sub_ys, sub_xs = np.where(pred > 0)
    for y in sub_ys:
        count[y] += 1

    count = np.insert(count, [0, len(count)], [0, 0])
    std_val = np.std(count)
    norm = (count - np.mean(count)) / (std_val if std_val > 0 else 1)
    centers, _ = find_peaks(norm, height=line_threshold, distance=8, prominence=1)
    centers -= 1
    norm = norm[1:-1]
    valid_centers, groups = filter_line_peaks(centers, norm)

    cc = centers[valid_centers]
    if len(cc) < 3:
        max_gap = 10.0
    else:
        max_gap = np.mean(np.sort(cc[1:] - cc[:-1])[:3])
        
    lines = [Line() for _ in range(len(centers))]
    
    # Tối ưu vector hóa (NumPy broadcasting): thay vòng lặp theo từng điểm ảnh.
    if len(sub_ys) > 0 and len(centers) > 0:
        # Tính khoảng cách tuyệt đối từ mọi điểm y đến toàn bộ centers cùng lúc.
        distances = np.abs(sub_ys_expanded := sub_ys[:, np.newaxis] - centers)
        closest_cen_indices = np.argmin(distances, axis=1)
        assigned_centers = centers[closest_cen_indices]
        
        # Điều kiện lọc vector hóa hiệu năng cao
        valid_mask = (
            valid_centers[closest_cen_indices] & 
            (norm[sub_ys] > min(line_threshold, 1.2)) & 
            (np.abs(sub_ys - assigned_centers) < max_gap)
        )
        
        # Chỉ trích xuất và gán các điểm hợp lệ
        valid_ys = sub_ys[valid_mask]
        valid_xs = sub_xs[valid_mask]
        valid_cen_ids = closest_cen_indices[valid_mask]
        
        for y, x, cen_idx in zip(valid_ys, valid_xs, valid_cen_ids):
            lines[cen_idx].add_point(y, x + x_offset)

    # Gán nhãn FIRST..FIFTH theo từng nhóm peak để gom thành khuông ở bước sau.
    last_group = groups[0] if len(groups) > 0 else 0
    cur_line_id = 0
    pack = sorted(zip(lines, valid_centers, groups), key=lambda obj: obj[0].y_center)
    for line, valid, grp in pack:
        if not valid:
            continue
        if grp != last_group:
            cur_line_id = 0
            last_group = grp

        line.label = LineLabel(cur_line_id)
        cur_line_id += 1

    lines = np.array(lines)[valid_centers]
    return lines, norm


def filter_line_peaks(peaks: ndarray, norm: ndarray, max_gap_ratio: float = 1.5) -> Tuple[ndarray, List[int]]:
    """Lọc các đỉnh (peaks) tìm được để hình thành nhóm các dòng hợp lệ.

    Heuristic:
    - Ước lượng khoảng cách chuẩn tạm thời `approx_unit` từ các gap nhỏ nhất.
    - Thiết lập `max_gap = approx_unit * max_gap_ratio` để phân nhóm.
    - Loại các nhóm có ít hơn 5 peak; với nhóm >5 chọn 5 peak có tín hiệu mạnh hơn.
    Trả về mảng boolean `valid_peaks` và danh sách nhãn nhóm tương ứng.

    Lý do chọn 5 peak mạnh nhất khi dư peak:
    - Một staff tiêu chuẩn có 5 dòng.
    - Peak dư thường do nhiễu hoặc giao cắt ký hiệu âm nhạc khác.
    """
    if len(peaks) == 0:
        return np.array([], dtype=bool), []
        
    valid_peaks = np.array([True for _ in range(len(peaks))])

    for idx, p in enumerate(peaks):
        if norm[p] > 15:
            valid_peaks[idx] = False

    gaps = peaks[1:] - peaks[:-1]
    count = max(5, round(len(peaks) * 0.2))
    approx_unit = np.mean(np.sort(gaps)[:count]) if len(gaps) > 0 else 10.0
    max_gap = approx_unit * max_gap_ratio

    ext_peaks = [peaks[0]-max_gap-1] + list(peaks)
    groups = []
    group = -1
    for i in range(1, len(ext_peaks)):
        if ext_peaks[i] - ext_peaks[i-1] > max_gap:
            group += 1
        groups.append(group)

    groups.append(groups[-1]+1)
    cur_g = groups[0]
    count = 1
    for idx in range(1, len(groups)):
        group = groups[idx]
        if group == cur_g:
            count += 1
            continue

        if count < 5:
            valid_peaks[idx-count:idx] = False
        elif count > 5:
            cand_peaks = peaks[idx-count:idx]
            head_part = cand_peaks[:5]
            tail_part = cand_peaks[-5:]
            if sum(norm[head_part]) > sum(norm[tail_part]):
                valid_peaks[idx-count+5:idx] = False
            else:
                valid_peaks[idx-count:idx-5] = False

        cur_g = group
        count = 1
    return valid_peaks, groups[:-1]


def align_staffs(staffs: List[List[Staff]], max_dist_ratio: int = 3) -> ndarray:
    """Căn chỉnh các `Staff` giữa nhiều zone.

    - Xây lưới với số cột = max số staff ở các zone.
    - Gán staff đã tồn tại vào vị trí tương ứng.
    - Với ô thiếu, tìm staff tham chiếu gần nhất (`nearby_sts`) và nội suy
      (sao chép + dịch chuyển theo tỉ lệ) để điền vào ô thiếu.
    - Các staff nội suy được gán `is_interp=True`.
    Trả về mảng `Staff` đã căn chỉnh.

    Đây là bước hợp nhất quan trọng:
    - Biến danh sách staff rời rạc theo từng zone thành lưới staff đồng bộ.
    - Bù thiếu dữ liệu bằng nội suy hình học để các bước sau không bị đứt mạch.
    """
    len_types = set(len(st_part) for st_part in staffs)
    if len(len_types) == 1:
        return np.array(staffs)

    max_len = max(len_types) if len_types else 0
    grid = np.zeros((len(staffs), max_len), dtype=object)
    for idx, st_part in enumerate(staffs):
        if len(st_part) == max_len:
            grid[idx] = np.array(st_part)

    def get_nearby_sts(j, row):
        # Lấy tối đa 2 khuông gần nhất theo trục zone trong cùng một hàng lưới.
        dists = [(idx, abs(idx-j)) for idx in range(len(row))]
        dists = sorted(dists, key=lambda it: it[1])
        idxs = [it[0] for it in dists]

        nearby_sts = []
        for near_idx in idxs:
            if isinstance(row[near_idx], Staff):
                nearby_sts.append((near_idx, row[near_idx]))
            if len(nearby_sts) >= 2:
                break
        return nearby_sts

    def get_nearest_ori_st(ref_st, ori_st_col):
        # Ưu tiên khuông gốc nếu y_center đủ gần khuông tham chiếu.
        max_dist = ref_st.unit_size * max_dist_ratio
        for st in ori_st_col:
            dist = abs(st.y_center - ref_st.y_center)
            if dist < max_dist:
                return st
        return None

    for i in range(max_len):
        row = grid[:, i]
        for j, obj in enumerate(row):
            if isinstance(obj, Staff):
                continue

            ori_st_part = staffs[j]
            sts = get_nearby_sts(j, row)

            if len(sts) == 0:
                continue

            ori_st = get_nearest_ori_st(sts[0][1], ori_st_part)
            if ori_st is not None:
                grid[j, i] = ori_st
                continue

            if len(sts) == 1:
                # Chỉ có 1 mốc: ngoại suy theo dịch chuyển ngang giữa các vùng zone.
                ref_idx, ref_st = sts[0]
                width = ref_st.x_right - ref_st.x_left
                x_offset = width * (j - ref_idx)
                new_st = ref_st.duplicate(x_offset=x_offset)
            else:
                # Có 2 mốc: nội suy/ngoại suy tuyến tính cả x và y.
                (idx1, ref1), (idx2, ref2) = sts

                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                    ref1, ref2 = ref2, ref1
                if j < idx1:
                    r1, r2 = idx1-j, idx2-idx1
                    x_offset = -(ref2.x_center-ref1.x_center) * (r1 / r2)
                    y_offset = -(ref2.y_center-ref1.y_center) * (r1 / r2)
                    new_st = ref1.duplicate(x_offset=x_offset, y_offset=y_offset)
                elif idx1 <= j < idx2:
                    r1, r2 = j-idx1, idx2-idx1
                    x_offset = (ref2.x_center-ref1.x_center) * (r1 / r2)
                    y_offset = (ref2.y_center-ref1.y_center) * (r1 / r2)
                    new_st = ref1.duplicate(x_offset=x_offset, y_offset=y_offset)
                else:
                    r1, r2 = idx2-idx1, j-idx2
                    x_offset = (ref2.x_center-ref1.x_center) * (r2 / r1)
                    y_offset = (ref2.y_center-ref1.y_center) * (r2 / r1)
                    new_st = ref2.duplicate(x_offset=x_offset, y_offset=y_offset)
            new_st.is_interp = True
            grid[j, i] = new_st
    return grid


def further_infer_track_nums(staffs: ndarray, min_degree: int = 75) -> int:
    """Ước lượng số track trong bản nhạc dựa trên phát hiện vạch nhịp dọc.

    - Kết hợp các lớp dự đoán (`symbols_pred`, `stems_rests_pred`,
      `notehead_pred`, `clefs_keys_pred`) để làm nổi bật vạch dọc.
    - Tìm các đường thẳng đứng và lọc theo góc (`min_degree`).
    - So sánh chiều cao vạch với `unit_size` của staff để ước lượng số track bằng
      các quy tắc heuristic.
        Trả về số track (int).

        Trực giác heuristic:
        - Nếu có nhiều vạch dọc cao vượt ngưỡng theo unit_size,
            khả năng bản nhạc có nhiều track chồng dọc sẽ tăng.
    """
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    clefs = layers.get_layer('clefs_keys_pred')

    # Khử các lớp gây nhiễu để làm nổi bật thành phần vạch dọc.
    mix = symbols - stems - notehead - clefs
    mix[mix < 0] = 0

    lines = find_lines(mix)
    lines = filter_lines(lines, staffs, min_degree=min_degree)
    bmap = get_barline_map(symbols, lines) + stems
    bmap[bmap > 1] = 1

    ker = np.ones((5, 2), dtype=np.uint8)
    ext_bmap = cv2.erode(cv2.dilate(bmap.astype(np.uint8), ker), ker)
    bboxes = get_bbox(ext_bmap)

    # Đặc trưng chính: tỉ lệ chiều cao barline so với unit_size cục bộ.
    h_ratios = []
    for box in bboxes:
        h = box[3] - box[1]
        unit_size = naive_get_unit_size(staffs, *get_center(box))
        if h > unit_size:
            h_ratios.append(h / unit_size)
    h_ratios = np.array(h_ratios)

    num_track = 1
    factor = 10
    for i in range(1, 10):
        valid_h = len(h_ratios[h_ratios > factor * i])
        if valid_h * (i + 1) > staffs.shape[1]:
            num_track += 1
        else:
            break
    return num_track


def get_degree(line: BBox) -> float:
    """Tính góc (độ) của một đoạn bbox, dùng để xác định barline gần thẳng đứng."""
    return float(np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0])))


def filter_lines(lines: List[BBox], staffs: ndarray, min_degree: int = 75) -> List[BBox]:
    """Lọc các đường tìm được theo phạm vi nằm trong biên staff và theo góc.

    Bỏ qua các đường có góc nhỏ hơn `min_degree` hoặc nằm ngoài vùng bao của
    các staff đã phát hiện.

    Mục tiêu: chỉ giữ những line có khả năng là barline hợp lệ trong hệ staff hiện tại.
    """
    if staffs.size == 0:
        return []
        
    min_y = 9999999
    min_x = 9999999
    max_y = 0
    max_x = 0
    for st in staffs.ravel():
        min_y = min(min_y, st.y_upper)
        min_x = min(min_x, st.x_left)
        max_y = max(max_y, st.y_lower)
        max_x = max(max_x, st.x_right)

    # Danh sách ứng viên cuối sau lọc hình học và lọc theo bao khuông.
    cands = []
    for line in lines:
        degree = get_degree(line)
        if degree < min_degree:
            continue

        if line[1] < min_y \
                or line[3] > max_y \
                or line[0] < min_x \
                or line[2] > max_x:
            continue

        cands.append(line)
    return cands


def get_barline_map(symbols: ndarray, bboxes: List[BBox]) -> ndarray:
    """Xây bản đồ barline từ các bounding box bằng cách cộng vùng ký hiệu.

    Trả về ảnh nhị phân, trong đó vùng barline được gán 1.

    Thay vì tô cứng bbox = 1, hàm cộng năng lượng từ symbols trong bbox
    để giảm ảnh hưởng của box rỗng hoặc box lệch.
    """
    img = np.zeros_like(symbols)
    for box in bboxes:
        box = list(box)
        if box[2]-box[0] == 0:
            box[2] += 1
        img[box[1]:box[3], box[0]:box[2]] += symbols[box[1]:box[3], box[0]:box[2]]
    img[img>1] = 1
    return img


def naive_get_unit_size(staffs: ndarray, x: int, y: int) -> float:
    """Ước lượng unit_size gần nhất cho một điểm (x,y) bằng cách tìm staff
    gần nhất trong danh sách đã căn chỉnh và trả về unit_size của nó.

    Đây là heuristic đơn giản nhưng hiệu quả cho bước hậu xử lý,
    vì không cần nội suy phức tạp theo biến dạng toàn trang.
    """
    flat_staffs = staffs.ravel()
    if len(flat_staffs) == 0:
        return 10.0

    def dist(st: Staff) -> float:
        x_diff = st.x_center - x
        y_diff = st.y_center - y
        return float(x_diff ** 2 + y_diff ** 2)

    dists = [(st.unit_size, dist(st)) for st in flat_staffs]
    dists = sorted(dists, key=lambda it: it[1])
    return dists[0][0]


def save_stafflines_viz(out_dir: str) -> None:
    """Lưu ảnh trực quan hóa staffline vào thư mục đầu ra.

    Mục đích:
    - Hiển thị đồng thời mask staff, biên zone, staff bbox và điểm line.
    - Phân biệt rõ staff gốc và staff nội suy (`is_interp`).
    - Hỗ trợ gỡ lỗi nhanh khi tinh chỉnh ngưỡng phát hiện.
    """
    try:
        img = layers.get_layer('original_image')
        if img is None:
            logger.debug('Khong co lop original_image')
            return
        canvas = img.copy()
    except Exception:
        logger.debug('original_image thieu hoac khong hop le')
        return

    staff_pred = layers.get_layer('staff_pred')
    staffs     = layers.get_layer('staffs')
    zones      = layers.get_layer('zones')

    # Lớp 1: phủ mask staff_pred lên ảnh gốc để quan sát vùng phát hiện.
    canvas = _overlay(canvas, staff_pred, C['staff_line'], alpha=0.25)

    # Lớp 2: vẽ đường biên các vùng zone để kiểm tra bước chia cột.
    if zones is not None:
        h = canvas.shape[0]
        for zone in zones:
            x_start = int(zone[0])
            cv2.line(canvas, (x_start, 0), (x_start, h - 1), C['zone'], 1)

    # Lớp 3: vẽ bbox khuông, điểm của từng dòng và thông tin group/track.
    if staffs is None:
        _save(out_dir, 'step1_stafflines', canvas)
        return

    line_colors = [
        (255, 80,  80), (80, 200,  80), (80, 160, 255), (220, 180,  0), (200,  60, 200)
    ]

    staffs_flat = staffs.ravel() if hasattr(staffs, 'ravel') else staffs
    drawn_labels: set = set()
    for st in staffs_flat:
        x1 = int(st.x_left); x2 = int(st.x_right); y1 = int(st.y_upper); y2 = int(st.y_lower)
        box_color = (80, 255, 200) if not st.is_interp else (120, 120, 120)
        cv2.rectangle(canvas, (x1, y1 - 2), (x2, y2 + 2), box_color, 1)
        for line in st.lines:
            if line.label is None:
                continue
            lid = line.label.value
            col = line_colors[lid]
            pts = np.array(line.points, dtype=np.int32)
            if len(pts) == 0:
                continue
            ys = pts[:, 0]; xs = pts[:, 1]
            for px, py in zip(xs[::4], ys[::4]):
                cv2.circle(canvas, (int(px), int(py)), 1, col, -1)

        label_key = (round(st.y_center), round(st.x_center / 100))
        if label_key not in drawn_labels:
            drawn_labels.add(label_key)
            tag = f"G{st.group}-T{st.track}  u={st.unit_size:.1f}  sl={st.slope:.3f}"
            if st.is_interp:
                tag += " [noi suy]"
            _text(canvas, tag, (x1 + 4, y1 - 5), box_color, scale=0.38)

    legend_items = [
        ("Dong 1 - FIRST",  line_colors[0]),
        ("Dong 2 - SECOND", line_colors[1]),
        ("Dong 3 - THIRD",  line_colors[2]),
        ("Dong 4 - FOURTH", line_colors[3]),
        ("Dong 5 - FIFTH",  line_colors[4]),
        ("Khung staff (group)",     (80, 255, 200)),
        ("Staff noi suy",           (120, 120, 120)),
        ("Bien zone (cot)",         C["zone"]),
    ]
    _legend(canvas, legend_items, x0=6, y0=18, dy=16)
    _save(out_dir, 'step1_stafflines', canvas)