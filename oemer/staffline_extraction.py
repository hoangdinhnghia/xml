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
from oemer.utils import get_logger
from oemer.bbox import BBox, find_lines, get_bbox, get_center

logger = get_logger(__name__)


class LineLabel(enum.Enum):
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4


class Line:
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []
        self.label: Union[LineLabel, None] = None

    def add_point(self, y: int, x: int) -> None:
        self.points.append((y, x))
        # Tự động xóa các cache đã tính toán khi có điểm mới thêm vào
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
    def __init__(self) -> None:
        self.lines: List[Line] = []
        self.track: Union[int, None] = None
        self.group: Union[int, None]  = None
        self.is_interp: bool = False

    def add_line(self, line: Line) -> None:
        self.lines.append(line)
        # Reset cache dữ liệu của khuông nhạc khi cấu trúc dòng kẻ thay đổi
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
    
    # SỬA LỖI LOGIC: Chia mảng đều tăm tắp bằng np.array_split tránh phần dư gộp vào cuối
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
    
    # TỐI ƯU HÓA VECTORIZATION (NUMPY BROADCASTING): Thay thế vòng lặp for từng điểm ảnh
    if len(sub_ys) > 0 and len(centers) > 0:
        # Tính khoảng cách tuyệt đối từ mọi điểm 'y' tới mọi đỉnh 'centers' cùng lúc
        distances = np.abs(sub_ys_expanded := sub_ys[:, np.newaxis] - centers)
        closest_cen_indices = np.argmin(distances, axis=1)
        assigned_centers = centers[closest_cen_indices]
        
        # Điều kiện Vector hóa hiệu năng cao
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

    # Assign labels
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
    len_types = set(len(st_part) for st_part in staffs)
    if len(len_types) == 1:
        return np.array(staffs)

    max_len = max(len_types) if len_types else 0
    grid = np.zeros((len(staffs), max_len), dtype=object)
    for idx, st_part in enumerate(staffs):
        if len(st_part) == max_len:
            grid[idx] = np.array(st_part)

    def get_nearby_sts(j, row):
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
                ref_idx, ref_st = sts[0]
                width = ref_st.x_right - ref_st.x_left
                x_offset = width * (j - ref_idx)
                new_st = ref_st.duplicate(x_offset=x_offset)
            else:
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
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    clefs = layers.get_layer('clefs_keys_pred')

    mix = symbols - stems - notehead - clefs
    mix[mix<0] = 0

    lines = find_lines(mix)
    lines = filter_lines(lines, staffs, min_degree=min_degree)
    bmap = get_barline_map(symbols, lines) + stems
    bmap[bmap>1] = 1

    ker = np.ones((5, 2), dtype=np.uint8)
    ext_bmap = cv2.erode(cv2.dilate(bmap.astype(np.uint8), ker), ker)
    bboxes = get_bbox(ext_bmap)

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
        valid_h = len(h_ratios[h_ratios>factor*i])
        if valid_h * (i+1) > staffs.shape[1]:
            num_track += 1
        else:
            break
    return num_track


def get_degree(line: BBox) -> float:
    return float(np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0])))


def filter_lines(lines: List[BBox], staffs: ndarray, min_degree: int = 75) -> List[BBox]:
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
    img = np.zeros_like(symbols)
    for box in bboxes:
        box = list(box)
        if box[2]-box[0] == 0:
            box[2] += 1
        img[box[1]:box[3], box[0]:box[2]] += symbols[box[1]:box[3], box[0]:box[2]]
    img[img>1] = 1
    return img


def naive_get_unit_size(staffs: ndarray, x: int, y: int) -> float:
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