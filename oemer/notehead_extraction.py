from __future__ import annotations

"""
Module trích xuất và phân loại đầu nốt (notehead).

Chịu trách nhiệm nhận các dự đoán `notehead_pred` từ tầng phân đoạn,
tiền xử lý mặt nạ, phát hiện các blob ứng viên, tách các đầu nốt dính nhau,
lọc theo đặc trưng hình học, gán đầu nốt về `Staff` tương ứng và phân loại sơ bộ
(rỗng/đặc). Các hàm chính gồm `detect_noteheads`, `get_notehead_bbox`,
`gen_notes`, `parse_stem_info` và `extract`.

File này sử dụng các heuristic dựa trên `unit_size` của khuông để điều chỉnh
tham số morphology, thresholds và kiểm tra kích thước; các giá trị mặc định
được khai báo trong các tham số hàm và hằng số trong `oemer.constant`.
"""

import enum
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import scipy.ndimage as ndi
from numpy import ndarray
from scipy.signal import find_peaks

from oemer import layers
from oemer.bbox import (
    BBox, get_bbox, get_center,
    merge_nearby_bbox, rm_merge_overlap_bbox, to_rgb_img,
)
from oemer.constant import NoteHeadConstant as nhc
from oemer.utils import get_logger
from oemer.staffline_extraction import Staff
from oemer.utils import find_closest_staffs, get_global_unit_size, get_unit_size

logger = get_logger(__name__)
from oemer.utils import C, _overlay, _rect, _text, _legend, _save

nn_img: ndarray   # hình ảnh để trực quan gỡ lỗi, được điền trong hàm extract()

# ─────────────────────────────────────────────────────────────────────────────
# Liệt kê (Enumerations)
# ─────────────────────────────────────────────────────────────────────────────

class NoteType(enum.Enum):
    """Kiểu trường độ sơ bộ của notehead.

    Giá trị này là nhãn sơ bộ dùng trong pipeline trước khi có phân tích
    cọng/beam đầy đủ. `HALF_OR_WHOLE` là trạng thái trung gian cho các đầu
    nốt rỗng cần phân giải thêm (ví dụ tách HALF hay WHOLE sau khi xem cọng).
    """
    WHOLE         = 0
    HALF          = 1
    QUARTER       = 2
    EIGHTH        = 3
    SIXTEENTH     = 4
    THIRTY_SECOND = 5
    SIXTY_FOURTH  = 6
    TRIPLET       = 7
    OTHERS        = 8
    HALF_OR_WHOLE = 9   # trạng thái trung gian; phân giải sau khi phân tích cọng


# ─────────────────────────────────────────────────────────────────────────────
# Mô hình dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoteHead:
    """
    Một notehead được phát hiện đơn lẻ cùng các siêu dữ liệu suy ra.

    Các trường được điền dần dần:
      bbox + points → group/track/staff_line_pos → label + hướng cọng.
    `solidity` (diện tích / diện tích bao lồi) là trường mới không có trong
    tham chiếu; nó được tính trong `gen_notes` và có thể dùng để lọc nhiễu.
    """
    bbox:           BBox | None           = None
    points:         list[tuple[int, int]] = field(default_factory=list)
    pitch:          int | None            = None
    has_dot:        bool                  = False
    stem_up:        bool | None           = None
    stem_right:     bool | None           = None
    track:          int | None            = None
    group:          int | None            = None
    staff_line_pos: int | None            = None
    invalid:        bool                  = False
    id:             int | None            = None
    note_group_id:  int | None            = None
    sfn:            Any                   = None   # dau hoa bat thuong (thang/giang/hoan)
    solidity:       float                 = 0.0    # MỚI: solidity từ bao lồi

    _label: NoteType | None = field(default=None, repr=False)

    @property
    def label(self) -> NoteType | None:
        if self.invalid:
            logger.warning("Note %s is marked invalid.", self.id)
            return None
        return self._label

    @label.setter
    def label(self, value: NoteType) -> None:
        if self._label is not None:
            logger.debug("Label already %s — use force_set_label() to override.", self._label)
            return
        self._label = value

    def force_set_label(self, value: NoteType) -> None:
        assert isinstance(value, NoteType)
        logger.debug("force_set_label: %s → %s", self._label, value)
        self._label = value

    def add_point(self, x: int, y: int) -> None:
        self.points.append((y, x))

    def __lt__(self, other: NoteHead) -> bool:
        return (self.staff_line_pos or 0) < (other.staff_line_pos or 0)

    def __repr__(self) -> str:
        return (
            f"NoteHead(id={self.id}, label={self._label}, bbox={self.bbox}, "
            f"track={self.track}, group={self.group}, pitch={self.pitch}, "
            f"pos={self.staff_line_pos}, solidity={self.solidity:.2f}, "
            f"valid={not self.invalid})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hình thái notehead — Làm mượt Gaussian + đóng hình học
#
#    Tham chiếu cũ: erode → dilate (open), rồi dilate thêm lần nữa.
#    Vấn đề: phẫu thuật dilation đôi có thể nối các notehead kế nhau.
#
#    Cách mới:
#      • Làm mờ Gaussian giảm nhiễu "muối và tiêu" trước khi nhị phân hoá.
#      • Một phép đóng hình học (dilate → erode) duy nhất lấp các khe nhỏ
#        mà không làm mảng nở ra mạnh như dilation đôi.
# ─────────────────────────────────────────────────────────────────────────────


def detect_noteheads(pred: ndarray, unit_size: float) -> ndarray:
    """
    Chuyển dự đoán notehead thô thành các blob nhị phân sạch.

    Làm mượt Gaussian loại bỏ các pixel nhiễu đơn lẻ trước khi nhị phân.
    Đóng hình học lấp các khe nhỏ bên trong mà không làm mảng nở quá mức.
    """
    blur_k = max(3, int(unit_size / 4) * 2 + 1)   # kích thước kernel phải là số lẻ
    smoothed = cv2.GaussianBlur(pred.astype(np.float32), (blur_k, blur_k), 0)
    binary   = (smoothed > 0.3).astype(np.uint8)

    size   = (
        int(round(unit_size * nhc.NOTEHEAD_MORPH_WIDTH_FACTOR)),
        int(round(unit_size * nhc.NOTEHEAD_MORPH_HEIGHT_FACTOR)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def morph_notehead(pred: ndarray, unit_size: float) -> ndarray:
    """Bí danh tương thích ngược được oemer.notehead_new sử dụng."""
    return detect_noteheads(pred, unit_size)


# ─────────────────────────────────────────────────────────────────────────────
# 2. fill_hole — dùng scipy.ndimage.binary_fill_holes
#
#    Tham chiếu cũ: quét hàng rồi cột với phát hiện khe hở.
#    Lỗi trong tham chiếu: quét cột nằm trong vòng lặp hàng (lỗi thụt lề),
#    và thuật toán không lấp được lỗ không lồi hoặc chéo đúng.
#
#    `scipy.binary_fill_holes` gắn nhãn các thành phần nền và giữ lại chỉ
#    những thành phần chạm biên — định nghĩa toán học đúng cho "lỗ" với mọi
#    hình dạng, được triển khai hiệu quả bằng C.
# ─────────────────────────────────────────────────────────────────────────────


def fill_hole(region: ndarray) -> ndarray:
    """
    Lấp các lỗ bên trong mặt nạ nhị phân bằng phân tích thành phần kết nối.

    `scipy.ndimage.binary_fill_holes` xác định vùng nền hoàn toàn bị bao bọc
    bởi pixel nền và lấp chúng, không phụ thuộc hình dạng.
    """
    return ndi.binary_fill_holes(region > 0).astype(region.dtype)


def legacy_style_fill(region: ndarray) -> ndarray:
    """
    Cài lại nhẹ thuật toán quét hàng-rồi-cột kiểu cũ để lấp lỗ.
    Hàm này tái tạo hành vi lấp lỗ cũ để tỷ lệ phát hiện rỗng khớp với số liệu
    lịch sử mà không cần gọi mã cũ.
    """
    tar = region.copy().astype(np.uint8)
    h, w = tar.shape

    # Quét theo hàng
    for yi in range(h):
        cur = 0
        cand = []
        # Di chuyen den diem tien canh dau tien
        while cur < w and tar[yi, cur] == 0:
            cur += 1
        # Di chuyen den nen tiep theo sau doan tien canh
        while cur < w and tar[yi, cur] > 0:
            cur += 1
        # Thu thap cac pixel nen ung vien den khi gap tien canh tiep theo
        while cur < w and tar[yi, cur] == 0:
            cand.append(cur)
            cur += 1
        if cur < w and cand:
            for xi in cand:
                tar[yi, xi] = 1

    # Quét theo cột
    for xi in range(w):
        cur = 0
        cand = []
        while cur < h and tar[cur, xi] == 0:
            cur += 1
        while cur < h and tar[cur, xi] > 0:
            cur += 1
        while cur < h and tar[cur, xi] == 0:
            cand.append(cur)
            cur += 1
        if cur < h and cand:
            for yi in cand:
                tar[yi, xi] = 1

    return tar


def save_noteheads_viz(out_dir: str) -> None:
    """Luu anh truc quan hoa phat hien notehead vao thu muc out_dir."""
    try:
        img = layers.get_layer('original_image')
        if img is None:
            return
        canvas = img.copy()
    except Exception:
        return

    note_mask = layers.get_layer('notehead_pred')
    notes = layers.get_layer('notes')
    if note_mask is not None:
        canvas = _overlay(canvas, note_mask, C['note_half'], alpha=0.20)

    label_color_map = {
        'WHOLE': C['note_whole'],
        'HALF': C['note_half'],
        'HALF_OR_WHOLE': C['note_half'],
        'QUARTER': C['note_quarter'],
    }

    if notes is None:
        _save(out_dir, 'step2_noteheads', canvas)
        return

    for note in notes:
        if note.bbox is None:
            continue
        x1, y1, x2, y2 = note.bbox
        lbl = note._label.name if note._label is not None else '?'
        col = label_color_map.get(lbl, C['note_other'])
        nid = note.id if note.id is not None else '?'
        _rect(canvas, note.bbox, col, thickness=1)
        stem_ch = '^' if note.stem_up is True else ('v' if note.stem_up is False else '?')
        tag1 = f"#{nid} {lbl[:3]} {stem_ch}"
        _text(canvas, tag1, (x1, max(y1 - 3, 8)), col, scale=0.32)
        pos = note.staff_line_pos if note.staff_line_pos is not None else '?'
        sol = f"{note.solidity:.2f}" if hasattr(note, 'solidity') else '-'
        tag2 = f"pos={pos} sol={sol}"
        _text(canvas, tag2, (x1, min(y2 + 10, canvas.shape[0] - 4)), col, scale=0.30)

    legend_items = [
        ("WHOLE", C['note_whole']),
        ("HALF / H_OR_W", C['note_half']),
        ("QUARTER", C['note_quarter']),
        ("Khac / khong ro", C['note_other']),
        ("^ : stem huong len", C['stem_up']),
        ("v : stem huong xuong", C['stem_down']),
    ]
    _legend(canvas, legend_items, x0=6, y0=18, dy=16)
    _save(out_dir, 'step2_noteheads', canvas)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chia bbox — phép chiếu theo cột + scipy.signal.find_peaks
#
#    Tham chiếu cũ: thuật toán watershed của OpenCV trên ảnh 3 kênh,
#    yêu cầu chuẩn bị marker và xử lý các trường hợp biên.
#
#    Cách mới: chiếu mặt nạ nhị phân lên trục x (cộng từng cột).
#    Hồ sơ 1-D thu được có các thung lũng giữa các notehead dính nhau.
#    `scipy.find_peaks` trên hồ sơ đảo tìm vị trí thung lũng với ngưỡng
#    prominence có thể điều chỉnh — đơn giản hơn và không cần chuyển dạng.
# ─────────────────────────────────────────────────────────────────────────────


def _projection_split(bbox: BBox, mask: ndarray) -> list[BBox]:
    """
    Chia bbox tại các thung lũng của phép chiếu theo cột sử dụng find_peaks.

    Trả về các hộp con tách tại các thung lũng được phát hiện, hoặc [] khi
    không tìm thấy thung lũng rõ (notehead đơn lẻ hoặc blob đồng nhất).
    """
    x1, y1, x2, y2 = bbox
    region   = mask[y1:y2, x1:x2].astype(np.float32)
    col_proj = region.sum(axis=0)

    peak_val = col_proj.max()
    if peak_val == 0:
        return []

    # Thung lũng = đỉnh trên hồ sơ đảo
    valleys, _ = find_peaks(-col_proj, prominence=peak_val * 0.35)
    if valleys.size == 0:
        return []

    cuts = [x1] + [x1 + int(v) for v in valleys] + [x2]
    return [
        (cuts[i], y1, cuts[i + 1], y2)
        for i in range(len(cuts) - 1)
        if cuts[i + 1] > cuts[i]
    ]


def _tighten_vertical(bbox: BBox, mask: ndarray) -> BBox | None:
    """Co lại phạm vi theo chiều dọc của bbox về các hàng có pixel tiền cảnh thực."""
    ys, _ = np.where(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] > 0)
    if ys.size == 0:
        return None
    return (bbox[0], int(ys.min()) + bbox[1] - 1, bbox[2], int(ys.max()) + bbox[1] + 1)


def check_bbox_size(bbox: BBox, mask: ndarray, unit_size: float) -> list[BBox]:
    """
    Thu nhỏ đệ quy bbox cho tới khi mỗi phần khớp xấp xỉ một notehead.

    Nếu quá rộng → chia theo phép chiếu cột (hoặc chia giữa làm phương án dự phòng).
    Nếu quá cao → chia thành N lát ngang bằng nhau.
    """
    x1, y1, x2, y2 = bbox
    w, h   = x2 - x1, y2 - y1
    note_w = nhc.NOTEHEAD_SIZE_RATIO * unit_size
    note_h = unit_size

    if w > note_w * 1.3:
        parts = _projection_split(bbox, mask)
        if not parts:
            mid   = (x1 + x2) // 2
            parts = [(x1, y1, mid, y2), (mid, y1, x2, y2)]

        tightened = [_tighten_vertical(p, mask) for p in parts]
        return [
            sub
            for p in tightened if p is not None
            for sub in check_bbox_size(p, mask, unit_size)
        ]

    n = max(1, int(round(h / note_h)))
    if n == 1:
        return [bbox]

    slice_h = h / n
    return [
        (x1, round(y1 + i * slice_h), x2, round(y1 + (i + 1) * slice_h))
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lọc bbox — thêm `solidity` làm tiêu chí thứ ba
#
#    Kiểm tra tham chiếu: kích thước (h, w) và tỉ lệ pixel nền.
#    Bổ sung mới: `solidity` = diện tích contour / diện tích convex hull.
#
#    Notehead thật thường giống elip → `solidity` cao (thường ≥ 0.65).
#    Dương giả (mảnh nét, phần clef, nhiễu staff) thường kéo dài/không đều →
#    `solidity` thấp. Điều này giảm dương giả mà không phải thắt chặt ngưỡng kích thước.
# ─────────────────────────────────────────────────────────────────────────────


def _compute_solidity(region: ndarray) -> float:
    """
    Trả về tỉ lệ diện tích contour / diện tích convex hull cho contour lớn nhất.
    """
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    cnt       = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return float(area / hull_area) if hull_area > 0 else 0.0


def filter_notehead_bbox(
    bboxes: list[BBox],
    notehead: ndarray,
    *,
    min_h_ratio: float = 0.4,
    max_h_ratio: float = 5.0,
    min_w_ratio: float = 0.3,
    max_w_ratio: float = 3.0,
    min_area_ratio: float = 0.6,
    min_solidity: float = 0.6,
) -> list[BBox]:
    """
    Chỉ giữ các bbox thỏa cả ba kiểm tra:
      1. Kích thước — chiều cao/chiều rộng trong ngưỡng so với đơn vị
      2. Mật độ  — tỉ lệ pixel tiền cảnh trong bbox
      3. Solidity — diện tích contour / diện tích convex hull ≥ min_solidity  (MỚI)
    """
    zones = layers.get_layer('zones')
    min_x, max_x = zones[0][0], zones[-1][-1]

    valid: list[BBox] = []
    for bbox in bboxes:
        cen_x, cen_y = get_center(bbox)
        u = get_unit_size(cen_x, cen_y)

        if not (min_x + nhc.CLEF_ZONE_WIDTH_UNIT_RATIO * u < cen_x <= max_x):
            continue

        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]

        if not (u * min_h_ratio <= h <= u * max_h_ratio):
            continue
        if not (u * min_w_ratio * nhc.NOTEHEAD_SIZE_RATIO
                <= w <=
                u * max_w_ratio * nhc.NOTEHEAD_SIZE_RATIO):
            continue

        region = notehead[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if (region > 0).sum() < h * w * min_area_ratio:
            continue

        if _compute_solidity(region) < min_solidity:
            continue

        valid.append(bbox)
    return valid


def get_notehead_bbox(
    pred: ndarray,
    global_unit_size: float,
    *,
    min_h_ratio: float = 0.4,
    max_h_ratio: float = 5.0,
    min_w_ratio: float = 0.3,
    max_w_ratio: float = 3.0,
    min_area_ratio: float = 0.65,
    min_solidity: float = 0.6,
) -> list[BBox]:
    """Toàn bộ đường ống phát hiện: làm mượt → blob → chia → lọc."""
    logger.debug("Detecting notehead blobs")
    note      = detect_noteheads(pred, global_unit_size)
    raw_boxes = rm_merge_overlap_bbox(get_bbox(note))

    split_boxes: list[BBox] = []
    for box in raw_boxes:
        u = get_unit_size(*get_center(box))
        split_boxes.extend(check_bbox_size(box, pred, u))
    logger.debug("Candidates after splitting: %d", len(split_boxes))

    result = filter_notehead_bbox(
        split_boxes, note,
        min_h_ratio=min_h_ratio,   max_h_ratio=max_h_ratio,
        min_w_ratio=min_w_ratio,   max_w_ratio=max_w_ratio,
        min_area_ratio=min_area_ratio, min_solidity=min_solidity,
    )
    logger.debug("Noteheads after filtering: %d", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Vị trí dòng khuông — np.linspace + np.searchsorted
#
#    Tham chiếu cũ: xây mảng vị trí xen kẽ dòng-khoảng với một vòng lặp thủ công
#    chèn tâm khoảng giữa các dòng từng bước.
#
#    Cách mới: np.linspace sinh tất cả vị trí dòng+khoảng trong một lần giữa
#    hai dòng ngoài cùng được mở rộng bởi ledger slots. np.searchsorted sau đó
#    xác định ô gần nhất trong O(log n) mà không phải dò từng phần tử.
# ─────────────────────────────────────────────────────────────────────────────


def _staff_line_position(cen_y: float, staff: Staff) -> int:
    """
    Ánh xạ toạ độ theo chiều dọc sang chỉ số ô vị trí trên khuông.

    Giữ tính tương thích với semantics tâm xen kẽ cũ để ánh xạ cao độ
    tương thích với các bộ dựng rhythm và XML phía sau.
    """
    step = staff.unit_size / 2
    pos_cen = [line.y_center for line in staff.lines[::-1]]
    tmp_inter = []
    for idx, cen in enumerate(pos_cen[:-1]):
        interp = (cen + pos_cen[idx + 1]) / 2
        tmp_inter.append(interp)
    for idx, interp in enumerate(tmp_inter):
        pos_cen.insert(idx * 2 + 1, interp)
    pos_cen = [pos_cen[0] + step] + pos_cen + [pos_cen[-1] - step]

    pos_idx = np.argmin(np.abs(np.array(pos_cen) - cen_y))
    if 0 < pos_idx < len(pos_cen) - 1:
        return int(pos_idx)
    elif pos_idx == 0:
        diff = abs(pos_cen[0] - cen_y)
        pos = round(diff / step)
        return -pos
    else:
        diff = abs(pos_cen[-1] - cen_y)
        pos = round(diff / step) + len(pos_cen) - 1
        return pos


# ─────────────────────────────────────────────────────────────────────────────
# Xây dựng NoteHead
# ─────────────────────────────────────────────────────────────────────────────


def gen_notes(bboxes: list[BBox], symbols: ndarray) -> list[NoteHead]:
    """Khoi tao cac doi tuong NoteHead voi bbox, diem, thong tin khuong va solidity."""
    notes: list[NoteHead] = []
    for bbox in bboxes:
        note = NoteHead(bbox=bbox)

        ys, xs = np.where(symbols[bbox[1]:bbox[3], bbox[0]:bbox[2]] > 0)
        for y, x in zip(ys + bbox[1], xs + bbox[0]):
            note.add_point(int(x), int(y))

        cen_x, cen_y = get_center(bbox)
        st1, st2 = find_closest_staffs(cen_x, cen_y)

        if st1.y_center == st2.y_center or st1.y_upper <= cen_y <= st1.y_lower:
            master = st1
        else:
            up, lo = (st1, st2) if st1.y_center < st2.y_center else (st2, st1)
            master = up if cen_y < (up.y_center + lo.y_center) / 2 else lo

        note.group         = master.group
        note.track         = master.track
        note.staff_line_pos = _staff_line_position(cen_y, master)

        region       = symbols[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        note.solidity = _compute_solidity(region)

        notes.append(note)
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hướng cọng — scipy.ndimage.center_of_mass
#
#    Tham chiếu cũ: np.where thu thập toạ độ pixel vào mảng Python,
#    rồi np.mean tính tâm — hai lần lặp ở cấp Python.
#
#    `scipy.center_of_mass` tính tâm có trọng số trong một lần gọi ở cấp C
#    trên mảng được gán nhãn, không tạo mảng toạ độ trung gian.
# ─────────────────────────────────────────────────────────────────────────────


def parse_stem_info(notes: list[NoteHead]) -> None:
    """
    Xác định cọng nằm về phải hay trái so với mỗi notehead.

    Dung scipy.ndimage.center_of_mass tren ban do cong da gan nhan:
    mot lan goi o cap C cho moi nhan thay vi tao mang toa do o Python.
    """
    stems  = layers.get_layer('stems_rests_pred')
    kernel = np.ones((3, 2), np.uint8)
    st_map, _ = ndi.label(cv2.dilate(stems.astype(np.uint8), kernel))

    for note in notes:
        x1, y1, x2, y2 = note.bbox  # type: ignore[misc]
        stem_labels = np.unique(st_map[y1:y2, x1:x2])
        stem_labels = stem_labels[stem_labels > 0]
        if stem_labels.size == 0:
            continue

        _, col_centroid = ndi.center_of_mass(st_map == int(stem_labels[0]))
        note.stem_right = bool(col_centroid > (x1 + x2) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Đường ống chính
# ─────────────────────────────────────────────────────────────────────────────


def extract(
    *,
    min_h_ratio: float = 0.4,
    max_h_ratio: float = 5.0,
    min_w_ratio: float = 0.3,
    max_w_ratio: float = 3.0,
    min_area_ratio: float = 0.65,
    min_solidity: float = 0.6,
    max_whole_note_width_factor: float = 1.5,
    y_dist_factor: int = 5,
    hollow_filled_ratio_th: float = 1.05,
) -> list[NoteHead]:
    """
    Trích xuất notehead đầu-cuối.

    Trả về các NoteHead với bbox, vị trí trên khuông, group/track,
    solidity và hướng cọng đã được điền. Notehead rỗng được gán nhãn
    HALF_OR_WHOLE; phân giải sang HALF hoặc WHOLE sẽ xảy ra phía sau.
    """
    # Lấy các dự đoán từ các layer đã đăng ký
    note_pred = layers.get_layer('notehead_pred')
    symbols = layers.get_layer('symbols_pred')

    # Tỉ lệ toàn cục dùng cho morphology và kiểm tra kích thước
    global_u = get_global_unit_size()

    # Phát hiện bbox ứng viên từ đường ống mới
    bboxes = get_notehead_bbox(
        note_pred,
        global_u,
        min_h_ratio=min_h_ratio,
        max_h_ratio=max_h_ratio,
        min_w_ratio=min_w_ratio,
        max_w_ratio=max_w_ratio,
        min_area_ratio=min_area_ratio,
        min_solidity=min_solidity,
    )

    notes = gen_notes(bboxes, symbols)

    # Thong tin cong (dien truong stem_right)
    parse_stem_info(notes)

    # Loc hau xu ly: loai phat hien nhieu co mat do/solidity thap
    filtered: list[NoteHead] = []
    for note in notes:
        if note.bbox is None:
            continue
        x1, y1, x2, y2 = note.bbox
        region = symbols[y1:y2, x1:x2].astype(np.uint8)
        area = max(1, (x2 - x1) * (y2 - y1))
        density = float((region > 0).sum()) / area
        # Giu lai neu mat do du cao hoac solidity cho thay hinh dang phu hop
        if density >= 0.5 or note.solidity >= 0.6:
            filtered.append(note)
        else:
            # Truong hop kha nang la note rong: thu danh gia lai bang lap kieu cu
            lfilled = legacy_style_fill(region)
            lratio = float((lfilled > 0).sum()) / max(1, (region > 0).sum())
            if lratio > hollow_filled_ratio_th:
                filtered.append(note)
            else:
                # Loai bo phat hien do tin cay thap
                continue
    notes = filtered

    # Quyet dinh rong hay dac bang tap quy tac ket hop, khong phu thuoc trien khai cu.
    # Dung ti le filled/unfilled va topology (dem contour/hole) de tang do ben vung.
    for note in notes:
        if note.bbox is None:
            continue
        x1, y1, x2, y2 = note.bbox
        region = symbols[y1:y2, x1:x2].astype(np.uint8)

        sym_count = int((region > 0).sum())
        if sym_count == 0:
            continue

        # Loai bo cong truoc khi phan tich topology de tranh cong noi hoac lap lo
        stems = layers.get_layer('stems_rests_pred')
        st_crop = stems[y1:y2, x1:x2].astype(np.uint8)
        # Phong to cong mot chut de dam bao loai bo
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        st_dil = cv2.dilate(st_crop, ker, iterations=1)
        region_nostem = np.where(st_dil > 0, 0, region)

        # Lam sach cac dom nho truoc khi phan tich contour
        clean_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        region_clean = cv2.morphologyEx(region_nostem, cv2.MORPH_OPEN, clean_ker)

        filled = fill_hole(region_clean)
        filled_count = int((filled > 0).sum())
        filled_ratio = filled_count / float(max(1, sym_count))

        # Tinh ti le lap kieu cu de tuong thich voi nguong lich su
        legacy_filled = legacy_style_fill(region_clean)
        legacy_filled_count = int((legacy_filled > 0).sum())
        legacy_ratio = legacy_filled_count / float(max(1, sym_count))

        # đo thêm: bao nhiêu pixel được thêm vào khi lấp lỗ
        hole_pixels = max(0, filled_count - sym_count)
        hole_area_ratio = float(hole_pixels) / float(max(1, filled_count))

        # Cau truc contour: dem lo tren mat na da lam sach
        cnts, hierarchy = cv2.findContours(region_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        if hierarchy is not None and hierarchy.size:
            h = hierarchy.reshape(-1, 4)
            # Contour con (parent != -1) cho thay co lo
            hole_count = int((h[:, 3] != -1).sum())

            # Quy tac heuristic cai tien: ket hop solidity va cac nguong mem
        w = x2 - x1
        u = get_unit_size(int((x1 + x2) / 2), int((y1 + y2) / 2))

        if w > u * nhc.NOTEHEAD_SIZE_RATIO * max_whole_note_width_factor:
            note.force_set_label(NoteType.WHOLE)
        else:
            # Quy tac note rong mem hon: chap nhan filled_ratio thap hon mot chut
            # khi topology lo hoac solidity thap goi y rong.
            solidity = getattr(note, 'solidity', 0.0)

            # Tom tat quy tac:
            # - hole_count (từ hierarchy contour) là bằng chứng mạnh
            # - hole_area_ratio (pixel thêm bởi fill) bắt các vòng mỏng
            # - legacy_ratio hữu ích ngay cả khi hole_count==0
            # - filled_ratio đơn lẻ yếu hơn vì cọng và nhiễu ảnh hưởng

            is_hollow = False

            # Topology truc tiep: contour cho thay lo
            if hole_count >= 1:
                if legacy_ratio > hollow_filled_ratio_th * 0.95 or filled_ratio > hollow_filled_ratio_th * 0.95:
                    is_hollow = True
                elif hole_area_ratio > 0.04:
                    is_hollow = True

            # Neu khong co contour ro rang, dung ti le phinh theo cach cu va hole area de du phong
            if not is_hollow:
                if legacy_ratio > (hollow_filled_ratio_th + 0.15):
                    is_hollow = True
                elif hole_area_ratio > 0.06:
                    is_hollow = True

            # solidity thap va filled_ratio vua phai thuong nghieng ve note rong
            if not is_hollow:
                if solidity < 0.65 and filled_ratio > (hollow_filled_ratio_th - 0.15):
                    is_hollow = True

            if is_hollow:
                note.force_set_label(NoteType.HALF_OR_WHOLE)
            else:
                note.force_set_label(NoteType.QUARTER)

    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Trực quan hoá
# ─────────────────────────────────────────────────────────────────────────────


def draw_notes(notes: list[NoteHead], ori_img: ndarray) -> ndarray:
    """Ve hop gioi han va ky hieu loai len anh ori_img."""
    img = np.array(ori_img, copy=True)
    for note in notes:
        x1, y1, x2, y2 = note.bbox  # type: ignore[misc]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if note.label:
            cv2.putText(img, note.label.name[0], (x2 + 2, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    return img
