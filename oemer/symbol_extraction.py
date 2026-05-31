from typing import List, Union, Any, Tuple
import enum

import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from oemer import layers
from oemer import exceptions as E
from oemer.inference import predict
from oemer.utils import get_global_unit_size, slope_to_degree, get_unit_size, find_closest_staffs
from oemer.utils import get_logger
from oemer.general_filtering_rules import filter_out_of_range_bbox, filter_out_small_area
from oemer.bbox import (
    BBox,
    merge_nearby_bbox,
    rm_merge_overlap_bbox,
    find_lines,
    draw_lines,
    draw_bounding_boxes,
    get_bbox,
    get_center,
    to_rgb_img
)


def get_kernel(kernel: Tuple[int, int]) -> ndarray:
    if isinstance(kernel, tuple):
        # Truyền vào dạng kích thước (h, w) thì tạo kernel toàn 1.
        kernel = np.ones(kernel, dtype=np.uint8)  # type: ignore
    return kernel  # type: ignore


def morph_open(img: ndarray, kernel: Tuple[int, int]) -> ndarray:
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, ker)


def morph_close(img: ndarray, kernel: Tuple[int, int]) -> ndarray:
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, ker)


def morph_hit_miss(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, ker)


# Globals
global_cs: ndarray
temp: ndarray

logger = get_logger(__name__)
from oemer.utils import C, _overlay, _rect, _text, _legend, _save


class ClefType(enum.Enum):
    G_CLEF = 1
    F_CLEF = 2


class SfnType(enum.Enum):
    FLAT = 1
    SHARP = 2
    NATURAL = 3


class RestType(enum.Enum):
    WHOLE_HALF = 1
    QUARTER = 2
    EIGHTH = 3
    SIXTEENTH = 4
    THIRTY_SECOND = 5
    SIXTY_FOURTH = 6
    WHOLE = 7
    HALF = 8


class Clef:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self._label: ClefType = None  # type: ignore

    @property
    def label(self) -> ClefType:
        return self._label

    @label.setter
    def label(self, val: ClefType) -> None:
        assert isinstance(val, ClefType)
        self._label = val

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Clef: {self.label.name} / Track: {self.track} / Group: {self.group}"


class Sfn:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.note_id: Union[int, None] = None
        self.is_key: Union[bool, None] = None  # Whether is key or accidental
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self._label: SfnType = None  # type: ignore

    @property
    def label(self) -> SfnType:
        return self._label

    @label.setter
    def label(self, val: SfnType) -> None:
        assert isinstance(val, SfnType)
        self._label = val

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"SFN: {self.label.name} / Note ID: {self.note_id} / Is key: {self.is_key}" \
            f" / Track: {self.track} / Group: {self.group}"


class Rest:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.has_dot: Union[bool, None] = None
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self._label: RestType = None  # type: ignore

    @property
    def label(self) -> RestType:
        return self._label

    @label.setter
    def label(self, val: RestType) -> None:
        assert isinstance(val, RestType)
        self._label = val

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Rest: {self.label.name} / Has dot: {self.has_dot} / Track: {self.track}" \
            f" / Group: {self.group}"


class Barline:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.group: Union[int, None] = None

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Barline / Group: {self.group}"


# --- Pha 1: Trích xuất barline ---
# Ý tưởng:
# 1) Lấy các line thẳng đứng từ map dự đoán (đã loại notehead).
# 2) Lọc theo hình học (độ dốc, chiều cao theo unit_size).
# 3) Chuẩn hóa chiều cao để loại line ngắn bất thường.
def filter_barlines(lines: List[BBox], min_height_unit_ratio: float = 3.75) -> ndarray:
    """
    Lọc danh sách line ứng viên để giữ lại barline đáng tin cậy.

    Quy trình 2 tầng:
    1) Lọc line theo hình học cơ bản (độ dốc gần thẳng đứng).
    2) Chuyển sang bbox, lọc theo chiều cao tương đối với unit_size.

    Tham số min_height_unit_ratio là ngưỡng theo đơn vị khoảng cách staff,
    giúp thuật toán thích nghi với ảnh có scale khác nhau.
    """
    # Lọc nhiễu hình học trước khi kiểm tra điều kiện barline.
    lines = filter_out_of_range_bbox(lines)
    # lines = merge_nearby_bbox(lines, 100, x_factor=100)
    lines = rm_merge_overlap_bbox(lines, mode='merge', overlap_ratio=0)

    # Vòng 1: kiểm tra trên dạng line (độ dốc gần thẳng đứng).
    valid_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        # unit_size cục bộ cho phép cùng một ngưỡng chạy được trên nhiều staff scale.
        unit_size = get_unit_size(*get_center(line))

        # Check slope. Degree should be within 80~100.
        deg = slope_to_degree(y2-y1, x2-x1)
        if abs(deg) < 75:
            continue

        valid_lines.append(line)

    # Vòng 2: chuyển sang bbox để kiểm tra chiều cao hữu hiệu.
    valid_lines = np.array(valid_lines)  # type: ignore
    max_x = np.max(valid_lines[..., 2])  # type: ignore
    max_y = np.max(valid_lines[..., 3])  # type: ignore
    # Rasterize line để gom các đoạn đứt thành vùng liên thông rồi lấy bbox chuẩn.
    data = np.zeros((max_y+10, max_x+10, 3))
    data = draw_lines(valid_lines, data, width=1)  # type: ignore
    boxes = get_bbox(data[..., 1])
    valid_box = []
    for box in boxes:
        _, y1, _, y2 = box

        # Check height
        if (y2 - y1) < unit_size * min_height_unit_ratio:
          continue

        valid_box.append(box)

    # Chuẩn hóa theo top-5 chiều cao lớn nhất:
    # - Giảm ảnh hưởng của outlier đơn lẻ.
    # - Bỏ các line thấp bất thường do nhiễu nét dọc.
    valid_box = sorted(valid_box, key=lambda box:box[3]-box[1])
    heights = [b[3] - b[1] for b in valid_box]
    top_5 = np.mean(heights[-5:])
    norm = np.array(heights) / top_5
    idx = np.where(norm > 0.5)[0]
    valid_box = np.array(valid_box)[idx]  # type: ignore

    return valid_box  # type: ignore


def parse_barlines(
    group_map: ndarray, 
    stems_rests: ndarray, 
    symbols: ndarray, 
    min_height_unit_ratio: float = 3.75
) -> ndarray:
    """
    Trích xuất bbox barline từ các map dự đoán segmentation.

    Dữ liệu dùng:
    - stems_rests: vùng nét dọc và rest.
    - symbols: vùng ký hiệu tổng quát.
    - group_map: vùng notehead/staff-group đã biết.

    Ý tưởng chính: tạo ứng viên nét dọc, sau đó giữ những thành phần có
    chồng lấp với symbols để giảm false positive, rồi đưa qua bộ lọc hình học.
    """
    # Đầu vào: group_map, stems_rests, symbols từ các lớp segmentation.
    # Đầu ra: danh sách bbox của barline đã được xác thực.
    # Thuật toán: giao cắt thành phần liên thông giữa ứng viên line và symbol map,
    # sau đó fit line + lọc hình học trong filter_barlines().
    # Loại notehead khỏi map stems/rests để lấy ứng viên barline.
    barline_cand = np.where(stems_rests-group_map>1, 1, 0)

    # Map ký hiệu không chứa notehead để đối chiếu chồng lấp.
    no_note = np.where(symbols-group_map>1, 1, 0)

    # Gán nhãn thành phần liên thông cho từng vùng pixel.
    bar_label, bnum = scipy.ndimage.label(barline_cand)
    sym_label, _ = scipy.ndimage.label(no_note)

    # Chỉ giữ các vùng line có chồng lấp với vùng symbol hợp lệ.
    sym_barline_map = np.zeros_like(no_note)
    for i in range(1, bnum+1):
        idx = (bar_label == i)
        region = sym_label[idx]
        labels = set(np.unique(region))
        if 0 in labels:
            labels.remove(0)
        for label in labels:
            sym_idx = (sym_label == label)
            sym_barline_map[sym_idx] += no_note[sym_idx]
    sym_barline_map[sym_barline_map>0] = 1

    lines = find_lines(sym_barline_map)
    line_box = filter_barlines(lines, min_height_unit_ratio)
    logger.debug("Detected barlines: %d", len(line_box))

    return line_box


def filter_clef_box(bboxes: List[BBox]) -> List[BBox]:
    # Lọc box clef bằng 2 tiêu chí chính:
    # - Kích thước đủ lớn so với unit_size cục bộ.
    # - Tâm box nằm trong dải staff gần nhất.
    valid_box = []
    for box in bboxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        cen_x, cen_y = get_center(box)
        unit_size = get_unit_size(cen_x, cen_y)

        # Clef quá nhỏ thường là nhiễu hoặc mảnh ký hiệu.
        if w < unit_size*1.5 or h < unit_size*1.5:
            continue

        # Tâm clef phải nằm trong phạm vi staff gần nhất.
        staff, _ = find_closest_staffs(cen_x, cen_y)
        if cen_y < staff.y_upper or cen_y > staff.y_lower:
            continue

        valid_box.append(box)
    return valid_box


def parse_clefs_keys(
    clefs_keys: ndarray, 
    unit_size: float, 
    clef_size_ratio: float = 3.5, 
    max_clef_tp_ratio: float = 0.45
) -> Tuple[List[BBox], List[BBox], List[str], List[str]]:
    """
    Tách và phân loại clef + accidental (sfn) từ map clefs_keys.

    Heuristic tách clef/sfn dựa trên 2 đại lượng:
    - area_size_ratio = (w*h)/unit_size^2: đặc trưng kích thước tương đối.
    - area_tp_ratio = pixel_on/(w*h): đặc trưng mật độ nét trong bbox.

    clef thường có vùng bao lớn hơn và rỗng hơn accidental,
    vì vậy dùng ngưỡng clef_size_ratio + max_clef_tp_ratio để phân nhánh.
    """
    # --- Pha 2: Tách clef và accidental (key signature/accidental) ---
    # Đầu vào: map clefs_keys và unit_size toàn cục.
    # Đầu ra: bbox + nhãn cho clef và sfn (sharp/flat/natural).
    # Thuật toán:
    # 1) Morphology để làm mượt vùng ký hiệu.
    # 2) Lấy bbox và lọc nhiễu (ngoài vùng, chồng lấp, diện tích nhỏ).
    # 3) Tách clef/key theo tỉ lệ diện tích và mật độ pixel.
    # 4) Cắt ROI từng box và gọi model phân loại tương ứng.
    global cs_img
    cs_img = to_rgb_img(clefs_keys)  # type: ignore

    # Erode/Dilate theo trục dọc để ổn định hình dạng trước khi lấy bbox.
    ker = np.ones((np.int64(unit_size//2), 1), dtype=np.uint8)
    clefs_keys = cv2.erode(cv2.dilate(clefs_keys.astype(np.uint8), ker), ker)
    # Lấy bbox từ mask đã morphology để giảm đứt nét trước khi phân loại.
    bboxes = get_bbox(clefs_keys)
    bboxes = filter_out_of_range_bbox(bboxes)
    bboxes = rm_merge_overlap_bbox(bboxes, mode='merge', overlap_ratio=0.3)
    bboxes = filter_out_small_area(bboxes, area_size_func=lambda usize: usize**2)
    # Gộp box gần nhau để tránh một ký hiệu bị cắt thành nhiều mảnh nhỏ.
    bboxes = merge_nearby_bbox(bboxes, unit_size*1.2)

    key_box = []
    clef_box = []
    for box in bboxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        region: ndarray = clefs_keys[box[1]:box[3], box[0]:box[2]]
        usize = get_unit_size(*get_center(box))
        area_size_ratio = w * h / usize**2
        area_tp_ratio = region[region>0].size / (w * h)
        #cv2.rectangle(cs_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        #cv2.putText(cs_img, f"{area_tp_ratio:.2f} / {area_size_ratio:.2f}", (box[2]+2, box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # Clef thường to hơn và mật độ pixel thấp hơn accidental.
        if area_size_ratio > clef_size_ratio \
                and area_tp_ratio < max_clef_tp_ratio:
            clef_box.append(box)
        elif w > usize/2 and h > usize/2:
            key_box.append(box)

    clef_box = filter_clef_box(clef_box)

    def pred_symbols(bboxes: List[BBox], model_name: str) -> List[str]:
        # Cắt từng vùng bbox và phân loại bằng model tương ứng.
        label = []
        for x1, y1, x2, y2 in bboxes:
            region = np.copy(clefs_keys[y1:y2, x1:x2])
            ll = predict(region, model_name)
            label.append(ll)
            #cv2.putText(cs_img, str(ll), (x2+2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return label

    clef_label = pred_symbols(clef_box, "clef")
    key_label = pred_symbols(key_box, "sfn")

    return clef_box, key_box, clef_label, key_label


def parse_rests(line_box: ndarray, unit_size: float) -> Tuple[List[BBox], List[str]]:
    """
    Trích xuất rest bằng phép trừ mask theo từng lớp semantic.

    Công thức:
    rests = stems_rests - notehead/group_map - barline_mask

    Sau khi có ứng viên:
    - Lấy bbox và lọc nhiễu theo diện tích/kích thước.
    - Phân loại rest bằng model 'rests'.
    - Nếu là nhóm 8th trở lên thì refine bằng model 'rests_above8'.
    """
    # --- Pha 3: Trích xuất rest ---
    # Ý tưởng: loại bỏ vùng đã biết (notehead/group và barline) khỏi stems_rests,
    # phần còn lại là ứng viên rest; sau đó lọc hình học và phân loại.
    stems_rests = layers.get_layer('stems_rests_pred')
    group_map = layers.get_layer('group_map')

    g_map = np.where(group_map>-1, 1, 0)

    # Tạo mask barline từ line_box để loại khỏi nhánh rest.
    data = np.zeros_like(group_map)
    for x1, y1, x2, y2 in line_box:
        data[y1:y2, x1:x2] = 1

    # Loại vùng group_map (notehead) và barline để còn lại ứng viên rest.
    rests = stems_rests - g_map - data
    rests[rests<0] = 0

    global temp
    temp = np.copy(rests)

    bboxes = get_bbox(rests)
    bboxes = filter_out_of_range_bbox(bboxes)
    if len(bboxes) == 0:
        return [], []
    bboxes = merge_nearby_bbox(bboxes, unit_size*1.2)
    bboxes = rm_merge_overlap_bbox(bboxes)
    bboxes = filter_out_small_area(bboxes, area_size_func=lambda usize: usize**2 * 0.7)
    temp = draw_bounding_boxes(bboxes, temp)

    label = []
    valid_box = []
    for box in bboxes:
        # Rest quá cao hoặc quá hẹp thường là nhiễu/ghép sai.
        if box[3] - box[1] > unit_size * 3.5 \
                or box[2] - box[0] < unit_size * 0.5:
            continue

        region = rests[box[1]:box[3], box[0]:box[2]]
        pred = predict(region, "rests")
        if "8th" in pred:
            pred = predict(region, "rests_above8")
        valid_box.append(box)
        label.append(pred)

    return valid_box, label


def gen_barlines(bboxes: ndarray) -> List[Barline]:
    # Chuyển bbox thuần sang object Barline có ngữ cảnh group.
    barlines = []
    for box in bboxes:
        # Gán mỗi barline vào group staff gần nhất.
        st1, _ = find_closest_staffs(*get_center(box))
        b = Barline()
        b.bbox = box
        b.group = st1.group
        barlines.append(b)
    return barlines


def save_symbols_viz(out_dir: str) -> None:
    """Lưu ảnh trực quan hóa các ký hiệu đã nhận diện (barline, clef, accidental, rest)."""
    try:
        img = layers.get_layer('original_image')
        if img is None:
            return
        canvas = img.copy()
    except Exception:
        return

    barlines = layers.get_layer('barlines') or []
    clefs = layers.get_layer('clefs') or []
    sfns = layers.get_layer('sfns') or []
    rests = layers.get_layer('rests') or []

    # Layer 1: vẽ barline trước để không che mất annotation chính.
    for b in barlines:
        if b.bbox is None:
            continue
        _rect(canvas, b.bbox, C['barline'], thickness=1)
        x1, y1, x2, y2 = b.bbox
        grp = b.group if b.group is not None else '?'
        _text(canvas, f"BAR g{grp}", (x1, y1 - 4), C['barline'], scale=0.35)

    # Layer 2: vẽ clef (dày hơn) vì là mốc mở đầu staff.
    for cl in clefs:
        if cl.bbox is None:
            continue
        _rect(canvas, cl.bbox, C['clef'], thickness=2)
        x1, y1, x2, y2 = cl.bbox
        lbl_name = cl.label.name if cl.label is not None else '?'
        tag = f"{lbl_name} T{cl.track} G{cl.group}"
        _text(canvas, tag, (x1, y1 - 5), C['clef'], scale=0.38)
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        cv2.circle(canvas, (cx, cy), 3, C['clef'], -1)

    sfn_short = {"SHARP": "#", "FLAT": "b", "NATURAL": "n"}
    # Layer 3: vẽ accidental và gắn trạng thái key/acc + note_id liên kết.
    for sf in sfns:
        if sf.bbox is None:
            continue
        _rect(canvas, sf.bbox, C['sfn'], thickness=1)
        x1, y1, x2, y2 = sf.bbox
        lbl_name = sf.label.name if sf.label is not None else '?'
        short = sfn_short.get(lbl_name, '?')
        is_key = 'key' if sf.is_key else ('acc' if sf.is_key is False else '?')
        nid_tag = f"n{sf.note_id}" if sf.note_id is not None else 'no_note'
        tag = f"{short} {is_key} {nid_tag}"
        _text(canvas, tag, (x1, y1 - 4), C['sfn'], scale=0.33)

    rest_short = {"WHOLE_HALF": "W/H", "QUARTER": "Q", "EIGHTH": "8", "SIXTEENTH": "16", "THIRTY_SECOND": "32", "SIXTY_FOURTH": "64", "WHOLE": "W", "HALF": "H"}
    # Layer 4: vẽ rest và đánh dấu tâm để dễ kiểm tra sai lệch bbox.
    for rs in rests:
        if rs.bbox is None:
            continue
        _rect(canvas, rs.bbox, C['rest'], thickness=2)
        x1, y1, x2, y2 = rs.bbox
        lbl_name = rs.label.name if rs.label is not None else '?'
        short = rest_short.get(lbl_name, lbl_name)
        dot_tag = '.' if rs.has_dot else ''
        tag = f"REST {short}{dot_tag} T{rs.track}"
        _text(canvas, tag, (x1, y1 - 5), C['rest'], scale=0.35)
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        cv2.drawMarker(canvas, (cx, cy), C['rest'], cv2.MARKER_DIAMOND, 6, 1)

    stats = (f"Barlines={len(barlines)}  Clefs={len(clefs)}  SFN={len(sfns)}  Rests={len(rests)}")
    _text(canvas, stats, (6, canvas.shape[0] - 8), C['white'], scale=0.42)

    legend_items = [
        (f"Barlines ({len(barlines)})", C['barline']),
        (f"Clefs ({len(clefs)})", C['clef']),
        (f"Accidentals ({len(sfns)})", C['sfn']),
        (f"Rests ({len(rests)})", C['rest']),
    ]
    _legend(canvas, legend_items, x0=6, y0=18, dy=16)
    _save(out_dir, 'step4_symbols', canvas)


def gen_clefs(bboxes: List[BBox], labels: List[str]) -> List[Clef]:
    """Sinh danh sách Clef object từ bbox + nhãn model, kèm thông tin track/group."""
    # Ánh xạ nhãn model -> enum nghiệp vụ, đồng thời gắn track/group theo staff gần nhất.
    name_type_map = {
        "gclef": ClefType.G_CLEF,
        "fclef": ClefType.F_CLEF
    }
    clefs = []
    for box, label in zip(bboxes, labels):
        st1, _ = find_closest_staffs(*get_center(box))
        cc = Clef()
        cc.bbox = box
        cc.label = name_type_map[label]
        cc.track = st1.track
        cc.group = st1.group
        clefs.append(cc)
    return clefs


def get_nearby_note_id(box: BBox, note_id_map: ndarray) -> Union[int, None]:
    """
    Tìm note_id gần accidental theo heuristic quét ngang sang phải.

    Giả định nghiệp vụ: accidental thường đứng ngay bên trái note mà nó tác động.
    Nếu không tìm thấy trong phạm vi 1 unit_size thì trả về None.
    """
    # Heuristic liên kết accidental với note:
    # quét ngang sang phải từ mép phải accidental trong phạm vi ~1 unit_size.
    cen_x, cen_y = get_center(box)
    unit_size = int(round(get_unit_size(cen_x, cen_y)))
    nid = None
    # Tìm note ở phía bên phải accidental trong khoảng 1 unit.
    for x in range(box[2], box[2]+unit_size):
        is_in_range = (0 <= cen_y < note_id_map.shape[0]) and (0 <= x < note_id_map.shape[1])
        if not is_in_range:
            continue
        if note_id_map[cen_y, x] != -1:
            nid = note_id_map[cen_y, x]
            break
    return nid


def gen_sfns(bboxes: List[BBox], labels: List[str]) -> List[Sfn]:
    """
    Sinh accidental object và liên kết với note nếu tìm được note_id lân cận.

    Khi track/group accidental không khớp note đích, đánh dấu note.invalid
    để downstream có thể bỏ qua hoặc xử lý lại.
    """
    # Sinh object accidental và thử gắn vào note tương ứng.
    # Nếu track/group không khớp thì đánh dấu note invalid để hậu xử lý.
    note_id_map = layers.get_layer('note_id')
    notes = layers.get_layer('notes')

    name_type_map = {
        "sharp": SfnType.SHARP,
        "flat": SfnType.FLAT,
        "natural": SfnType.NATURAL
    }
    sfns = []
    for box, label in zip(bboxes, labels):
        st1, _ = find_closest_staffs(*get_center(box))
        ss = Sfn()
        ss.bbox = box
        ss.label = name_type_map[label]
        ss.note_id = get_nearby_note_id(box, note_id_map)
        ss.track = st1.track
        ss.group = st1.group

        # Nếu tìm được note kề bên thì gán accidental trực tiếp vào note đó.
        if ss.note_id is not None:
            note = notes[ss.note_id]
            if ss.track != note.track:
                print(f"Track of sfn and note mismatch: {ss}\n{note}") 
                notes[ss.note_id].invalid = True
            elif ss.group != note.group:
                print(f"Group of sfn and note mismatch: {ss}\n{note}")
                notes[ss.note_id].invalid = True
            else:
                notes[ss.note_id].sfn = ss.label
                ss.is_key = False

        sfns.append(ss)
    return sfns


def gen_rests(bboxes: List[BBox], labels: List[str]) -> List[Rest]:
    """
    Sinh Rest object và dò chấm dôi (augmentation dot).

    Heuristic dot:
    - Tìm vùng hẹp bên phải rest trong phạm vi 1 unit_size.
    - Tổng pixel > 0 nhưng nhỏ hơn ngưỡng (unit_size^2/7) thì xem là chấm dôi.
    """
    # Sinh object rest và kiểm tra chấm dôi bằng vùng lân cận bên phải ký hiệu.
    symbols = layers.get_layer('symbols_pred')

    name_type_map = {
        "rest_whole": RestType.WHOLE_HALF,
        "rest_quarter": RestType.QUARTER,
        "rest_8th": RestType.EIGHTH,
        "rest_16th": RestType.SIXTEENTH,
        "rest_32nd": RestType.THIRTY_SECOND,
        "rest_64th": RestType. SIXTY_FOURTH
    }
    rests = []
    for box, label in zip(bboxes, labels):
        st1, _ = find_closest_staffs(*get_center(box))
        rr = Rest()
        rr.bbox = box
        rr.label = name_type_map[label]
        rr.track = st1.track
        rr.group = st1.group

        # Dò chấm dôi ở bên phải rest bằng ngưỡng diện tích nhỏ.
        unit_size = int(round(get_unit_size(*get_center(box))))
        dot_range = range(box[2]+1, min(box[2]+unit_size, symbols.shape[1] - 1))
        dot_region = symbols[box[1]:box[3], dot_range]
        if 0 < np.sum(dot_region) < unit_size**2 / 7:
            rr.has_dot = True

        rests.append(rr)
    return rests


def extract(min_barline_h_unit_ratio: float = 3.75) -> Tuple[List[Barline], List[Clef], List[Sfn], List[Rest]]:
    """
    Hàm điều phối chính cho bước symbol extraction.

    Trả về tuple theo thứ tự:
    (barlines, clefs, sfns, rests)

    Thứ tự pipeline được giữ cố định vì:
    - Rest extraction cần line_box (barline) để loại vùng gây nhiễu.
    - SFN/Clef nên được gán sớm để hỗ trợ các bước nhạc lý phía sau.
    """
    # Pipeline tổng: tách barline -> clef/key -> rest, rồi sinh object kết quả.
    # Đây là điểm vào chính của module symbol_extraction.
    symbols = layers.get_layer('symbols_pred')
    stems_rests = layers.get_layer('stems_rests_pred')
    clefs_keys = layers.get_layer('clefs_keys_pred')
    group_map = layers.get_layer('group_map')

    line_box = parse_barlines(group_map, stems_rests, symbols, min_height_unit_ratio=min_barline_h_unit_ratio)
    barlines = gen_barlines(line_box)

    unit_size = get_global_unit_size()
    clef_box, key_box, clef_label, key_label = parse_clefs_keys(clefs_keys, unit_size)
    clefs = gen_clefs(clef_box, clef_label)
    sfns = gen_sfns(key_box, key_label)

    rest_box, rest_label = parse_rests(line_box, unit_size)
    rests = gen_rests(rest_box, rest_label)

    return barlines, clefs, sfns, rests





