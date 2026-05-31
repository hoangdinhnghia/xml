from typing import Tuple, TYPE_CHECKING
import os
import logging

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from . import layers
from typing import Dict

if TYPE_CHECKING:
    # Chỉ import khi kiểm tra kiểu để tránh vòng lặp import lúc chạy thật.
    from oemer.staffline_extraction import Staff


def get_logger(name, level="warn"):
    """Hàm tiện ích tạo logger tương thích ngược.

    Hàm này được chuyển từ `oemer.logger` sang đây để các module cũ vẫn có
    thể gọi `get_logger` từ `oemer.utils` mà không cần đổi import.

    Cách hoạt động:
    - Đọc mức log từ biến môi trường `LOG_LEVEL` (nếu có), ưu tiên hơn tham số `level`.
    - Dùng định dạng thông điệp khác nhau cho từng mức log.
    - Gỡ các `StreamHandler` cũ rồi gắn handler mới để tránh in log lặp nhiều lần.
    """
    logger = logging.getLogger(name)
    level = os.environ.get("LOG_LEVEL", level)

    msg_formats = {
        "debug": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "info": "%(asctime)s %(message)s  [at %(filename)s:%(lineno)d]",
        "warn": "%(asctime)s %(message)s",
        "warning": "%(asctime)s %(message)s",
        "error": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "critical": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
    }
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=msg_formats[level.lower()], datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if len(logger.handlers) > 0:
        rm_idx = [idx for idx, handler in enumerate(logger.handlers) if isinstance(handler, logging.StreamHandler)]
        for idx in rm_idx:
            del logger.handlers[idx]
    logger.addHandler(handler)
    logger.setLevel(level_mapping[level.lower()])
    return logger


def _flatten_valid_staffs(staffs) -> list['Staff']:
    """Làm phẳng cấu trúc `staffs` và loại bỏ phần tử `None`.

    Một số lớp trung gian lưu `staffs` ở dạng mảng nhiều chiều; hàm này đưa
    về danh sách 1 chiều để các bước xử lý tiếp theo (tìm staff gần nhất,
    nội suy unit size, đếm track) dùng chung dễ dàng.
    """
    flat = np.array(staffs, dtype=object).reshape(-1)
    return [st for st in flat if st is not None]


def _ensure_dir(path: str) -> None:
    """Đảm bảo thư mục đích tồn tại trước khi ghi file."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_intermediate(out_dir: str, artifacts: Dict[str, np.ndarray]) -> None:
    """Lưu các kết quả trung gian (mask/ảnh) ra thư mục `out_dir`.

    `artifacts` là ánh xạ `tên -> ndarray`.

    Quy ước lưu:
    - Mảng 2D: lưu PNG trực quan (thang 0..255) và lưu thô `.npy`.
    - Mảng 3D có 3 kênh: coi như ảnh màu, lưu PNG (đổi RGB -> BGR cho OpenCV) và `.npy`.
    - Mảng 3D nhiều kênh xác suất: trực quan kênh đầu tiên thành PNG và vẫn lưu `.npy` đầy đủ.

    Mục tiêu là vừa có file dễ xem nhanh (PNG), vừa có dữ liệu gốc để debug sâu (`.npy`).
    """
    _ensure_dir(out_dir)
    for name, arr in artifacts.items():
        target_png = os.path.join(out_dir, f"{name}.png")
        target_npy = os.path.join(out_dir, f"{name}.npy")
        try:
            a = np.array(arr)
            if a.ndim == 2:
                # Bản đồ nhị phân/xác suất: chuẩn hóa về dải 0..255 để lưu PNG.
                if a.dtype != np.uint8:
                    # Nếu là float thì chuẩn hóa theo giá trị cực đại hiện có.
                    if np.issubdtype(a.dtype, np.floating):
                        ma = np.max(a) if np.max(a) != 0 else 1.0
                        a = (a / ma * 255).astype(np.uint8)
                    else:
                        a = (a.astype(np.uint8) * 255)
                cv2.imwrite(target_png, a)
            elif a.ndim == 3:
                # Giả định định dạng ảnh màu HWC.
                if a.shape[2] == 3:
                    cv2.imwrite(target_png, a[..., ::-1])  # Đổi RGB -> BGR theo chuẩn OpenCV.
                else:
                    # Nhiều kênh xác suất: lấy kênh đầu làm ảnh trực quan nhanh.
                    ch = a[..., 0]
                    ma = np.max(ch) if np.max(ch) != 0 else 1.0
                    cv2.imwrite(target_png, (ch / ma * 255).astype(np.uint8))
            # Luôn lưu thêm bản mảng gốc dạng `.npy` để tái sử dụng chính xác.
            np.save(target_npy, arr)
        except Exception:
            logging.warning("Failed to save intermediate artifact: %s", name)

def count(data, intervals):
    """Đếm số phần tử rơi vào từng khoảng giá trị.

    Quy trình:
    - Sắp xếp dữ liệu đầu vào.
    - Bổ sung biên trái/phải bằng min/max của dữ liệu.
    - Với từng cặp biên liên tiếp `[l, r)`, đếm số phần tử thoả `l <= x < r`.
    """
    occur = []
    data = np.sort(data)
    intervals = np.insert(intervals, [0, len(intervals)], [np.min(data), np.max(data)])
    for idx in range(len(intervals[:-1])):
        sub = data[data>=intervals[idx]]
        sub = sub[sub<intervals[idx+1]]
        occur.append(len(sub))
    return occur


def find_closest_staffs(x: int, y: int) -> Tuple['Staff', 'Staff']:
    """Tìm 2 staff gần điểm `(x, y)` nhất để hỗ trợ nội suy.

    Chiến lược:
    - Lấy toàn bộ staff hợp lệ trong layer `staffs`.
    - Sắp xếp theo độ gần với điểm truy vấn.
    - Với nhiều ứng viên, dùng vị trí `y_upper/y_lower/y_center` để chọn cặp staff
      nằm cùng phía hợp lý (trên/dưới) quanh điểm đang xét.
    """
    staffs = layers.get_layer('staffs')

    staffs = _flatten_valid_staffs(staffs)
    if len(staffs) == 0:
        raise ValueError("No valid staff objects are available in layer 'staffs'.")

    diffs = sorted(staffs, key=lambda st: st - [x, y])
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    elif len(diffs) == 2:
        return (diffs[0], diffs[1])

    # Trường hợp có từ 3 ứng viên trở lên, dùng heuristic vị trí theo trục y.
    first = diffs[0]
    second = diffs[1]
    third = diffs[2]
    if abs(first.y_lower - y) <= abs(first.y_upper - y):
        # Điểm gần biên dưới của staff gần nhất.
        if second.y_center > first.y_center:
            return first, second
        elif third.y_center > first.y_center:
            return first, third
        else:
            return first, first
    else:
        # Điểm gần biên trên của staff gần nhất.
        if second.y_center < first.y_center:
            return first, second
        elif third.y_center < first.y_center:
            return first, third
        else:
            return first, first


def get_unit_size(x: int, y: int) -> float:
    """Ước lượng `unit_size` tại vị trí `(x, y)`.

    - Nếu chỉ có 1 staff hiệu lực quanh điểm: trả về trực tiếp `unit_size` staff đó.
    - Nếu điểm nằm trong dải staffline: dùng `unit_size` của staff chứa điểm.
    - Nếu điểm nằm ngoài: nội suy tuyến tính theo khoảng cách tới tâm 2 staff gần nhất.
    """
    st1, st2 = find_closest_staffs(x, y)
    if st1.y_center == st2.y_center:
        return float(st1.unit_size)

    # Điểm nằm trong vùng staffline của staff thứ nhất.
    if st1.y_upper <= y <= st1.y_lower:
        return float(st1.unit_size)

    # Điểm nằm ngoài vùng staffline: nội suy tuyến tính theo khoảng cách.
    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    w1 = dist1 / (dist1 + dist2)
    w2 = dist2 / (dist1 + dist2)
    unit_size = w1 * st1.unit_size + w2 * st2.unit_size
    return float(unit_size)


def get_global_unit_size() -> float:
    """Tính `unit_size` trung bình toàn cục từ tất cả staff hợp lệ."""
    staffs = layers.get_layer('staffs')
    usize = [st.unit_size for st in _flatten_valid_staffs(staffs)]
    if not usize:
        raise ValueError("No valid staff objects are available in layer 'staffs'.")
    return sum(usize) / len(usize)


def get_total_track_nums() -> int:
    """Đếm tổng số track khác nhau trong layer `staffs`."""
    staffs = layers.get_layer('staffs')
    tracks = [st.track for st in _flatten_valid_staffs(staffs)]
    return len(set(tracks))


def remove_stems(data):
    """Làm suy yếu/xóa nét dọc (stem) bằng phép hình thái học đơn giản.

    Dùng kernel ngang `(5, 1)` với chuỗi `erode -> dilate` để giữ cấu trúc ngang
    tốt hơn và giảm ảnh hưởng của thành phần dọc mảnh.
    """
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.dilate(cv2.erode(data.astype(np.uint8), ker), ker)


def estimate_degree(points, **kwargs):
    """Ước lượng góc nghiêng của tập điểm 2D bằng hồi quy tuyến tính.

    Đầu vào là danh sách toạ độ `(x, y)`. Hàm fit mô hình `y = ax + b` bằng
    `LinearRegression`, sau đó chuyển hệ số góc `a` sang góc (độ).
    """
    points = np.array(points)
    model = LinearRegression(**kwargs)
    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    return slope_to_degree(model.coef_[0])


def slope_to_degree(y_diff: int, x_diff: int) -> float:
    """Chuyển chênh lệch trục `(y_diff, x_diff)` sang góc tính theo độ."""
    return np.rad2deg(np.arctan2(y_diff, x_diff))


# ----------------------- Nhóm hàm trực quan hóa ------------------------
# Các tiện ích vẽ nhỏ dùng chung cho `ete.py` và các extractor.
# Mục tiêu là gom logic hiển thị vào một nơi để tránh lặp code.
C = {
    "staff_line":  (255, 180,  60),
    "zone":        (0,   220, 220),
    "note_whole":  (80,  200, 255),
    "note_half":   (0,   200, 80),
    "note_quarter":(50,   50, 255),
    "note_other":  (200,  80, 200),
    "stem_up":     (0,   255, 180),
    "stem_down":   (0,   120, 255),
    "barline":     (50,   50, 255),
    "clef":        (255, 220,   0),
    "sfn":         (255,   0, 200),
    "rest":        (0,   165, 255),
    "beam":        (100, 255, 100),
    "dot":         (0,   200, 255),
    "white":       (255, 255, 255),
    "black":       (0,     0,   0),
    "gray":        (160,  160, 160),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _overlay(canvas: np.ndarray, mask: np.ndarray,
             color: tuple, alpha: float = 0.40) -> np.ndarray:
    """Chồng một `mask` nhị phân lên `canvas` bằng màu và hệ số alpha.

    Chỉ các pixel có `mask > 0` mới bị pha màu; các vùng khác giữ nguyên.
    """
    if mask is None:
        return canvas
    try:
        colored = np.zeros_like(canvas, dtype=np.uint8)
        colored[mask > 0] = color
        m = mask > 0
        canvas[m] = (canvas[m] * (1 - alpha) + colored[m] * alpha).astype(np.uint8)
    except Exception:
        # Phòng thủ: nếu mask sai kích thước thì bỏ qua để không làm hỏng pipeline.
        pass
    return canvas


def _rect(canvas, bbox, color, thickness=1):
    """Vẽ hình chữ nhật theo định dạng bbox `(x1, y1, x2, y2)`."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)


def _text(canvas, txt, pos, color, scale=0.45, thickness=1, font=FONT):
    """Vẽ chữ có viền đen mỏng để tăng độ tương phản khi hiển thị."""
    cv2.putText(canvas, txt, pos, font, scale, C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, txt, pos, font, scale, color,    thickness,     cv2.LINE_AA)


def _legend(canvas, items: list, x0: int = 6, y0: int = 20, dy: int = 18):
    """Vẽ bảng chú giải nhỏ (ô màu + nhãn) lên ảnh `canvas`."""
    for i, (label, color) in enumerate(items):
        y = y0 + i * dy
        cv2.rectangle(canvas, (x0, y - 10), (x0 + 14, y + 2), color, -1)
        _text(canvas, label, (x0 + 18, y), color, scale=0.40)


def _save(out_dir: str, name: str, img: np.ndarray) -> None:
    """Lưu ảnh trực quan hóa ra `out_dir` dưới dạng PNG."""
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}.png")
    try:
        cv2.imwrite(path, img)
    except Exception:
        logging.getLogger(__name__).exception('Failed to write image: %s', path)
    else:
        logging.getLogger(__name__).info("[viz] Saved: %s", path)

