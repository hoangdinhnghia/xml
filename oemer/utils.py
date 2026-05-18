from typing import Tuple, TYPE_CHECKING
import os
import logging

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from . import layers
from typing import Dict

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports at runtime
    from oemer.staffline_extraction import Staff


def get_logger(name, level="warn"):
    """Compatibility logger helper (moved from oemer.logger).

    Kept here so other modules can import get_logger from oemer.utils.
    The function preserves the previous formatting and LOG_LEVEL behaviour.
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
    flat = np.array(staffs, dtype=object).reshape(-1)
    return [st for st in flat if st is not None]


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_intermediate(out_dir: str, artifacts: Dict[str, np.ndarray]) -> None:
    """Save intermediate artifacts (masks and images) to `out_dir`.

    artifacts: mapping from name -> ndarray. For 2D arrays they will be
    saved as PNG (scaled 0/255) and as .npy; for 3-channel images saved as PNG and .npy.
    """
    _ensure_dir(out_dir)
    for name, arr in artifacts.items():
        target_png = os.path.join(out_dir, f"{name}.png")
        target_npy = os.path.join(out_dir, f"{name}.npy")
        try:
            a = np.array(arr)
            if a.ndim == 2:
                # binary/prob maps - scale to 0..255
                if a.dtype != np.uint8:
                    # normalize if float
                    if np.issubdtype(a.dtype, np.floating):
                        ma = np.max(a) if np.max(a) != 0 else 1.0
                        a = (a / ma * 255).astype(np.uint8)
                    else:
                        a = (a.astype(np.uint8) * 255)
                cv2.imwrite(target_png, a)
            elif a.ndim == 3:
                # assume color image in RGB or HWC
                if a.shape[2] == 3:
                    cv2.imwrite(target_png, a[..., ::-1])  # RGB->BGR
                else:
                    # multi-channel probability map: save a visualization (first channel)
                    ch = a[..., 0]
                    ma = np.max(ch) if np.max(ch) != 0 else 1.0
                    cv2.imwrite(target_png, (ch / ma * 255).astype(np.uint8))
            # save npy
            np.save(target_npy, arr)
        except Exception:
            logging.warning("Failed to save intermediate artifact: %s", name)

def count(data, intervals):
    """Count elements in different intervals"""
    occur = []
    data = np.sort(data)
    intervals = np.insert(intervals, [0, len(intervals)], [np.min(data), np.max(data)])
    for idx in range(len(intervals[:-1])):
        sub = data[data>=intervals[idx]]
        sub = sub[sub<intervals[idx+1]]
        occur.append(len(sub))
    return occur


def find_closest_staffs(x: int, y: int) -> Tuple['Staff', 'Staff']:
    staffs = layers.get_layer('staffs')

    staffs = _flatten_valid_staffs(staffs)
    if len(staffs) == 0:
        raise ValueError("No valid staff objects are available in layer 'staffs'.")

    diffs = sorted(staffs, key=lambda st: st - [x, y])
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    elif len(diffs) == 2:
        return (diffs[0], diffs[1])

    # There are over three candidates
    first = diffs[0]
    second = diffs[1]
    third = diffs[2]
    if abs(first.y_lower - y) <= abs(first.y_upper - y):
        # Closer to the lower bound of the first candidate.
        if second.y_center > first.y_center:
            return first, second
        elif third.y_center > first.y_center:
            return first, third
        else:
            return first, first
    else:
        # Closer to the upper bound of the first candidate.
        if second.y_center < first.y_center:
            return first, second
        elif third.y_center < first.y_center:
            return first, third
        else:
            return first, first


def get_unit_size(x: int, y: int) -> float:
    st1, st2 = find_closest_staffs(x, y)
    if st1.y_center == st2.y_center:
        return float(st1.unit_size)

    # Within the stafflines
    if st1.y_upper <= y <= st1.y_lower:
        return float(st1.unit_size)

    # Outside stafflines.
    # Infer the unit size by linear interpolation.
    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    w1 = dist1 / (dist1 + dist2)
    w2 = dist2 / (dist1 + dist2)
    unit_size = w1 * st1.unit_size + w2 * st2.unit_size
    return float(unit_size)


def get_global_unit_size() -> float:
    staffs = layers.get_layer('staffs')
    usize = [st.unit_size for st in _flatten_valid_staffs(staffs)]
    if not usize:
        raise ValueError("No valid staff objects are available in layer 'staffs'.")
    return sum(usize) / len(usize)


def get_total_track_nums() -> int:
    staffs = layers.get_layer('staffs')
    tracks = [st.track for st in _flatten_valid_staffs(staffs)]
    return len(set(tracks))


def remove_stems(data):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.dilate(cv2.erode(data.astype(np.uint8), ker), ker)


def estimate_degree(points, **kwargs):
    """Accepts list of (x, y) coordinates."""
    points = np.array(points)
    model = LinearRegression(**kwargs)
    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    return slope_to_degree(model.coef_[0])


def slope_to_degree(y_diff: int, x_diff: int) -> float:
    return np.rad2deg(np.arctan2(y_diff, x_diff))
