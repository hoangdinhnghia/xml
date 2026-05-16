
from typing import Union, Any, List, Tuple, Dict

import cv2
from cv2.typing import RotatedRect
import numpy as np
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering


BBox = Tuple[int, int, int, int]


def get_bbox(data: ndarray) -> List[BBox]:
    """Return bounding boxes (x1,y1,x2,y2) for non-zero connected regions.

    Safe for empty input; returns empty list when no contours found.
    """
    if data is None:
        return []
    contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes: List[BBox] = []
    if contours is None:
        return bboxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((int(x), int(y), int(x + w), int(y + h)))
    return bboxes


def get_center(bbox: Union[BBox, ndarray]) -> Tuple[int, int]:
    """Return center (x, y) of a bbox or a numpy array [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cen_x = int(round((x1 + x2) / 2))
    cen_y = int(round((y1 + y2) / 2))
    return cen_x, cen_y


def get_edge(data):
    if len(data.shape) == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = cv2.GaussianBlur(data, (5, 5), 0)
    data = cv2.Canny(data, 10, 80)
    return data


def merge_nearby_bbox(bboxes: List[BBox], distance: float, x_factor: int = 1, y_factor: int = 1) -> List[BBox]:
    """Merge nearby bboxes using agglomerative clustering on bbox centers.

    Keeps original behavior but is robust to empty input.
    """
    if not bboxes:
        return []
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, compute_full_tree=True)
    centers = np.array([((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) for bb in bboxes])
    centers[:, 0] *= x_factor
    centers[:, 1] *= y_factor
    model.fit(centers)
    labels = np.unique(model.labels_)
    new_box: List[BBox] = []
    for label in labels:
        idx = np.where(model.labels_ == label)[0]
        xs = [bboxes[i][0] for i in idx] + [bboxes[i][2] for i in idx]
        ys = [bboxes[i][1] for i in idx] + [bboxes[i][3] for i in idx]
        x1, x2 = int(np.min(xs)), int(np.max(xs))
        y1, y2 = int(np.min(ys)), int(np.max(ys))
        new_box.append((x1, y1, x2, y2))
    return new_box


def rm_merge_overlap_bbox(
    bboxes: List[BBox],
    mode: str = 'remove',
    overlap_ratio: float = 0.5
) -> List[BBox]:
    """Remove or merge bounding boxes that significantly overlap.

    Uses pairwise overlap checks (IoU-style) to avoid allocating large masks.
    Keeps behavior compatible with original function: larger boxes take precedence.
    """
    assert mode in ['remove', 'merge'], mode
    if not bboxes:
        return []

    # Prepare boxes sorted by area descending
    infos = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        infos.append({'bbox': tuple(map(int, box)), 'area': area})
    infos.sort(key=lambda it: it['area'], reverse=True)

    selected: List[Dict[str, Any]] = []

    def overlap_area(a: BBox, b: BBox) -> int:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    for info in infos:
        box = info['bbox']
        area = info['area']
        keep = True
        for sel in selected:
            sel_box = sel['bbox']
            sel_area = sel['area']
            ov = overlap_area(box, sel_box)
            # ratio relative to the smaller (original behavior used overlap/area_size)
            if area == 0:
                continue
            ratio = ov / area
            if ratio > overlap_ratio:
                if mode == 'merge':
                    # expand selected box to include current
                    x1 = min(sel_box[0], box[0])
                    y1 = min(sel_box[1], box[1])
                    x2 = max(sel_box[2], box[2])
                    y2 = max(sel_box[3], box[3])
                    sel['bbox'] = (x1, y1, x2, y2)
                    sel['area'] = (x2 - x1) * (y2 - y1)
                keep = False
                break
        if keep:
            selected.append({'bbox': box, 'area': area})

    return [sel['bbox'] for sel in selected]


def find_lines(data: ndarray, min_len: int = 10, max_gap: int = 20) -> List[BBox]:
    assert len(data.shape) == 2, f"{type(data)} {data.shape}"
    lines = cv2.HoughLinesP(data.astype(np.uint8), 1, np.pi / 180, 50, None, min_len, max_gap)
    new_line: List[BBox] = []
    if lines is None:
        return new_line
    for arr in lines:
        x1, y1, x2, y2 = arr[0]
        if x1 <= x2:
            top_x, bt_x = int(x1), int(x2)
        else:
            top_x, bt_x = int(x2), int(x1)
        if y1 <= y2:
            top_y, bt_y = int(y1), int(y2)
        else:
            top_y, bt_y = int(y2), int(y1)
        new_line.append((top_x, top_y, bt_x, bt_y))
    return new_line


def draw_lines(lines: ndarray, ori_img: ndarray, width: int = 3) -> ndarray:
    img = ori_img.copy()
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), width, cv2.LINE_AA)
    return img


def to_rgb_img(data: ndarray) -> ndarray:
    """Return a 3-channel uint8 white image with non-zero pixels drawn black."""
    if data is None:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    if data.ndim >= 3:
        return data.astype(np.uint8)
    img = np.ones(data.shape + (3,), dtype=np.uint8) * 255
    idx = np.where(data > 0)
    img[idx[0], idx[1]] = 0
    return img


def draw_bounding_boxes(
    bboxes: List[BBox], 
    img: ndarray, 
    color: Tuple[int, int, int] = (0, 255, 0), 
    width: int = 2, 
    inplace: bool = False
) -> ndarray:
    if len(img.shape) < 3:
        img = to_rgb_img(img)
    if not inplace:
        img = np.array(img)
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, width)
    return img


def get_rotated_bbox(data: ndarray) -> List[RotatedRect]:
    if data is None:
        return []
    contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes: List[RotatedRect] = []
    if contours is None:
        return bboxes
    for cnt in contours:
        bboxes.append(cv2.minAreaRect(cnt))
    return bboxes


def draw_rotated_bounding_boxes(bboxes: List[RotatedRect], img: ndarray, color=(0, 255, 0), width=2, inplace=False) -> ndarray:
    if len(img.shape) < 3:
        img = to_rgb_img(img)
    if not inplace:
        img = np.array(img)
    for rbox in bboxes:
        box = cv2.boxPoints(rbox).astype(np.int64)
        cv2.drawContours(img, [box], 0, color, width)
    return img
    