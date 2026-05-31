from typing import Tuple, List, Any, Dict
import random
import math

import cv2
from cv2.typing import RotatedRect
import scipy.ndimage
import numpy as np
from numpy import ndarray

from oemer import layers
from oemer.utils import get_unit_size
from oemer.bbox import BBox, get_center, get_rotated_bbox, to_rgb_img, draw_bounding_boxes
from oemer.notehead_extraction import NoteType
from oemer.symbol_extraction import morph_open, morph_close
from oemer.utils import _save, C, _overlay, _rect, _text, _legend

# Biến toàn cục phục vụ gỡ lỗi trực quan hóa
dot_img: ndarray
ratio_img: ndarray
beam_img: ndarray
ratio_map: ndarray


def scan_dot(
        symbols: ndarray, 
        note_id_map: ndarray, 
        bbox: BBox, 
        unit_size: float, 
        min_count: int, 
        max_count: int
) -> bool:
    """
    Quét vùng bên phải nốt để phát hiện chấm dôi.

        Đầu vào:
        - symbols: map nhị phân ký hiệu sau tiền xử lý.
        - note_id_map: map id note, dùng để tránh quét đè sang note lân cận.
        - bbox: bbox nốt hiện tại.
        - unit_size: kích thước chuẩn cục bộ của staff.
        - min_count/max_count: ngưỡng pixel để nhận diện chấm dôi.

        Ý tưởng:
    - Từ mép phải bbox nốt, mở rộng sang phải cho tới khi gặp note khác
      hoặc vượt giới hạn theo unit_size.
    - Tính tổng pixel trong cửa sổ quét; nếu nằm trong [min_count, max_count]
      thì coi là có chấm dôi.

        Đầu ra:
        - True nếu phát hiện dot hợp lệ, ngược lại False.
    """
    right_bound = bbox[2] + 1
    start_y = bbox[1] - round(unit_size / 2)
    while True:
        # Tìm biên phải xa nhất cho cửa sổ quét dot.
        # Bề rộng phải nhỏ hơn unit_size và không chạm note lân cận.
        try:
            cur_scan_line = note_id_map[int(start_y):int(bbox[3]), int(right_bound)]
        except IndexError as e:
            print(e)
            break

        ids = set(np.unique(cur_scan_line))
        if -1 in ids:
            ids.remove(-1)
        if len(ids) > 0:
            break
        right_bound += 1
        if right_bound >= bbox[2] + unit_size:
            break

    left_bound = bbox[2] + round(unit_size * 0.4)
    dot_region = symbols[int(start_y):int(bbox[3]), int(left_bound):int(right_bound)]
    pixels = np.sum(dot_region)
    if min_count <= pixels <= max_count:
        color = (255, random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(dot_img, (int(left_bound), int(start_y)), (int(right_bound), int(bbox[3])), color, 1)
        msg = f"{min_count:.2f}/{pixels:.2f}/{max_count:.2f}"
        cv2.putText(dot_img, msg, (int(bbox[0]), int(bbox[3]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        return True

    # Khối debug cũ (để tham khảo): vẽ cửa sổ quét dot và số pixel.
    # cv2.rectangle(temp, (bbox[2]+1, start_y), (right_bound, bbox[3]), color, 1)
    # if pixels > 0:
    #     msg = f"{min_count:.2f}/{pixels:.2f}/{max_count:.2f}"
    #     cv2.putText(temp, msg, (bbox[0], bbox[3]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    return False


def parse_dot(min_area_ratio: float = 0.08, max_area_ratio: float = 0.2) -> None:
    """
    Gán thuộc tính has_dot cho từng nốt dựa trên ảnh symbols đã khử stem/clef.

    Cơ chế:
    - Tiền xử lý morphology để giảm nhiễu hạt.
    - Quét chấm dôi cho từng note trong từng group.
    - Với group có stem rõ ràng, ép đồng nhất trạng thái dot nếu đa số nghiêng về 1 phía.

    Ghi chú heuristic:
    - min_area_ratio/max_area_ratio tỉ lệ theo unit_size^2 để scale theo kích thước bản nhạc.
    - Majority vote trong cùng group giúp giảm sai số cục bộ từng nốt.
    """
    # Lấy dữ liệu cần thiết
    groups = layers.get_layer('note_groups')
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    clefs_sfns = layers.get_layer('clefs_keys_pred')
    notes = layers.get_layer('notes')
    note_id_map = layers.get_layer('note_id')

    # Tạo map ký hiệu không chứa stem/clef để chấm dôi nổi bật hơn.
    no_stem = np.where(symbols-stems-clefs_sfns>0, 1, 0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    no_stem = cv2.dilate(cv2.erode(no_stem.astype(np.uint8), ker), ker)
    global dot_img
    dot_img = to_rgb_img(no_stem)

    # Dò chấm dôi cạnh từng nốt trong mỗi cụm group.
    for group in groups:
        nids = group.note_ids
        gbox = group.bbox
        unit_size = get_unit_size(*get_center(gbox))
        nbox = np.array([notes[nid].bbox for nid in nids])
        nbox = (np.min(nbox[:, 0]), np.min(nbox[:, 1]), np.max(nbox[:, 2]), np.max(nbox[:, 3]))  # type: ignore
        min_count = round(unit_size**2 * min_area_ratio)
        max_count = round(unit_size**2 * max_area_ratio)

        dots = []
        for nid in nids:
            bbox = notes[nid].bbox
            bbox = (bbox[0], bbox[1], max(bbox[2], gbox[2]), bbox[3])
            has_dot = scan_dot(no_stem, note_id_map, bbox, unit_size, min_count, max_count)
            dots.append(has_dot)
            notes[nid].has_dot = has_dot

        # Các nốt cùng stem trong một group thường cùng trường độ, nên dot cũng đồng bộ.
        all_same = not (all(dots) ^ any(dots))  # Các note trong group đồng nhất có/không có dot
        if group.stem_up is not None:
            if not all_same:
            # Các note cùng stem nên có cùng trường độ.
                true_count = len([dot for dot in dots if dot])
                false_count = len(dots) - true_count
                to_dot = (true_count >= false_count)
                for nid in nids:
                    notes[nid].has_dot = to_dot


def polish_symbols(
        staff_pred: ndarray, 
        symbols: ndarray, 
        stems: ndarray, 
        clefs_sfns: ndarray, 
        group_map: ndarray
) -> ndarray:
    """
    Làm sạch bản đồ symbols để chuẩn bị tách beam/flag.

    Thuật toán:
    - Khử notehead, stem, clef/sfn khỏi symbols.
    - Mở/đóng hình thái học để cắt nối giả và hàn nét đứt.
    - Trộn lại với stem mở rộng để giữ cấu trúc nhịp cần thiết.
    """
    st_width = 5
    beams_in_staff = morph_open(staff_pred, (st_width, 1))

    gp_map = np.where(group_map>-1, 1, 0)
    mix = symbols + beams_in_staff - gp_map - stems - clefs_sfns
    mix = np.where(mix > 0, 1, 0)
    mix = morph_open(mix, (2, 3))  # Loại các kết nối giả giữa beam và slur
    ext_stems = morph_close(stems, (5, 1))
    beams = mix + ext_stems + gp_map
    beams[beams>1] = 1
    return beams


def parse_beams(
        min_area_ratio: float = 0.07, 
        min_tp_ratio: float = 0.4, 
        min_width_ratio: float = 0.2
) -> Tuple[ndarray, List[RotatedRect], ndarray]:
    """
    Trích xuất vùng beam hợp lệ bằng rotated bbox + bộ lọc hình học.

    Bộ lọc chính:
    - Diện tích tối thiểu theo unit_size.
    - Bề rộng tối thiểu của rotated box.
    - Tỉ lệ pixel thật trong contour (true area ratio).

    Trả về:
    - poly_map: map beam hợp lệ dạng nhị phân.
    - valid_box: danh sách rotated boxes hợp lệ.
    - invalid_map: vùng bị loại để dùng ở bước sau nếu cần.

    Ý nghĩa tham số:
    - min_area_ratio: ngưỡng diện tích tối thiểu của contour theo unit_size^2.
    - min_tp_ratio: mật độ pixel thật trong contour (tránh contour rỗng).
    - min_width_ratio: bề dày tối thiểu của beam theo unit_size.
    """
    # Lấy dữ liệu cần thiết
    symbols = layers.get_layer('symbols_pred')
    staff_pred = layers.get_layer('staff_pred')
    stems = layers.get_layer('stems_rests_pred')
    group_map = layers.get_layer('group_map')
    clefs_sfns = layers.get_layer('clefs_keys_pred')

    beams = polish_symbols(staff_pred, symbols, stems, clefs_sfns, group_map)
    beams = beams - np.where(group_map>-1, 1, 0) - stems
    beams[beams<0] = 0

    rboxes = get_rotated_bbox(beams)
    poly_map = np.ones(symbols.shape+(3,), dtype=np.uint8) * 255
    idx = np.where(beams>0)
    poly_map[idx[0], idx[1]] = 0
    invalid_map = np.zeros_like(poly_map)  # Dùng để loại vùng không hợp lệ ở các bước sau.

    global ratio_map
    ratio_map = np.copy(poly_map)

    null_color = (255, 255, 255)
    valid_box: List[RotatedRect] = []
    valid_idxs = []
    idx_map = np.zeros_like(poly_map) - 1
    for box_idx, rbox in enumerate(rboxes):  # type: ignore
        # Dùng truy vết pixel thuộc contour về sau; cần đặt trước mọi nhánh continue.
        box_idx %= 255  # type: ignore
        if box_idx == 0:
            idx_map = np.zeros_like(poly_map) - 1

        # Lấy contour của rotated box
        cnt = cv2.boxPoints(rbox)
        if any(cc < 0 for cc in cnt.reshape(-1, 1).squeeze()):
            # Đảm bảo không có tọa độ âm để tránh tràn số khi ép kiểu unsigned.
            continue
        cnt = cnt.astype(np.uint64)
        centers = np.sum(cnt, axis=0) / 4
        unit_size = get_unit_size(round(centers[0]), round(centers[1]))

        # Lọc theo diện tích để loại contour rất nhỏ.
        cv2.drawContours(ratio_map, [cnt], 0, (255, 0, 0), 2)
        area = cv2.contourArea(cnt)
        min_area = unit_size**2 * min_area_ratio
        if area < min_area:
            cv2.fillPoly(poly_map, [cnt], color=null_color)
            cv2.fillPoly(invalid_map, [cnt], color=null_color)
            continue

        # Mẹo lấy chỉ số các pixel nằm trong vùng contour.
        cv2.fillPoly(idx_map, [cnt], color=(box_idx, 0, 0))
        yi, xi = np.where(idx_map[..., 0] == box_idx)
        pts = beams[yi, xi]
        meta_idx = np.where(pts>0)[0]
        ryi = yi[meta_idx]
        rxi = xi[meta_idx]

        # Lọc theo bề rộng tối thiểu để giảm nhiễu nét mảnh.
        r_width = min(rbox[1])
        if r_width < unit_size * min_width_ratio:
            poly_map[ryi, rxi] = np.array(null_color)
            invalid_map[ryi, rxi] = np.array(null_color)
            ratio_map[ryi, rxi] = np.array((255, 235, 15))
            continue

        # Lọc theo mật độ pixel thật trong contour.
        ratio = len(meta_idx) / (len(yi) + 1e-8)
        if ratio < min_tp_ratio:
            poly_map[ryi, rxi] = np.array(null_color)
            invalid_map[ryi, rxi] = np.array(null_color)
            ratio_map[ryi, rxi] = np.array((0, 150, 255))
            cv2.putText(ratio_map, f"{ratio:.2f}", (cnt[1][0], cnt[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            continue

        valid_idxs.append((ryi, rxi))
        valid_box.append(rbox)

    for ryi, rxi in valid_idxs:
        poly_map[ryi, rxi] = 255 - np.array(null_color)
        ratio_map[ryi, rxi] = 255 - np.array(null_color)

    # Chuyển canvas RGB phụ trợ về map nhị phân 1-kênh cho các bước sau.
    poly_map = np.where(np.sum(poly_map, axis=-1)<1, 1, 0)
    invalid_map = invalid_map[..., 0] / 255
    return poly_map, valid_box, invalid_map


def parse_beam_overlap_regions(poly_map: ndarray, invalid_map: ndarray) -> Tuple[ndarray, Dict[int, Dict[str, Any]]]:
    """
    Gom vùng chồng lấp giữa beam và symbols để liên kết với các nhóm nốt.

    Mục tiêu:
    - Loại vùng không giao với group_map.
    - Tạo map_info chứa bbox tổng hợp và tập gids liên quan cho từng vùng.

    Ghi chú:
    - Dùng connected-components trên cả poly_map và mix để gom vùng liên thông.
    - invalid_map hiện chưa trừ trực tiếp trong mix (giữ linh hoạt cho tuning sau).
    """
    # Lấy dữ liệu cần thiết
    symbols = layers.get_layer('symbols_pred')
    group_map = layers.get_layer('group_map')
    barlines = layers.get_layer('barlines')

    mix = poly_map + symbols #- invalid_map
    mix[mix<0] = 0
    ker = np.ones((3, 3), dtype=np.uint8)
    mix = cv2.dilate(cv2.erode(mix.astype(np.uint8), ker), ker)
    poly_map = cv2.dilate(cv2.erode(poly_map.astype(np.uint8), ker), ker)

    # Xóa barline để tránh nhầm thành beam dọc trong vùng overlap.
    for bl in barlines:
        box = bl.bbox
        mix[box[1]:box[3], box[0]:box[2]] = 0
        poly_map[box[1]:box[3], box[0]:box[2]] = 0

    mix[mix>1] = 1
    reg_map, feat_num = scipy.ndimage.label(poly_map)
    sym_map, _ = scipy.ndimage.label(mix)

    out_map = np.zeros_like(reg_map)
    map_info: Dict = {}
    for idx in range(1, feat_num+1):
        mask = (reg_map == idx)
        sym_labels = set(np.unique(sym_map[mask]))
        if 0 in sym_labels:
            sym_labels.remove(0)

        yi, xi = [], []
        for label in sym_labels:
            yy, xx = np.where(sym_map==label)
            yi.extend(list(yy))
            xi.extend(list(xx))

        g_overlap = group_map[np.array(yi).astype(int), np.array(xi).astype(int)]
        gids = set(np.unique(g_overlap))
        if -1 in gids:
            gids.remove(-1)
        if len(gids) == 0:
            # Loại vùng không giao với bất kỳ nhóm nốt nào.
            continue

        out_map[mask] = 1
        box = (np.min(xi), np.min(yi), np.max(xi), np.max(yi))
        for sym_label in sym_labels:
            if sym_label in map_info:
                bb = map_info[sym_label]['bbox']
                gids.update(map_info[sym_label]['gids'])
                box = (min(bb[0], box[0]), min(bb[1], box[1]), max(bb[2], box[2]), max(bb[3], box[3]))
            map_info[sym_label] = {'bbox': box, 'gids': gids}

    ker = np.ones((3, 3), dtype=np.uint8)
    out_map = cv2.erode(cv2.dilate(out_map.astype(np.uint8), ker), ker)  # Làm mượt biên vùng
    return out_map, map_info


def save_rhythm_viz(out_dir: str) -> None:
    """Lưu ảnh trực quan hóa liên quan nhịp: dots, beams, ratio_map và overlay tổng hợp."""
    try:
        img = layers.get_layer('original_image')
        if img is None:
            return
        canvas = img.copy()
    except Exception:
        return

    notes = layers.get_layer('notes')
    # dot_img, ratio_map, beam_img là biến toàn cục được gán từ parse_dot/parse_beams
    global dot_img, ratio_map, beam_img
    try:
        if 'dot_img' in globals() and dot_img is not None:
            _save(out_dir, 'step5_dots', dot_img)
    except Exception:
        pass
    try:
        if 'ratio_map' in globals() and ratio_map is not None:
            _save(out_dir, 'step5_ratio_map', ratio_map)
    except Exception:
        pass
    try:
        if 'beam_img' in globals() and beam_img is not None:
            _save(out_dir, 'step5_beams', beam_img)
    except Exception:
        pass

    # Tạo ảnh tổng hợp: tô note theo nhãn trường độ + đánh dấu chấm dôi.
    RHYTHM_COLORS = {
        'WHOLE': (0, 200, 255), 'HALF': (0, 220, 80), 'HALF_OR_WHOLE': (140, 140, 140),
        'QUARTER': (50, 50, 255), 'EIGHTH': (255, 80, 80), 'SIXTEENTH': (200, 40, 200),
        'THIRTY_SECOND': (0, 140, 255), 'SIXTY_FOURTH': (0, 100, 200), 'TRIPLET': (180,255,60), 'OTHERS': (80,80,80)
    }
    if notes is None:
        return
    count_map = {}
    for note in notes:
        if note.bbox is None:
            continue
        lbl = note._label.name if note._label is not None else '?'
        col = RHYTHM_COLORS.get(lbl, C['gray'])
        _rect(canvas, note.bbox, col, thickness=2)
        short = lbl[:3] if lbl != 'HALF_OR_WHOLE' else 'H/W'
        _text(canvas, short, (note.bbox[0] + 1, note.bbox[1] + 10), col, scale=0.32)
        if getattr(note, 'has_dot', False):
            x2 = note.bbox[2]; y1 = note.bbox[1]; y2 = note.bbox[3]
            cv2.circle(canvas, (x2 + 4, int((y1 + y2) / 2)), 3, col, -1)
        count_map[lbl] = count_map.get(lbl, 0) + 1

    # Vẽ thống kê số lượng theo từng nhãn
    x_off = canvas.shape[1] - 160
    y_off = 18
    for lbl, cnt in sorted(count_map.items()):
        col = RHYTHM_COLORS.get(lbl, C['gray'])
        _text(canvas, f"{lbl[:8]}: {cnt}", (x_off, y_off), col, scale=0.35)
        y_off += 15
    legend_items = [(k[:12], v) for k, v in RHYTHM_COLORS.items() if k in count_map]
    _legend(canvas, legend_items, x0=6, y0=18, dy=15)
    _save(out_dir, 'step5_rhythm', canvas)


def refine_map_info(map_info: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Chuẩn hóa map_info bằng cách mở rộng bbox theo group thật và gộp vùng trùng gid.

    Sau bước này, mỗi gid chỉ thuộc về một region đại diện duy nhất,
    giúp parse_rhythm tra cứu ổn định hơn.

    Thuật toán gộp:
    - Dùng rev_map (gid -> region) để phát hiện trùng gid giữa các region.
    - Khi trùng, hợp nhất bbox/gids và cập nhật ánh xạ ngược.
    """
    # Lấy dữ liệu cần thiết
    groups = layers.get_layer('note_groups')
    group_map = layers.get_layer('group_map')

    new_map_info: Dict = {}
    rev_map = {}
    for reg, info in map_info.items():
        cur_gids = info['gids']
        bbox = info['bbox']
        new_map_info[reg] = {'bbox': None, 'gids': set()}

        bbox = np.array([groups[gid].bbox for gid in cur_gids] + [bbox])
        bbox = (np.min(bbox[:, 0]), np.min(bbox[:, 1]), np.max(bbox[:, 2]), np.max(bbox[:, 3]))
        new_map_info[reg]['bbox'] = bbox
        region = group_map[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        gids = set(np.unique(region))
        if -1 in gids:
            gids.remove(-1)

        for gid in gids:
            if gid not in rev_map:
                new_map_info[reg]['gids'].add(gid)
                rev_map[gid] = reg
            else:
                ori_reg = rev_map[gid]
                if ori_reg == reg:
                    continue

                ori_bbox = new_map_info[ori_reg]['bbox']
                ori_gids = new_map_info[ori_reg]['gids']
                new_box = (
                    min(ori_bbox[0], bbox[0]),
                    min(ori_bbox[1], bbox[1]),
                    max(ori_bbox[2], bbox[2]),
                    max(ori_bbox[3], bbox[3])
                )
                ori_gids.add(gid)
                new_map_info[reg]['gids'].update(ori_gids)
                new_map_info[reg]['bbox'] = new_box
                rev_map[gid] = reg

                del new_map_info[ori_reg]
                for ogid in ori_gids:
                    rev_map[ogid] = reg

    return new_map_info


def get_stem_x(gbox: BBox, nboxes: List[ndarray], unit_size: float, is_right: bool = True) -> int:
    """Ước lượng trục x của stem để đặt cửa sổ quét beam/flag.

    Nếu nốt trong group không cùng một mép stem thì dùng tâm group,
    ngược lại chọn mép phải hoặc trái theo hướng stem.
    """
    all_same_side = all(abs(nb[2]-gbox[2])<unit_size/3 for nb in nboxes)
    stem_at_center = not all_same_side
    if stem_at_center:
        return round((gbox[0] + gbox[2]) / 2)
    elif is_right:
        return gbox[2]
    else:
        return gbox[0]


def scan_beam_flag(
    poly_map: ndarray,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    threshold: float = 0.1,
    min_width_ratio: float = 0.25,
    max_width_ratio: float = 0.9
) -> int:
    """
    Đếm số beam/flag bằng cách quét theo cột trong cửa sổ [x,y].

    Cách làm:
    - Mỗi cột đếm số dải pixel liên tục (chuyển trạng thái 0/1).
    - Quy đổi bề dày dải sang số beam bằng max_width.
    - Lấy kết quả đồng thuận theo ngưỡng threshold.

    Ý nghĩa tham số:
    - threshold: tỉ lệ số cột cần đồng thuận để chấp nhận kết quả.
    - min_width_ratio: ngưỡng tối thiểu để xem một dải là beam hợp lệ.
    - max_width_ratio: dùng quy đổi dải dày thành nhiều beam chồng.

    Trả về:
    - Số beam/flag ước lượng trong cửa sổ quét.
    """

    start_x = int(start_x)
    start_y = int(start_y)
    end_x = int(end_x)
    end_y = int(end_y)

    cv2.line(beam_img, (start_x, start_y), (end_x, start_y), (42, 110, 200), 2, cv2.LINE_8)
    cv2.line(beam_img, (start_x, end_y), (end_x, end_y), (42, 110, 200), 2, cv2.LINE_8)

    counter = [0 for _ in range(end_x-start_x)]

    if end_y < start_y:
        start_y, end_y = end_y, start_y

    unit_size = max(1.0, get_unit_size(start_x, start_y))
    min_width = max(1, int(round(unit_size * min_width_ratio)))
    max_width = max(1, int(round(unit_size * max_width_ratio)))
    for idx, x in enumerate(range(start_x, end_x)):
        cur_y = start_y
        last_val = int(poly_map[cur_y, x])

        # Bắt đầu quét theo trục dọc tại cột hiện tại.
        while cur_y < end_y:
            hit = False
            width = 0
            while cur_y < end_y:
                cur_val = int(poly_map[cur_y, x])
                if last_val ^ cur_val:
                    hit = last_val > cur_val
                    last_val = cur_val
                    cur_y += 1
                    break
                cur_y += 1
                width += 1
            if hit and width >= min_width:
                beam_count = math.ceil(width / max_width)
                counter[idx] += beam_count
        if last_val == 1:
            # Chưa gặp điểm chuyển mức nhưng vòng lặp đã kết thúc.
            beam_count = math.ceil(width / max_width) if hit else 1
            counter[idx] += beam_count

    # Thống kê số beam theo từng cột để lấy giá trị đồng thuận.
    stat = {}
    for c in counter:
        if c not in stat:
            stat[c] = 0
        stat[c] += 1
    stat = sorted(stat.items(), key=lambda s: s[0], reverse=True)  # type: ignore

    # Chọn số beam/flag có đủ mức đồng thuận theo tỷ lệ threshold.
    accum = 0
    min_num = len(counter) * threshold
    for c, num in stat:  # type: ignore
        accum += num  # type: ignore
        if accum > min_num:
            return c
    # Fallback an toàn:
    # Nếu đồng thuận yếu nhưng ROI vẫn có mật độ pixel đáng kể,
    # coi là có 1 beam để tránh bỏ sót nốt móc mảnh trong ảnh nhiễu.
    try:
        roi = poly_map[start_y:end_y, start_x:end_x]
        total_pixels = int(roi.sum())
        if total_pixels > max(3, len(counter) * 1):
            return 1
    except Exception:
        pass
    return 0


def parse_inner_groups(poly_map, group, set_box, note_type_map, half_scan_width, threshold=0.1):
    """
    Xử lý group phức hợp (nhiều notehead cùng cụm) khi chưa rõ stem_up toàn cục.

    Hàm chia các nhánh top/bottom theo vị trí note và kiểu notehead,
    sau đó gán nhãn trường độ tương ứng từ số beam/flag quét được.

    Trường hợp xử lý:
    - len(nts)==2: giả định cặp đối xứng stem up/down.
    - all_same_type: top 1 note, còn lại bottom (heuristic phổ biến).
    - mixed type: tìm điểm tách giữa nhóm half và nhóm solid rồi gán riêng.
    """
    # Lấy dữ liệu cần thiết
    notes = layers.get_layer('notes')

    nts = np.copy([notes[nid] for nid in group.note_ids])  # Sao chép để tránh làm thay đổi thứ tự gốc.
    nts = sorted(nts, reverse=True)  # Sắp theo vị trí dòng khuông.

    def get_label(nbox, stem_up):
        cen_x = nbox[2] if stem_up else nbox[0]
        start_y = nbox[1] if stem_up else nbox[3]
        end_y = set_box[1] if stem_up else set_box[3]
        count = scan_beam_flag(
            poly_map,
            start_x=max(set_box[0], cen_x-half_scan_width),
            start_y=start_y,
            end_x=min(set_box[2], cen_x+half_scan_width),
            end_y=end_y,
            threshold=threshold
        )        
        if count >= len(note_type_map):
            return note_type_map[len(note_type_map) - 1]
        return note_type_map[count]

    if len(nts) == 2:
        # Một nốt có stem hướng lên, nốt còn lại hướng xuống.
        notes[nts[0].id].force_set_label(get_label(nts[0].bbox, stem_up=True))
        notes[nts[1].id].force_set_label(get_label(nts[1].bbox, stem_up=False))
        notes[nts[0].id].stem_up = True
        notes[nts[1].id].stem_up = False
        group.top_note_ids.append(nts[0].id)
        group.bottom_note_ids.append(nts[1].id)
    elif group.all_same_type:
        # Tất cả nốt cùng kiểu đặc hoặc rỗng.
        # Giả định có 1 nốt phía trên, các nốt còn lại ở phía dưới.
        notes[nts[0].id].label = get_label(nts[0].bbox, stem_up=True)
        notes[nts[0].id].stem_up = True
        group.top_note_ids.append(nts[0].id)
        bt_label = get_label(nts[-1].bbox, stem_up=False)
        for nn in nts[1:]:
            notes[nn.id].label = bt_label
            notes[nn.id].stem_up = False
            group.bottom_note_ids.append(nn.id)
    else:
        # Trường hợp phức tạp: trộn cả nốt đặc và nốt rỗng.
        # Cần tìm điểm tách giữa hai nhóm.
        idx = 0
        while idx < len(nts):
            if nts[0].label != nts[idx].label:
                break
            idx += 1

        if nts[0].label == NoteType.HALF_OR_WHOLE:
            # Nhóm phía trên là nốt trắng (half).
            for nn in nts[:idx]:
                # assert nn.label == NoteType.HALF_OR_WHOLE
                notes[nn.id].force_set_label(NoteType.HALF)
                notes[nn.id].stem_up = True
                group.top_note_ids.append(nn.id)
            bt_label = get_label(nts[-1].bbox, stem_up=False)
            for nn in nts[idx:]:
                notes[nn.id].label = bt_label
                notes[nn.id].stem_up = False
                group.bottom_note_ids.append(nn.id)
        else:
            # Nhóm phía dưới là nốt trắng (half).
            for nn in nts[idx:]:
                # assert nn.label == NoteType.HALF_OR_WHOLE, nn
                notes[nn.id].force_set_label(NoteType.HALF)
                notes[nn.id].stem_up = False
                group.bottom_note_ids.append(nn.id)
            top_label = get_label(nts[-1].bbox, stem_up=True)
            for nn in nts[:idx]:
                notes[nn.id].label = top_label
                notes[nn.id].stem_up = True
                group.top_note_ids.append(nn.id)


def parse_rhythm(beam_map: ndarray, map_info: Dict[int, Dict[str, Any]], agree_th: float = 0.15) -> ndarray:
    """
    Gán nhãn trường độ cho nốt dựa trên beam/flag và thông tin group.

    Luồng chính:
    - Xây ánh xạ gid -> vùng beam liên quan.
    - Với từng group: xử lý trường hợp không stem, stem chưa rõ, hoặc stem đã rõ.
    - Quét beam/flag để suy ra NoteType theo bảng note_type_map.

        Nhánh quyết định cho mỗi group:
        - stem_up is None và không có stem: ưu tiên WHOLE/HO theo nhánh dự phòng.
        - stem_up is None nhưng có stem: giao cho parse_inner_groups.
        - stem_up đã rõ: quét quanh trục stem rồi suy ra label từ beam_flag_count.

        Ghi chú độ tin cậy:
        - Các nhánh dự phòng về QUARTER được dùng để giữ pipeline không gãy khi
            beam detection kém ổn định trên trang nhiễu.
    """
    # Lấy dữ liệu cần thiết
    groups = layers.get_layer('note_groups')
    notes = layers.get_layer('notes') 
    notehead = layers.get_layer('notehead_pred')

    # Thu thập thông tin cần thiết
    rev_map_info = {}
    for reg, info in map_info.items():
        gids = info['gids']
        box = info['bbox']
        for gid in gids:
            rev_map_info[gid] = {'reg': reg, 'bbox': box}

    # Bảng ánh xạ số beam/flag sang loại nốt.
    # Lưu ý: count=4 hiện ánh xạ về SIXTEENTH theo heuristic hiện tại của pipeline.
    note_type_map: Dict[int, NoteType] = {
        0: NoteType.QUARTER,
        1: NoteType.EIGHTH,
        2: NoteType.SIXTEENTH,
        3: NoteType.THIRTY_SECOND,
        4: NoteType.SIXTEENTH,
        #5: None,
        #6: None
    }

    global beam_img
    beam_img = to_rgb_img(np.where(beam_map+notehead>0, 1, 0))
    # bboxes = [v['bbox'] for v in map_info.values()]
    # beam_img = draw_bounding_boxes(bboxes, beam_img)

    # Bắt đầu gán nhãn nhịp cho từng group.
    bin_beam_map = np.where(beam_map>0, 1, 0)
    for gid in range(len(groups)):
        group = groups[gid]
        gbox = group.bbox
        reg_box = rev_map_info[gid]['bbox'] if gid in rev_map_info else gbox
        unit_size = get_unit_size(*get_center(gbox))
        half_scan_width = round(unit_size / 2)

        # Nhánh xử lý theo trạng thái stem của group.
        if group.stem_up is None:
            if not group.has_stem:
                # Trường hợp này chỉ có thể là nốt tròn (whole).
                for nid in group.note_ids:
                    if notes[nid].label != NoteType.HALF_OR_WHOLE:
                        # Dùng fallback an toàn thay vì loại bỏ nốt.
                        # Một số bản nhạc phát hiện stem chưa hoàn hảo nhưng vẫn
                        # biểu diễn nốt đen (quarter) hợp lệ tại vị trí này.
                        notes[nid].force_set_label(NoteType.QUARTER)
                        continue
                    notes[nid].force_set_label(NoteType.WHOLE)
            else:
                parse_inner_groups(
                    poly_map=bin_beam_map,
                    group=group,
                    set_box=reg_box,
                    note_type_map=note_type_map,
                    half_scan_width=half_scan_width,
                    threshold=agree_th
                )
            continue

        # Đồng thuận nhãn hiện có trong group để chọn hướng xử lý tiếp.
        labels = [notes[nid].label for nid in group.note_ids]
        count = {k: 0 for k in set(labels)}
        for l in labels:
            count[l] += 1
        count = sorted(count.items(), key=lambda c: c[1], reverse=True)  # type: ignore
        label = count[0][0]  # type: ignore
        if label == NoteType.HALF_OR_WHOLE:
            # Nhóm này chỉ chứa nốt trắng (half).
            for nid in group.note_ids:
                notes[nid].force_set_label(NoteType.HALF)
            continue

        if gid not in rev_map_info:
            # Không có beam/flag gắn với nhóm này,
            # nên chỉ có thể là nốt đen (quarter).
            for nid in group.note_ids:
                # assert notes[nid].label is None, notes[nid]
                notes[nid].label = NoteType.QUARTER
            continue

        gbox = group.bbox
        gbox = (gbox[0], min(gbox[1], reg_box[1]), gbox[2], max(gbox[3], reg_box[3]))  # Chỉ hiệu chỉnh theo trục y
        nbox = [notes[nid].bbox for nid in group.note_ids]
        unit_size = get_unit_size(*get_center(gbox))
        if group.stem_up:
            cen_x = get_stem_x(gbox, nbox, unit_size)
            start_y = min(nb[1] for nb in nbox)
            end_y = gbox[1]
        else:
            cen_x = get_stem_x(gbox, nbox, unit_size, is_right=False)
            start_y = max(nb[3] for nb in nbox)
            end_y = gbox[3]

        # Quét cửa sổ quanh stem để ước lượng số beam/flag.
        beam_flag_count = scan_beam_flag(  # type: ignore
            bin_beam_map,
            max(reg_box[0], cen_x-half_scan_width),
            start_y,
            min(reg_box[2], cen_x+half_scan_width),
            end_y,
            threshold=agree_th
        )

        # cv2.rectangle(beam_img, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (255, 0, 255), 1)
        cv2.putText(beam_img, str(beam_flag_count), (int(cen_x), int(gbox[3])+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Gán nhãn cuối cùng cho các note chưa được gán.
        for nid in group.note_ids:
            if notes[nid].label is None:
                if beam_flag_count in note_type_map:
                    notes[nid].label = note_type_map[beam_flag_count]
                else:
                    # Nhánh dự phòng về nốt đen (quarter) thay vì đánh invalid.
                    # Cách này giúp pipeline vẫn dùng được khi đếm beam/flag
                    # bỏ sót ký hiệu trên trang nhiễu.
                    notes[nid].force_set_label(NoteType.QUARTER)

    return beam_img


def extract(
    dot_min_area_ratio: float = 0.08,
    dot_max_area_ratio: float = 0.2,
    beam_min_area_ratio: float = 0.07,
    agree_th: float = 0.15
) -> Tuple[ndarray, List[RotatedRect]]:
    """
    Pipeline tổng của rhythm extraction.

    Trình tự:
    1) parse_dot: phát hiện chấm dôi.
    2) parse_beams: tách vùng beam hợp lệ.
    3) parse_beam_overlap_regions + refine_map_info: liên kết beam với groups.
    4) parse_rhythm: gán nhãn trường độ cho nốt.

    Đầu ra:
    - beam_img: ảnh debug có vùng quét và số beam/flag đã đếm.
    - valid_box: danh sách rotated beam-box hợp lệ sau lọc.
    """
    parse_dot(max_area_ratio=dot_max_area_ratio, min_area_ratio=dot_min_area_ratio)

    poly_map, valid_box, invalid_map = parse_beams(min_area_ratio=beam_min_area_ratio)

    out_map, map_info = parse_beam_overlap_regions(poly_map, invalid_map)

    map_info = refine_map_info(map_info)

    beam_img = parse_rhythm(out_map, map_info, agree_th=agree_th)
    return beam_img, valid_box



