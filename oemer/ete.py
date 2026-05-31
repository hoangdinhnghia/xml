import os
import pickle
import argparse
from pathlib import Path
from typing import Tuple
from argparse import Namespace, ArgumentParser

from PIL import Image
from numpy import ndarray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from oemer import MODULE_PATH
from oemer import layers
from oemer.inference import inference
from oemer.utils import get_logger
from oemer.dewarp import estimate_coords, dewarp
from oemer.staffline_extraction import extract as staff_extract
import oemer.staffline_extraction as staffline_extraction
from oemer.notehead_extraction import extract as note_extract
from oemer.note_group_extraction import extract as group_extract
from oemer.symbol_extraction import extract as symbol_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.build_system import MusicXMLBuilder
from oemer.draw_teaser import teaser


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Màu BGR và font
# ---------------------------------------------------------------------------
C = {
    "staff_line":  (255, 180,  60),   # xanh da trời nhạt
    "zone":        (0,   220, 220),   # vàng nhạt
    "note_whole":  (80,  200, 255),   # vàng
    "note_half":   (0,   200, 80),    # xanh lá
    "note_quarter":(50,   50, 255),   # đỏ
    "note_other":  (200,  80, 200),   # tím
    "stem_up":     (0,   255, 180),   # xanh ngọc
    "stem_down":   (0,   120, 255),   # cam
    "barline":     (50,   50, 255),   # đỏ
    "clef":        (255, 220,   0),   # cyan
    "sfn":         (255,   0, 200),   # magenta
    "rest":        (0,   165, 255),   # cam nhạt
    "beam":        (100, 255, 100),   # xanh lá nhạt
    "dot":         (0,   200, 255),   # vàng nhạt
    "white":       (255, 255, 255),
    "black":       (0,     0,   0),
    "gray":        (160,  160, 160),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _base(alpha: float = 1.0) -> np.ndarray:
    """Tạo canvas nền để vẽ debug.

    Trả về một bản sao ảnh gốc đang lưu trong layer `original_image`
    (thường là ảnh đã được dewarp nếu tùy chọn đó được bật).
    Việc trả về bản sao giúp mọi bước vẽ overlay không làm hỏng dữ liệu gốc.
    """
    img = layers.get_layer('original_image')
    return img.copy()


def _overlay(canvas: np.ndarray, mask: np.ndarray,
             color: tuple, alpha: float = 0.40) -> np.ndarray:
    """Phủ một mask nhị phân lên canvas bằng kỹ thuật alpha blending.

    Thuật toán:
    - Tạo một ảnh màu rỗng cùng kích thước canvas.
    - Tô các pixel có mask > 0 bằng màu chỉ định.
    - Trộn tuyến tính theo công thức: out = canvas*(1-alpha) + color*alpha.
    """
    colored = np.zeros_like(canvas, dtype=np.uint8)
    colored[mask > 0] = color
    m = mask > 0
    canvas[m] = (canvas[m] * (1 - alpha) + colored[m] * alpha).astype(np.uint8)
    return canvas


def _rect(canvas, bbox, color, thickness=1):
    """Vẽ khung chữ nhật theo định dạng bbox = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)


def _text(canvas, txt, pos, color, scale=0.45, thickness=1, font=FONT):
    """Vẽ chữ có viền đen mỏng để tăng khả năng đọc trên nền nhiễu."""
    cv2.putText(canvas, txt, pos, font, scale, C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, txt, pos, font, scale, color,    thickness,     cv2.LINE_AA)


def _legend(canvas, items: list, x0: int = 6, y0: int = 20, dy: int = 18):
    """Vẽ legend hình chữ nhật màu + nhãn. items = [(label, color), ...]"""
    for i, (label, color) in enumerate(items):
        y = y0 + i * dy
        cv2.rectangle(canvas, (x0, y - 10), (x0 + 14, y + 2), color, -1)
        _text(canvas, label, (x0 + 18, y), color, scale=0.40)


def _save(out_dir: str, name: str, img: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(path, img)
    # Ghi ảnh bước và log đường dẫn rõ ràng để dễ tra cứu
    logger.info("[step img] Saved: %s", path)


def save_step_dewarp(out_dir: str, before_img: np.ndarray, after_img: np.ndarray,
                     coords_x: np.ndarray, coords_y: np.ndarray) -> None:
    """Lưu ảnh trước/sau dewarp và bản đồ ánh xạ (vector field) để debug.

    - `step0_before_dewarp`: ảnh trước khi dewarp
    - `step0_after_dewarp`: ảnh sau khi dewarp
    - `step0_dewarp_map`: ảnh trước dewarp với mũi tên chỉ hướng biến đổi điểm mẫu
    """
    try:
        # Ủy quyền cho module dewarp để tái sử dụng đúng chuẩn visualization.
        from oemer.dewarp import save_dewarp_viz
        save_dewarp_viz(out_dir, before_img, after_img, coords_x, coords_y)
    except Exception:
        logger.exception('Failed to save dewarp debug images')


# ===========================================================================
# BƯỚC 1 — Staff lines & Zones
# ===========================================================================
def save_step_stafflines(out_dir: str) -> None:
    """Lưu ảnh trực quan hóa stafflines bằng hàm chuyên trách.

    Thiết kế tách hàm vẽ sang `staffline_extraction` giúp:
    - `ete.py` chỉ tập trung điều phối pipeline.
    - Logic vẽ nằm gần logic trích xuất để dễ bảo trì.
    """
    try:
        staffline_extraction.save_stafflines_viz(out_dir)
    except Exception:
        logger.exception('Failed to save staffline visualization')


# ===========================================================================
# BƯỚC 2 — Notehead extraction
# ===========================================================================
def save_step_noteheads(out_dir: str) -> None:
    """Lưu ảnh kiểm tra Noteheads.

    Ảnh hiển thị mask notehead và bbox từng note với màu theo loại (WHOLE/HALF/QUARTER/other).
    Trên mỗi bbox có: id, nhãn ngắn, hướng cọng (▲/▼/?), vị trí trên staff và chỉ số độ rắn (solidity).
    Dùng để xác minh phát hiện notehead và gán nhãn trường độ sau này.
    """
    from oemer.notehead_extraction import NoteType

    try:
        from oemer.notehead_extraction import save_noteheads_viz
        save_noteheads_viz(out_dir)
    except Exception:
        logger.exception('Failed to save noteheads viz')


# ===========================================================================
# BƯỚC 3 — Note group extraction
# ===========================================================================
def save_step_note_groups(out_dir: str) -> None:
    """Lưu ảnh nhóm note (NoteGroups).

        - Mỗi nhóm có một màu; vẽ bbox nhóm và bbox từng note cùng màu.
        - Thêm nhãn tóm tắt: group id, track, stem/has_stem, số note.
        - Vẽ đường nối từ tâm nhóm tới tâm từng note để kiểm tra kết nối.
        Mục đích: xác minh kết quả phân nhóm note và gán nhãn track sau này.
    """
    GROUP_COLORS = [
        (255, 80,  80),  (80, 255,  80),  (80, 100, 255),
        (255, 220,  0),  (0,  220, 220),  (220,   0, 220),
        (255, 160,  40), (40, 200, 180),  (180, 100, 255),
        (80, 255, 180),  (255, 100, 180), (180, 255,  60),
    ]

    try:
        from oemer.note_group_extraction import save_note_groups_viz
        save_note_groups_viz(out_dir)
    except Exception:
        logger.exception('Failed to save note groups viz')


# ===========================================================================
# BƯỚC 4 — Symbol extraction (barlines, clefs, sfns, rests)
# ===========================================================================
def save_step_symbols(out_dir: str) -> None:
    """Lưu ảnh các kí hiệu (symbols): barlines, clefs, accidentals (SFN), rests.

        Mỗi loại được vẽ riêng trên cùng canvas với màu khác nhau và nhãn ngắn giúp
        nhận diện nhanh: e.g. "BAR g{group}", clef type, SFN ký hiệu và rest type.
        Có thể tách thành ảnh riêng cho từng loại nếu cần debug sâu hơn.
        """
    try:
        from oemer.symbol_extraction import save_symbols_viz
        save_symbols_viz(out_dir)
        try:
            # Giữ tương thích ngược: vẫn xuất thêm các layer phụ như trước.
            save_step_extra_layers(out_dir)
        except Exception:
            logger.debug('save_step_extra_layers encountered an error')
    except Exception:
        logger.exception('Failed to save symbols viz')


# ===========================================================================
# BƯỚC 5 — Rhythm extraction
# ===========================================================================
def save_step_rhythm(out_dir: str) -> None:
    """Lưu ảnh kiểm tra phân loại trường độ (rhythm) cho từng note.

    - Màu bbox thể hiện nhãn rhythm (WHOLE/HALF/QUARTER/EIGHTH/...)
    - Chấm hiện diện nếu note có dot
    - Bảng đếm số note theo loại giúp so sánh thống kê nhanh
    """
    from oemer.notehead_extraction import NoteType

    try:
        from oemer.rhythm_extraction import save_rhythm_viz
        save_rhythm_viz(out_dir)
    except Exception:
        logger.exception('Failed to save rhythm viz')


# ===========================================================================
# BƯỚC 6 — Final overlay (tổng hợp tất cả lớp)
# ===========================================================================
def save_step_final_overlay(out_dir: str) -> None:
    """Lưu ảnh overlay tổng hợp (tổng quan kết quả pipeline).

        Ảnh này tổng hợp staff, symbol, notehead và các bbox để nhanh kiểm tra
        kết quả cuối cùng và các số liệu tóm tắt ở góc ảnh.
        """
    canvas = _base()

    staff_mask = layers.get_layer('staff_pred')
    note_mask  = layers.get_layer('notehead_pred')
    sym_mask   = layers.get_layer('symbols_pred')

    canvas = _overlay(canvas, staff_mask, C["staff_line"], alpha=0.28)
    extra  = np.clip(sym_mask.astype(int) - note_mask.astype(int), 0, 1).astype(np.uint8)
    canvas = _overlay(canvas, extra,      C["zone"],       alpha=0.28)
    canvas = _overlay(canvas, note_mask,  (80, 200, 80),   alpha=0.38)

    barlines = layers.get_layer('barlines')
    clefs    = layers.get_layer('clefs')
    sfns     = layers.get_layer('sfns')
    rests    = layers.get_layer('rests')
    notes    = layers.get_layer('notes')
    groups   = layers.get_layer('note_groups')

    for b in barlines:
        if b.bbox is not None:
            _rect(canvas, b.bbox, C["barline"], 1)
    for cl in clefs:
        if cl.bbox is not None:
            _rect(canvas, cl.bbox, C["clef"], 2)
    for sf in sfns:
        if sf.bbox is not None:
            _rect(canvas, sf.bbox, C["sfn"], 1)
    for rs in rests:
        if rs.bbox is not None:
            _rect(canvas, rs.bbox, C["rest"], 2)
    for note in notes:
        if note.bbox is not None:
            _rect(canvas, note.bbox, (80, 200, 80), 1)

    # Thống kê tổng hợp để kiểm tra nhanh chất lượng đầu ra toàn pipeline.
    staffs = layers.get_layer('staffs')
    n_staffs = staffs.size if hasattr(staffs, 'size') else len(staffs)
    summary = (
        f"Staffs={n_staffs}  Notes={len(notes)}  Groups={len(groups)}  "
        f"Bars={len(barlines)}  Clefs={len(clefs)}  SFN={len(sfns)}  Rests={len(rests)}"
    )
    _text(canvas, summary, (6, canvas.shape[0] - 8), C["white"], scale=0.42)

    legend_items = [
        ("Staff lines",  C["staff_line"]),
        ("Stems/Other",  C["zone"]),
        ("Noteheads",    (80, 200, 80)),
        ("Barlines",     C["barline"]),
        ("Clefs",        C["clef"]),
        ("SFN",          C["sfn"]),
        ("Rests",        C["rest"]),
    ]
    _legend(canvas, legend_items, x0=6, y0=18, dy=15)
    _save(out_dir, "step6_final_overlay", canvas)


# ===========================================================================
# Core pipeline (không thay đổi so với bản gốc)
# ===========================================================================

def clear_data() -> None:
    """Xóa toàn bộ layer tạm trong bộ nhớ dùng chung của pipeline.

    Hàm này được gọi trước mỗi lần chạy để tránh rò rỉ trạng thái giữa
    các lần suy luận khác nhau.
    """
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)


def generate_pred(
    img_path: str,
    use_tf: bool = False,
    print_artifacts: bool = False,
    generate_artifacts: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Sinh các bản đồ phân đoạn đầu vào cho pipeline OMR.

    Đầu ra gồm 5 lớp nhị phân:
    - `staff`: đường kẻ khuông.
    - `symbols`: lớp ký hiệu tổng quát (trừ phần tách riêng).
    - `stems_rests`: thân nốt + rest.
    - `notehead`: đầu nốt.
    - `clefs_keys`: khóa nhạc và hóa biểu.

    Quy trình chạy 2 model:
    1) `unet_big` để tách staff/symbols thô.
    2) `seg_net` để tách chi tiết theo loại ký hiệu.
    """
    logger.info("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/unet_big"),
        img_path,
        use_tf=use_tf,
        print_artifacts=print_artifacts,
        generate_artifacts=generate_artifacts,
    )
    staff   = np.where(staff_symbols_map == 1, 1, 0)
    symbols = np.where(staff_symbols_map == 2, 1, 0)

    logger.info("Extracting layers of different symbols")
    sep, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/seg_net"),
        img_path,
        manual_th=None,
        use_tf=use_tf,
        print_artifacts=print_artifacts,
        generate_artifacts=generate_artifacts,
    )
    stems_rests = np.where(sep == 1, 1, 0)
    notehead    = np.where(sep == 2, 1, 0)
    clefs_keys  = np.where(sep == 3, 1, 0)

    return staff, symbols, stems_rests, notehead, clefs_keys


def polish_symbols(rgb_black_th=300):
    """Làm sạch lớp `symbols_pred` bằng thông tin điểm ảnh tối từ ảnh gốc.

    Ý tưởng: các nét nhạc thường tối, nên lấy ngưỡng tổng RGB để lọc nền sáng,
    sau đó đóng/mở hình thái học nhẹ và hợp nhất lại với mask symbols hiện có.
    """
    img     = layers.get_layer('original_image')
    sym_pred = layers.get_layer('symbols_pred')
    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0]))
    arr = np.sum(np.array(img), axis=-1)
    arr = np.where(arr < rgb_black_th, 1, 0)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker)
    mix = np.where(sym_pred + arr > 1, 1, 0)
    return mix


def register_notehead_bbox(bboxes):
    """Ghi bbox notehead vào layer `bboxes` theo từng pixel thuộc ký hiệu.

    Mục tiêu: tạo bản đồ tra cứu nhanh bbox từ tọa độ pixel để các bước sau
    có thể truy ngược đối tượng mà không cần quét lại danh sách bbox.
    """
    symbols = layers.get_layer('symbols_pred')
    layer   = layers.get_layer('bboxes')
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2] > 0)
        yi += y1
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2])
    return layer


def register_note_id() -> None:
    """Gán id note vào layer `note_id` cho mọi pixel thuộc bbox note."""
    symbols = layers.get_layer('symbols_pred')
    layer   = layers.get_layer('note_id')
    notes   = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2] > 0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def save_step_extra_layers(out_dir: str) -> None:
    """Lưu các lớp phụ cũ chưa được hiển thị: stems_rests, clefs_keys, polished symbols.

    Mục đích: tích hợp nhanh các layer phụ (đã được tính nhưng chưa xuất) vào
    thư mục step để dễ so sánh với các bước chính.
    """
    canvas = _base()

    # Layer stems_rests
    try:
        sr = layers.get_layer('stems_rests_pred')
        if sr is not None:
            _overlay(canvas, sr, C['beam'], alpha=0.35)
            _save(out_dir, 'step4_stems_rests', canvas.copy())
    except Exception:
        logger.debug('No stems_rests_pred layer')

    # Layer clefs_keys
    try:
        ck = layers.get_layer('clefs_keys_pred')
        if ck is not None:
            canvas2 = _base()
            _overlay(canvas2, ck, C['clef'], alpha=0.40)
            _save(out_dir, 'step4_clefs_keys', canvas2)
    except Exception:
        logger.debug('No clefs_keys_pred layer')

    # Layer symbols đã làm sạch (hỗ trợ debug kế thừa)
    try:
        mix = polish_symbols()
        if mix is not None:
            canvas3 = _base()
            _overlay(canvas3, mix, C['zone'], alpha=0.40)
            _save(out_dir, 'step4_polished_symbols', canvas3)
    except Exception:
        logger.debug('polish_symbols failed or not available')


def extract(args: Namespace) -> str:
    """Chạy toàn bộ pipeline OMR end-to-end cho một ảnh đầu vào.

    Luồng xử lý chính:
    1) Nạp cache hoặc chạy suy luận segmentation.
    2) Chuẩn hóa ảnh và dewarp (nếu bật).
    3) Trích stafflines → noteheads → note groups → symbols → rhythm.
    4) Dựng MusicXML và ghi file kết quả.

    Trả về đường dẫn file `.musicxml` đã sinh.
    """
    img_path = Path(args.img_path)
    f_name   = os.path.splitext(img_path.name)[0]
    pkl_path = img_path.parent / f"{f_name}.pkl"

    if pkl_path.exists():
        pred        = pickle.load(open(pkl_path, "rb"))
        notehead    = pred["note"]
        symbols     = pred["symbols"]
        staff       = pred["staff"]
        clefs_keys  = pred["clefs_keys"]
        stems_rests = pred["stems_rests"]
    else:
        if args.use_tf:
            ori_inf_type = os.environ.get("INFERENCE_WITH_TF", None)
            os.environ["INFERENCE_WITH_TF"] = "true"
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(
            str(img_path),
            use_tf=args.use_tf,
            print_artifacts=getattr(args, 'print_artifacts', False),
            generate_artifacts=getattr(args, 'generate_artifacts', False),
        )
        if args.use_tf and ori_inf_type is not None:
            os.environ["INFERENCE_WITH_TF"] = ori_inf_type
        if args.save_cache:
            data = {
                'staff':       staff,
                'note':        notehead,
                'symbols':     symbols,
                'stems_rests': stems_rests,
                'clefs_keys':  clefs_keys,
            }
            pickle.dump(data, open(pkl_path, "wb"))

    image_pil = Image.open(str(img_path))
    if "GIF" != image_pil.format:
        image = cv2.imread(str(img_path))
    else:
        gif_image   = image_pil.convert('RGB')
        gif_img_arr = np.array(gif_image)
        image       = gif_img_arr[:, :, ::-1].copy()

    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))

    if not args.without_deskew:
        logger.info("Dewarping")
        coords_x, coords_y = estimate_coords(staff)
        # Lưu bản sao trước dewarp để xuất ảnh so sánh trước/sau.
        orig_image = image.copy()
        staff       = dewarp(staff,       coords_x, coords_y)
        symbols     = dewarp(symbols,     coords_x, coords_y)
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys  = dewarp(clefs_keys,  coords_x, coords_y)
        notehead    = dewarp(notehead,    coords_x, coords_y)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], coords_x, coords_y)

    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols > 1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred",  clefs_keys)
    layers.register_layer("notehead_pred",    notehead)
    layers.register_layer("symbols_pred",     symbols)
    layers.register_layer("staff_pred",       staff)
    layers.register_layer("original_image",   image)

    # Thư mục lưu ảnh trực quan hóa theo từng bước pipeline.
    step_dir = None
    if getattr(args, 'save_intermediate', False):
        step_dir = os.path.join(args.output_path, f"{f_name}_steps")

    # Nếu có dewarp, lưu ảnh trước/sau và bản đồ biến đổi để kiểm tra.
    if step_dir:
        try:
            if 'orig_image' in locals():
                save_step_dewarp(step_dir, orig_image, image, coords_x, coords_y)
        except Exception:
            logger.debug('save_step_dewarp encountered an error')

    # ---- Bước 1: Trích xuất stafflines ----
    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)
    layers.register_layer("zones",  zones)
    if step_dir:
        save_step_stafflines(step_dir)

    # ---- Bước 2: Trích xuất noteheads ----
    logger.info("Extracting noteheads")
    notes = note_extract()
    layers.register_layer('notes',   np.array(notes))
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64) - 1)
    register_note_id()
    if step_dir:
        save_step_noteheads(step_dir)

    # ---- Bước 3: Gom nhóm note ----
    logger.info("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map',   group_map)
    if step_dir:
        save_step_note_groups(step_dir)

    # ---- Bước 4: Trích xuất symbols ----
    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs',    np.array(clefs))
    layers.register_layer('sfns',     np.array(sfns))
    layers.register_layer('rests',    np.array(rests))
    if step_dir:
        save_step_symbols(step_dir)
        # Lưu thêm các lớp phụ để debug sâu khi cần.
        try:
            save_step_extra_layers(step_dir)
        except Exception:
            logger.debug('save_step_extra_layers encountered an error')

    # ---- Bước 5: Suy luận trường độ (rhythm) ----
    logger.info("Extracting rhythm types")
    rhythm_extract()
    if step_dir:
        save_step_rhythm(step_dir)
        save_step_final_overlay(step_dir)

    # ---- Bước 6: Dựng tài liệu MusicXML ----
    logger.info("Building MusicXML document")
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder  = MusicXMLBuilder(title=basename.capitalize())
    builder.build()
    xml = builder.to_musicxml()

    out_path = args.output_path
    if not out_path.endswith(".musicxml"):
        out_path = os.path.join(out_path, basename + ".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)

    return out_path


def get_parser() -> ArgumentParser:
    """Khởi tạo parser CLI."""
    parser = argparse.ArgumentParser(
        "Oemer",
        description="Công cụ OMR chạy end-to-end từ ảnh sang MusicXML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("img_path", type=str)
    parser.add_argument("-o", "--output-path", type=str, default="./")
    parser.add_argument("--use-tf",           action="store_true")
    parser.add_argument("--save-cache",       action='store_true')
    parser.add_argument("-d", "--without-deskew", action='store_true')
    parser.add_argument(
        "--save-intermediate",
        help="Lưu 6 ảnh test chi tiết từng bước vào <output>/<tên>_steps/",
        action='store_true')
    parser.add_argument("--print-artifacts",    action='store_true')
    parser.add_argument("--generate-artifacts", action='store_true')
    parser.add_argument("--use-new-extractors", action='store_true')
    return parser


def main() -> None:
    """Điểm vào CLI: kiểm tra đầu vào, chạy pipeline, xuất teaser image."""
    parser = get_parser()
    args   = parser.parse_args()

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    clear_data()
    mxl_path = extract(args)
    img = teaser()
    img.save(mxl_path.replace(".musicxml", "_teaser.png"))


if __name__ == "__main__":
    main()