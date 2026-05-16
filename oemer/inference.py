"""Module nhận diện ký hiệu nhạc từ ảnh bằng cách chia nhỏ ảnh và xử lý từng phần.

Quy trình: Chia ảnh thành từng khúc nhỏ -> Sử dụng mô hình AI dự đoán mỗi khúc -> Ghép lại kết quả.
"""

import os
import sys
import pickle
import warnings
import hashlib
import json
import time
from sklearn.exceptions import InconsistentVersionWarning
from PIL import Image
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray

from oemer import MODULE_PATH
from oemer.logger import get_logger
from oemer.postproc import (
    gaussian_weighted_merge,
    save_cache,
    load_cache,
    generate_report,
    entropy_heatmap,
    confidence_filter,
    generate_gui,
)
from oemer.postproc import low_conf_bboxes, create_overlay
from oemer import note_group_extraction as nge
from oemer.utils import get_unit_size
from oemer import config as oemer_config

logger = get_logger(__name__)


def resize_image(image: Image.Image):
    """Thay đổi kích thước ảnh sao cho phù hợp với mô hình (3-4.35 triệu điểm ảnh).
    Nếu xóa hay nén ảnh quá mạnh sẽ mất các chi tiết nhỏ của ký hiệu nhạc.
    """
    w, h = image.size
    pis = w * h
    if 3000000 <= pis <= 4350000:
        return image
    lb = 3000000 / pis
    ub = 4350000 / pis
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    return image.resize((tar_w, tar_h))


def load_model(model_path: str, use_tf: bool = False):
    """Load model from disk and return (model_obj, metadata).

    For TF: returns (tf_model, {'input_shape': ..., 'output_shape': ...}, 'tf')
    For ONNX: returns (onnx_session, metadata, 'onnx')
    """
    if use_tf:
        import tensorflow as tf

        arch_path = os.path.join(model_path, "arch.json")
        w_path = os.path.join(model_path, "weights.h5")
        if not os.path.exists(arch_path) or not os.path.exists(w_path):
            raise FileNotFoundError(f"TensorFlow model files missing in {model_path}")
        with open(arch_path, "r") as fh:
            model = tf.keras.models.model_from_json(fh.read())
        model.load_weights(w_path)
        metadata = {"input_shape": model.input_shape, "output_shape": model.output_shape}
        return model, metadata, "tf"

    # ONNX path
    import onnxruntime as rt

    onnx_path = os.path.join(model_path, "model.onnx")
    meta_path = os.path.join(model_path, "metadata.pkl")
    if not os.path.exists(onnx_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"ONNX model files missing in {model_path}")
    with open(meta_path, "rb") as fh:
        metadata = pickle.load(fh)

    # Select providers conservatively; let the runtime pick when not available
    if sys.platform == "darwin":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    sess = rt.InferenceSession(onnx_path, providers=providers)
    return sess, metadata, "onnx"


def tile_image(image: ndarray, win_size: int, step_size: int = 128):
    """Tile `image` into patches of `win_size` with stride `step_size`.
    Returns list of patches and list of (x,y) coordinates (top-left) for each patch.
    """
    patches = []
    coords = []
    h, w = image.shape[:2]
    for y in range(0, h, step_size):
        yy = y if y + win_size <= h else h - win_size
        for x in range(0, w, step_size):
            xx = x if x + win_size <= w else w - win_size
            patches.append(image[yy : yy + win_size, xx : xx + win_size])
            coords.append((xx, yy))
    return patches, coords


def predict_patches(model_obj, metadata: dict, model_type: str, patches: list, batch_size: int = 16):
    """Run predictions on list of patches and return a list of outputs in same order.
    Output per patch is assumed to be an ndarray of shape (win, win, channels).
    """
    outputs = []
    n = len(patches)
    for i in range(0, n, batch_size):
        batch = np.array(patches[i : i + batch_size])
        if model_type == "tf":
            out = model_obj.predict(batch)
        else:
            # ONNX expects a dict input name -> array; metadata contains output_names
            out = model_obj.run(metadata["output_names"], {"input": batch})[0]
        # ensure iterable of outputs
        for o in out:
            outputs.append(o)
    return outputs


def merge_patches(patch_outputs: list, coords: list, image_shape: tuple, win_size: int, method: str = "count"):
    """Merge predicted patches back to full image shape.

    patch_outputs: list of arrays shape (win_size, win_size, channels)
    coords: list of (x, y) top-left coordinates in same order as patch_outputs
    image_shape: (h, w)
    method: 'count' (original average) or 'gaussian' (weighted blending)
    """
    if method == "gaussian":
        return gaussian_weighted_merge(patch_outputs, coords, image_shape, win_size)

    # fallback: original count-based merge
    h, w = image_shape[:2]
    channels = patch_outputs[0].shape[-1]
    out = np.zeros((h, w, channels), dtype=np.float32)
    mask = np.zeros((h, w, channels), dtype=np.float32)
    for patch, (x, y) in zip(patch_outputs, coords):
        out[y : y + win_size, x : x + win_size] += patch
        mask[y : y + win_size, x : x + win_size] += 1
    # Avoid division by zero
    mask[mask == 0] = 1
    out = out / mask
    return out


def inference(
    model_path: str,  # thư mục chứa mô hình AI đã huấn luyện
    img_path: str,  # đường dẫn file ảnh nhạc cần phân tích
    step_size: int = 128,  # khoảng cách giữa các khúc ảnh (128 điểm ảnh)
    batch_size: int = 16,  # xử lý bao nhiêu khúc ảnh cùng 1 lúc (nhiều hơn = nhanh hơn nhưng tiêu tốn RAM)
    manual_th: Optional[Any] = None,  # giá trị ngưỡng tùy chỉnh (mặc định = chọn loại có xác suất cao nhất)
    use_tf: bool = False,  # dùng TensorFlow (True) hay ONNX (False, tối ưu hơn)
    use_cache: bool = True,
    merge_method: str = "gaussian",
) -> Tuple[ndarray, ndarray]:
    # Load model
    t0 = time.perf_counter()
    model_obj, metadata, model_type = load_model(model_path, use_tf=use_tf)
    t_model = time.perf_counter() - t0

    # Read and prepare image
    image_pil = Image.open(img_path)
    if "GIF" != image_pil.format:
        image_cv = cv2.imread(img_path)
        image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert("RGB")
    image = np.array(resize_image(image_pil))

    # Determine window size from metadata
    win_size = metadata["input_shape"][1]

    # Prepare cache key and try load
    cache_key = hashlib.sha256(f"{model_path}|{img_path}|{step_size}|{batch_size}|{merge_method}".encode("utf-8")).hexdigest()
    cache_dir = os.path.join(os.path.dirname(model_path), "..", "output", "cache")
    cache_dir = os.path.abspath(cache_dir)
    if use_cache:
        cached = load_cache(cache_dir, cache_key)
        if cached is not None:
            class_map, merged, meta = cached
            logger.debug("Loaded inference artifacts from cache: %s", cache_key)
            # try to generate lightweight artifacts (report/heat) if missing
            try:
                out_dir = os.path.dirname(img_path) or os.getcwd()
                heat = entropy_heatmap(merged)
                heat_path = os.path.join(out_dir, f"{os.path.basename(img_path)}.entropy.png")
                cv2.imwrite(heat_path, heat)
                generate_report(img_path, class_map, merged, {"cached": True}, out_dir=out_dir)
            except Exception:
                pass
            return class_map, merged

    t_tile_start = time.perf_counter()
    patches, coords = tile_image(image, win_size=win_size, step_size=step_size)
    t_tile = time.perf_counter() - t_tile_start
    logger.debug("Tiled image into %d patches", len(patches))

    # Predict
    t_pred_start = time.perf_counter()
    pred_patches = predict_patches(model_obj, metadata, model_type, patches, batch_size=batch_size)
    t_pred = time.perf_counter() - t_pred_start

    # Merge
    t_merge_start = time.perf_counter()
    merged = merge_patches(pred_patches, coords, image.shape, win_size=win_size, method=merge_method)
    t_merge = time.perf_counter() - t_merge_start

    # Convert probabilities to class map or threshold map
    if manual_th is None:
        class_map = np.argmax(merged, axis=-1)
    else:
        assert len(manual_th) == merged.shape[-1] - 1, f"{manual_th}, {merged.shape[-1]}"
        class_map = np.zeros(merged.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(merged[..., idx + 1] > th, 1, 0)

    total_time = time.perf_counter() - t0
    timings = {
        "model_load_s": round(t_model, 4),
        "tiling_s": round(t_tile, 4),
        "predict_s": round(t_pred, 4),
        "merge_s": round(t_merge, 4),
        "total_s": round(total_time, 4),
    }

    # Save cache for future runs
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # compute model checksum if available
        meta = {"model": model_path, "timings": timings}
        try:
            model_file = os.path.join(model_path, "model.onnx")
            if os.path.exists(model_file):
                import hashlib as _h
                with open(model_file, "rb") as fh:
                    md = _h.sha256(fh.read()).hexdigest()
                meta["model_sha256"] = md
        except Exception:
            pass
        save_cache(cache_dir, cache_key, class_map=class_map, merged=merged, meta=meta)
    except Exception:
        pass

    # Post-processing artifacts: entropy heatmap and report
    try:
        out_dir = os.path.dirname(img_path) or os.getcwd()
        heat = entropy_heatmap(merged)
        heat_path = os.path.join(out_dir, f"{os.path.basename(img_path)}.entropy.png")
        cv2.imwrite(heat_path, heat)

        low_conf_mask = confidence_filter(merged, threshold=0.6)

        # derive boxes from low confidence mask and create overlay
        boxes = low_conf_bboxes(low_conf_mask, min_area=64)
        overlay_path = os.path.join(out_dir, f"{os.path.basename(img_path)}.overlay.png")
        try:
            create_overlay(img_path, boxes, overlay_path)
        except Exception:
            overlay_path = None

        report = {
            "image": img_path,
            "shape": image.shape,
            "nonzero_pixels": int(np.count_nonzero(class_map)),
            "low_conf_pixels": int(np.count_nonzero(low_conf_mask)),
            "low_conf_regions": len(boxes),
            "timings": timings,
        }
        generate_report(img_path, class_map, merged, report, out_dir=out_dir)

        # detect note centers and perform advanced grouping + constraints
        try:
            centers = nge.extract_note_centers_from_classmap(class_map, class_value=2)
            unit = round(np.mean([get_unit_size(x, y) for x, y in centers])) if centers else 10
            groups = nge.advanced_group_notes(centers, unit_size=unit)
            constrained = nge.apply_music_constraints([g["indices"] for g in groups], centers=centers)
            report["note_centers"] = len(centers)
            report["group_candidates"] = len(groups)
        except Exception:
            centers = []
            groups = []
            constrained = []

        # overlay group highlights (draw centroids and group ids)
        try:
            if centers:
                # draw circles for centers and group bounding boxes
                group_overlay = os.path.join(out_dir, f"{os.path.basename(img_path)}.groups.png")
                img_cv = cv2.imread(img_path)
                for gi, g in enumerate(groups):
                    color = (int((gi*73)%255), int((gi*127)%255), int((gi*47)%255))
                    for idx in g.get("indices", g):
                        x, y = centers[idx]
                        cv2.circle(img_cv, (x, y), 3, color, -1)
                    # centroid
                    cx, cy = g.get("centroid", (0,0))
                    cv2.putText(img_cv, str(gi), (cx+2, cy+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.imwrite(group_overlay, img_cv)
                overlay_path = overlay_path or group_overlay
        except Exception:
            pass

        # generate interactive GUI viewer
        try:
            html_path = generate_gui(img_path, overlay_path, heat_path, boxes=boxes, out_dir=out_dir)
            logger.debug("Generated interactive viewer: %s", html_path)
        except Exception:
            pass
    except Exception:
        logger.exception("Post-processing artifacts generation failed")

    return class_map, merged



def predict(region: ndarray, model_name: str) -> str:
    """Nhận diện loại ký hiệu nhạc từ một khúc ảnh nhỏ bằng mô hình phân loại. 
    Trả về tên ký hiệu (ví dụ: "nốt đỏ", "ký hiệu nhạc", v.v.).
    """
    if np.max(region) == 1:
        region *= 255
    
    # Tải mô hình, resize vùng ảnh, chuẩn bị dữ liệu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        m_info = pickle.load(
            open(os.path.join(MODULE_PATH, f"sklearn_models/{model_name}.model"), "rb")
        )
    
    model = m_info["model"]
    w = m_info["w"]
    h = m_info["h"]
    region = np.array(Image.fromarray(region.astype(np.uint8)).resize((w, h)))

    # Dự đoán loại ký hiệu: hỗ trợ cả mô hình Keras (hình ảnh) và sklearn (bảng dữ liệu)
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) == 4:
        x = region.reshape(1, h, w, 1).astype(np.float32)
        pred_raw = model.predict(x, verbose=0)
        pred_idx = int(np.argmax(pred_raw, axis=-1)[0])
    else:
        pred_raw = model.predict(region.reshape(1, -1))
        if isinstance(pred_raw, np.ndarray) and pred_raw.ndim > 1:
            pred_idx = int(np.argmax(pred_raw, axis=-1)[0])
        else:
            pred_idx = int(pred_raw[0])

    return m_info["class_map"][pred_idx]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    img_path = os.path.join(project_root, "docs/images/de/dan-ga-con.png")

    # Mô hình unet_big: lấy class_map với staff (1) và symbols (2)
    model_path = os.path.join(script_dir, "checkpoints/unet_big")
    class_map, out = inference(model_path, img_path)

    print(f"Kích thước bản đồ kết quả: {class_map.shape}")
    print("Một phần ma trận class_map (khoảng cố định [763:771, 520:528]):")
    print(np.array2string(class_map[763:771, 520:528], separator=", "))
    print(f"Số pixel đã được gán trong class_map: {int(np.count_nonzero(class_map))}")

    # Tách staff và symbols từ class_map
    staff = np.where(class_map == 1, 255, 0).astype(np.uint8)
    symbols = np.where(class_map == 2, 255, 0).astype(np.uint8)

    print(f"Số pixel staff đã được gán: {int(np.count_nonzero(staff))}")
    print(f"Số pixel symbols đã được gán: {int(np.count_nonzero(symbols))}")
    print("Một phần ma trận staff (khoảng cố định [791:799, 270:278]):")
    print(np.array2string(staff[791:799, 270:278], separator=", "))
    print("Một phần ma trận symbols (khoảng cố định [746:754, 40:48]):")
    print(np.array2string(symbols[746:754, 40:48], separator=", "))
