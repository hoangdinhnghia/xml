"""Post-processing helpers for Oemer: blending, cache, confidence filtering,
report generation, entropy heatmap, simple grouping and lightweight GUI.

These functions are intended to be non-invasive helpers used after model
inference; they do not change model weights or training.
"""

import os
import json
import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2


def _gaussian_1d(size: int, sigma: Optional[float] = None) -> np.ndarray:
    if sigma is None:
        sigma = size / 6.0
    x = np.linspace(- (size - 1) / 2.0, (size - 1) / 2.0, size)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return g / g.max()


def gaussian_weighted_merge(patch_outputs: List[np.ndarray], coords: List[Tuple[int, int]], image_shape: Tuple[int, int, int], win_size: int) -> np.ndarray:
    """Merge patches using a gaussian window to reduce seam artifacts.

    patch_outputs: list of (win_size, win_size, channels)
    coords: list of (x, y)
    image_shape: (h, w, ...)
    """
    h, w = image_shape[:2]
    channels = patch_outputs[0].shape[-1]
    out = np.zeros((h, w, channels), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    g1 = _gaussian_1d(win_size)
    gw = np.outer(g1, g1).astype(np.float32)

    for patch, (x, y) in zip(patch_outputs, coords):
        ph, pw = patch.shape[:2]
        wmap = gw[:ph, :pw]
        for c in range(channels):
            out[y:y+ph, x:x+pw, c] += patch[..., c] * wmap
        weight_sum[y:y+ph, x:x+pw] += wmap

    # avoid division by zero
    weight_sum[weight_sum == 0] = 1.0
    out = out / weight_sum[..., None]
    return out


def save_cache(cache_dir: str, key: str, class_map: np.ndarray, merged: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"{key}.npz")
    np.savez_compressed(npz_path, class_map=class_map.astype(np.int16), merged=merged.astype(np.float32))
    meta = meta or {}
    with open(os.path.join(cache_dir, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def load_cache(cache_dir: str, key: str):
    npz_path = os.path.join(cache_dir, f"{key}.npz")
    json_path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as data:
        class_map = data["class_map"]
        merged = data["merged"]
    meta = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    return class_map, merged, meta


def confidence_filter(merged: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Return a boolean mask where max probability < threshold (low confidence)."""
    probs = merged
    if probs.ndim == 2:
        # single channel
        return np.zeros_like(probs, dtype=np.uint8)
    maxp = np.max(probs, axis=-1)
    low = (maxp < threshold).astype(np.uint8)
    return low


def entropy_heatmap(merged: np.ndarray) -> np.ndarray:
    """Compute per-pixel entropy from softmax probabilities and return a BGR heatmap image."""
    probs = merged.astype(np.float32)
    if probs.ndim == 2:
        probs = np.stack([1 - probs, probs], axis=-1)
    # normalize
    probs = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(probs * np.log(probs), axis=-1)
    # scale to 0-255
    ent = ent / (np.log(probs.shape[-1]) + 1e-12)
    ent_img = (255 * (ent / (ent.max() + 1e-12))).astype(np.uint8)
    colored = cv2.applyColorMap(ent_img, cv2.COLORMAP_JET)
    return colored


def generate_report(image_path: str, class_map: np.ndarray, merged: np.ndarray, report: dict, out_dir: Optional[str] = None, print_to_stdout: bool = False) -> None:
    """Generate a report. If `print_to_stdout` is True, print the report to
    terminal instead of writing files.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]

    r = dict(report)
    r["pixels_per_class"] = {}
    if class_map.ndim == 2:
        vals, counts = np.unique(class_map, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            r["pixels_per_class"][str(int(v))] = int(c)
    else:
        # multi-channel fallback
        for i in range(class_map.shape[-1]):
            r["pixels_per_class"][str(i)] = int(np.count_nonzero(class_map[..., i]))

    # basic timing not available here, but keep placeholder
    r.setdefault("notes", 0)
    if print_to_stdout:
        # Pretty-print to terminal: text then JSON
        lines = [f"Report for: {base}", "=" * 60]
        for k, v in r.items():
            lines.append(f"{k}: {v}")
        print("\n".join(lines))
        print("\nJSON:\n")
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    out_dir = out_dir or os.path.dirname(image_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{base}.report.json")
    txt_path = os.path.join(out_dir, f"{base}.report.txt")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(r, fh, ensure_ascii=False, indent=2)

    lines = [f"Report for: {base}", "=" * 60]
    for k, v in r.items():
        lines.append(f"{k}: {v}")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_manifest(image_path: str, meta: Dict[str, Any], out_dir: Optional[str] = None) -> None:
    out_dir = out_dir or os.path.dirname(image_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    manifest_path = os.path.join(out_dir, f"{base}.manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Grouping helpers were moved to `note_group_extraction.py` to keep
# `postproc` focused on visualization and provenance. See
# `oemer/note_group_extraction.py` for canonical grouping implementations.


def low_conf_bboxes(low_conf_mask: np.ndarray, min_area: int = 64) -> List[Tuple[int, int, int, int]]:
    """Convert binary low-confidence mask to list of bounding boxes (x1,y1,x2,y2)."""
    if low_conf_mask.dtype != np.uint8:
        mask = (low_conf_mask > 0).astype(np.uint8)
    else:
        mask = low_conf_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def create_overlay(image_path: str, boxes: List[Tuple[int, int, int, int]], out_path: str) -> str:
    """Draw semi-transparent red boxes over the original image and save result."""
    img = cv2.imread(image_path)
    if img is None:
        # try reading via PIL fallback
        from PIL import Image
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imwrite(out_path, img)
    return out_path


def generate_gui(image_path: str, overlay_path: Optional[str], heatmap_path: Optional[str], boxes: Optional[List[Tuple[int, int, int, int]]] = None, out_dir: Optional[str] = None, print_to_stdout: bool = False) -> str:
    """Generate a simple HTML file to visualize original image, overlay and heatmap.

    Returns path to generated HTML file.
    """
    out_dir = out_dir or os.path.dirname(image_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    html_path = os.path.join(out_dir, f"{base}.viewer.html")

    def _rel(p):
        return os.path.basename(p) if p else ""

    # Build interactive HTML with boxes that can be toggled and downloaded as JSON
    html = ["<html><head><meta charset=\"utf-8\"><title>OMR Viewer</title>",
            "<style>body{font-family:Arial;} .canvas{position:relative;display:inline-block;} .box{position:absolute;border:2px solid red;opacity:0.6;} .box.toggled{border-color:lime;opacity:0.9}</style>",
            "</head><body>"]
    html.append(f"<h2>OMR Viewer - {base}</h2>")
    html.append("<div>")
    html.append(f"<div class='canvas'><img id='img' src='{_rel(image_path)}' style='max-width:900px;display:block;'><div id='boxes' style='position:absolute;left:0;top:0;'></div></div>")
    if heatmap_path:
        html.append(f"<div style='margin-top:8px;'><h4>Entropy heatmap</h4><img src='{_rel(heatmap_path)}' style='max-width:900px;display:block;'></div>")
    html.append("</div>")

    # Script: place boxes relative to displayed image size, toggle and download JSON
    script = [
        "<script>",
        "const boxes = "+ (json.dumps(boxes) if boxes else '[]') +";",
        "const img = document.getElementById('img');",
        "const container = document.getElementById('boxes');",
        "function placeBoxes(){",
        "  container.innerHTML='';",
        "  const rect = img.getBoundingClientRect();",
        "  const iw = img.naturalWidth, ih = img.naturalHeight;",
        "  const dw = rect.width, dh = rect.height;",
        "  const scaleX = dw/iw, scaleY = dh/ih;",
        "  boxes.forEach((b,idx)=>{",
        "    const [x1,y1,x2,y2]=b;",
        "    const el = document.createElement('div');",
        "    el.className='box';",
        "    el.style.left = (x1*scaleX)+'px';",
        "    el.style.top = (y1*scaleY)+'px';",
        "    el.style.width = ((x2-x1)*scaleX)+'px';",
        "    el.style.height = ((y2-y1)*scaleY)+'px';",
        "    el.dataset.idx = idx;",
        "    el.onclick = function(e){ this.classList.toggle('toggled'); };",
        "    container.appendChild(el);",
        "  });",
        "}",
        "window.addEventListener('load', placeBoxes);",
        "window.addEventListener('resize', placeBoxes);",
        "function downloadJSON(){",
        "  const toggled = Array.from(document.querySelectorAll('.box.toggled')).map(e=>parseInt(e.dataset.idx));",
        "  const payload = {boxes: boxes, reviewed: toggled};",
        "  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));",
        "  const a = document.createElement('a'); a.setAttribute('href', dataStr); a.setAttribute('download', '"+base+".review.json'); document.body.appendChild(a); a.click(); a.remove();",
        "}",
        "</script>"
    ]

    html.extend(script)
    html.append("<div style='margin-top:10px;'><button onclick='downloadJSON()'>Download review JSON</button></div>")
    html.append("</body></html>")

    if print_to_stdout:
        print(f"Viewer for: {base}")
        print("Overlay:", overlay_path)
        print("Entropy heatmap:", heatmap_path)
        print("Boxes:")
        if boxes:
            for i, b in enumerate(boxes):
                print(f"  [{i}] {b}")
        else:
            print("  (none)")
        return ""

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))

    return html_path
