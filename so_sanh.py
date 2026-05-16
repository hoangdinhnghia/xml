#!/usr/bin/env python3
"""Simple evaluation script - Compare predicted pitches with ground truth"""

import argparse
import json
import re
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from typing import Tuple




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pitch accuracy by image name")
    parser.add_argument("image_name", help="Image name (e.g., dan-ga-con or test0.png)")
    parser.add_argument("--gt-dir", default="dap_an", help="Ground truth directory (default: dap_an)")
    parser.add_argument("--pred-dir", default="output/de", help="Prediction directory (default: output/de)")
    return parser.parse_args()


def extract_pitches_from_text(text: str) -> List[str]:
    """Extract simple pitch tokens from text."""
    return re.findall(r"[a-g][#b]?\d", text.lower())


def normalize_symbol_list(symbols: List[str]) -> List[str]:
    """Normalize pitch/rest tokens for comparison."""
    placeholders = {"-", "—", "_", "none", "null", ""}
    normalized: List[str] = []
    for symbol in symbols:
        value = str(symbol).strip().lower()
        if value in placeholders:
            normalized.append("rest")
        else:
            normalized.append(value)
    return normalized


def format_symbol_list(symbols: List[str]) -> str:
    """Format symbols for display."""
    display_tokens = []
    for symbol in symbols:
        display_tokens.append("lặng tròn" if symbol == "rest" else symbol)
    return " ".join(display_tokens)


def extract_pitches_from_xml(file_path: Path) -> List[str]:
    """Extract pitches from MusicXML in file order."""
    try:
        root = ET.parse(file_path).getroot()
    except Exception:
        return extract_pitches_from_text(file_path.read_text(encoding="utf-8", errors="ignore"))

    pitches: List[str] = []
    for note in root.findall(".//note"):
        if note.find("./rest") is not None:
            pitches.append("rest")
            continue

        pitch_elem = note.find("./pitch")
        if pitch_elem is None:
            continue

        step = pitch_elem.findtext("step", "").lower()
        if not step:
            continue

        alter = pitch_elem.findtext("alter", "")
        octave = pitch_elem.findtext("octave", "")
        alter_str = "#" if alter == "1" else ("b" if alter == "-1" else "")
        pitches.append(f"{step}{alter_str}{octave}")

    return normalize_symbol_list(pitches)


def load_pitches(file_path: Path) -> List[str]:
    """Load pitch tokens from XML, JSON or text."""
    suffix = file_path.suffix.lower()
    if suffix in [".xml", ".musicxml"]:
        return extract_pitches_from_xml(file_path)

    if suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return extract_pitches_from_text(file_path.read_text(encoding="utf-8", errors="ignore"))

        if isinstance(data, dict):
            if isinstance(data.get("pitches"), list):
                return normalize_symbol_list([str(p) for p in data["pitches"]])
            if isinstance(data.get("rows"), list):
                pitches: List[str] = []
                for item in data["rows"]:
                    if isinstance(item, dict) and isinstance(item.get("pitches"), list):
                        pitches.extend([str(p) for p in item["pitches"]])
                return normalize_symbol_list(pitches)
            if isinstance(data.get("measure_staff"), list):
                pitches: List[str] = []
                for item in data["measure_staff"]:
                    if isinstance(item, dict) and isinstance(item.get("pitches"), list):
                        pitches.extend([str(p) for p in item["pitches"]])
                return normalize_symbol_list(pitches)
            # Handle new measure/staff/voice structure
            if isinstance(data.get("measures"), list):
                pitches: List[str] = []
                for measure in data["measures"]:
                    if isinstance(measure.get("staves"), dict):
                        for staff_num in sorted(measure["staves"].keys(), key=lambda x: int(x) if x.isdigit() else 0):
                            staff = measure["staves"][staff_num]
                            if isinstance(staff, dict):
                                for voice_num in sorted(staff.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                                    voice = staff[voice_num]
                                    if isinstance(voice, list):
                                        pitches.extend([str(p) for p in voice])
                return normalize_symbol_list(pitches)

        return extract_pitches_from_text(file_path.read_text(encoding="utf-8", errors="ignore"))

    return extract_pitches_from_text(file_path.read_text(encoding="utf-8", errors="ignore"))


def find_prediction_file(base_name: str, pred_dir: Path) -> Path | None:
    """Find a prediction file in the configured directory or sibling output folders."""
    text_exts = {".xml", ".musicxml", ".txt", ".json"}

    search_dirs = [pred_dir]
    output_root = pred_dir.parent
    if output_root.exists():
        for child_dir in sorted(output_root.iterdir()):
            if child_dir.is_dir() and child_dir not in search_dirs:
                search_dirs.append(child_dir)

    for directory in search_dirs:
        for ext in text_exts:
            candidate = directory / f"{base_name}{ext}"
            if candidate.exists():
                return candidate

    return None


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Calculate edit distance between two sequences"""
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    if not seq2:
        return len(seq1)
    
    prev_row = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1, start=1):
        cur_row = [i]
        for j, c2 in enumerate(seq2, start=1):
            ins = prev_row[j] + 1
            delete = cur_row[j - 1] + 1
            sub = prev_row[j - 1] + (0 if c1 == c2 else 1)
            cur_row.append(min(ins, delete, sub))
        prev_row = cur_row
    return prev_row[-1]


def calculate_accuracy(pred_pitches: List[str], gt_pitches: List[str]) -> Tuple[float, str]:
    """Calculate pitch accuracy and status"""
    if not gt_pitches:
        return 1.0 if not pred_pitches else 0.0, "OK" if not pred_pitches else "FAIL"
    
    distance = levenshtein_distance(pred_pitches, gt_pitches)
    accuracy = max(0.0, 1.0 - (distance / len(gt_pitches)))
    status = "✓" if distance == 0 else "✗"
    
    return accuracy, status


def extract_xml_structure(file_path: Path) -> List[Tuple[int, int, int, List[str]]]:
    """Extract (measure, staff, voice, pitches) from MusicXML in order."""
    try:
        root = ET.parse(file_path).getroot()
    except Exception:
        return []

    rows: List[Tuple[int, int, int, List[str]]] = []
    for measure in root.findall(".//measure"):
        m_text = measure.get("number", "")
        measure_no = int(m_text) if m_text.isdigit() else 0
        voice_groups: dict[Tuple[int, int], List[str]] = {}
        rest_by_staff: set[int] = set()

        for note in measure.findall("note"):
            if note.find("./rest") is not None:
                staff = int(note.findtext("staff", "1")) if note.findtext("staff", "1").isdigit() else 1
                rest_by_staff.add(staff)
                continue

            pitch_elem = note.find("./pitch")
            if pitch_elem is None:
                continue

            step = pitch_elem.findtext("step", "").lower()
            if not step:
                continue

            alter = pitch_elem.findtext("alter", "")
            octave = pitch_elem.findtext("octave", "")
            alter_str = "#" if alter == "1" else ("b" if alter == "-1" else "")
            pitch = f"{step}{alter_str}{octave}"

            staff = int(note.findtext("staff", "1")) if note.findtext("staff", "1").isdigit() else 1
            voice = int(note.findtext("voice", "1")) if note.findtext("voice", "1").isdigit() else 1

            voice_groups.setdefault((staff, voice), []).append(pitch)

        for (staff, voice), pitches in sorted(voice_groups.items()):
            rows.append((measure_no, staff, voice, pitches))

        staff_with_pitches = {staff for staff, _voice in voice_groups.keys()}
        for staff in sorted(rest_by_staff - staff_with_pitches):
            rows.append((measure_no, staff, 1, ["rest"]))

    return rows




def format_structured_comparison(gt_data: dict, pred_rows: List[Tuple[int, int, int, List[str]]]) -> List[str]:
    """Format comparison by measure/staff structure."""
    lines: List[str] = []
    lines.append("\n  So sánh:")
    lines.append("  " + "="*70)
    
    if not isinstance(gt_data.get("measures"), list):
        lines.append("  (Đáp án không có cấu trúc measure/staff)")
        return lines
    
    # Build GT structure (combine all voices)
    gt_dict: dict = {}
    for measure in gt_data["measures"]:
        m_num = measure.get("number", 0)
        if m_num not in gt_dict:
            gt_dict[m_num] = {}
        staves = measure.get("staves", {})
        for staff_num, voices_dict in staves.items():
            staff_num_int = int(staff_num) if isinstance(staff_num, str) and staff_num.isdigit() else staff_num
            if staff_num_int not in gt_dict[m_num]:
                gt_dict[m_num][staff_num_int] = []
            # Combine all voices for this staff
            for voice_num, pitches in voices_dict.items():
                gt_dict[m_num][staff_num_int].extend(normalize_symbol_list([str(p) for p in pitches]))
    
    # Build Pred structure (combine all voices)
    pred_dict: dict = {}
    for measure, staff, voice, pitches in pred_rows:
        if measure not in pred_dict:
            pred_dict[measure] = {}
        if staff not in pred_dict[measure]:
            pred_dict[measure][staff] = []
        pred_dict[measure][staff].extend(normalize_symbol_list([str(p) for p in pitches]))
    
    # Compare
    matched = 0
    total = 0
    for m_num in sorted(set(list(gt_dict.keys()) + list(pred_dict.keys()))):
        gt_staves = gt_dict.get(m_num, {})
        pred_staves = pred_dict.get(m_num, {})
        
        for staff_num in sorted(set(list(gt_staves.keys()) + list(pred_staves.keys()))):
            gt_pitches = gt_staves.get(staff_num, [])
            pred_pitches = pred_staves.get(staff_num, [])
            
            total += 1
            match = "✓" if gt_pitches == pred_pitches else "✗"
            if gt_pitches == pred_pitches:
                matched += 1
            
            gt_str = format_symbol_list(gt_pitches) if gt_pitches else "—"
            pred_str = format_symbol_list(pred_pitches) if pred_pitches else "—"
            
            staff_name = "Treble" if staff_num == 1 else "Bass"
            lines.append(f"  M{m_num}|{staff_name:6s} {match} | Đáp án: {gt_str:<25s} | XML: {pred_str:<25s}")
    
    lines.append("  " + "="*70)
    lines.append(f"  Tổng: {matched}/{total} đúng ({matched*100/total if total > 0 else 0:.1f}%)")
    return lines


def main():
    args = parse_args()
    
    # Extract image name (without extension, auto-add .png if needed)
    image_path = Path(args.image_name)
    if image_path.suffix == "":
        image_path = Path(f"{args.image_name}.png")
    base_name = image_path.stem  # dan-ga-con from dan-ga-con.png
    
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    
    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        return
    if not pred_dir.exists():
        print(f"❌ Prediction directory not found: {pred_dir}")
        return
    
    # Find ground truth file
    text_exts = {".xml", ".musicxml", ".txt", ".json"}
    gt_file = None
    for ext in text_exts:
        candidate = gt_dir / f"{base_name}{ext}"
        if candidate.exists():
            gt_file = candidate
            break
    
    if not gt_file:
        print(f"❌ Ground truth file not found for: {base_name}")
        print(f"   Searched in: {gt_dir}")
        return
    
    # Find prediction file (try configured dir first, then sibling output folders)
    pred_file = find_prediction_file(base_name, pred_dir)
    
    if not pred_file:
        print(f"❌ Prediction file not found for: {base_name}")
        print(f"   Searched in: {pred_dir} and sibling output folders")
        return
    
    # Extract and compare pitches
    pred_pitches = load_pitches(pred_file)
    
    # Load GT data
    try:
        gt_data = json.loads(gt_file.read_text(encoding="utf-8", errors="ignore"))
    except:
        gt_data = {}
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Evaluation for: {base_name}")
    print(f"{'='*70}")
    print(f"Ground Truth: {gt_file}")
    print(f"Prediction:  {pred_file}")
    
    # Structured comparison
    xml_rows = extract_xml_structure(pred_file)
    for line in format_structured_comparison(gt_data, xml_rows):
        print(line)
    
    print(f"{'='*70}\n")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
