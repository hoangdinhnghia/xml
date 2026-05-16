#!/usr/bin/env python3
"""Run the Oemer pipeline on all images with ground-truth in `dap_an/` and aggregate reports.
Produces bench_summary.csv with basic metrics.
"""
import os
import subprocess
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "dap_an"
OUTPUT_DIR = ROOT / "output" / "de"
SO_SANH = ROOT / "so_sanh.py"
ETE = ROOT / "oemer" / "ete.py"

results = []
for gt in sorted(GT_DIR.glob("*.json")):
    name = gt.stem
    img_candidate = ROOT / "images" / "de" / f"{name}.png"
    if not img_candidate.exists():
        print(f"Image not found for {name}, skipping")
        continue
    # run ete to produce output MusicXML
    cmd = ["python", str(ETE), str(img_candidate), "-o", str(OUTPUT_DIR)]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("ETE failed for", name)
        continue
    # run so_sanh
    cmd2 = ["python", str(SO_SANH), str(img_candidate), "--gt-dir", str(GT_DIR), "--pred-dir", str(OUTPUT_DIR)]
    print("Evaluating:", " ".join(cmd2))
    try:
        proc = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        out = proc.stdout
    except subprocess.CalledProcessError as e:
        out = e.stdout + "\n" + e.stderr
    # collect report.json if exists
    report_json = ROOT / "images" / "de" / f"{name}.report.json"
    entry = {"name": name, "so_sanh_output": out.strip()}
    if report_json.exists():
        try:
            r = json.loads(report_json.read_text(encoding="utf-8"))
            entry.update(r)
        except Exception:
            pass
    results.append(entry)

# write CSV
csv_path = ROOT / "bench_summary.csv"
if results:
    keys = sorted(set().union(*(r.keys() for r in results)))
    with open(csv_path, "w", encoding="utf-8", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("Wrote bench summary to", csv_path)
else:
    print("No results collected.")
