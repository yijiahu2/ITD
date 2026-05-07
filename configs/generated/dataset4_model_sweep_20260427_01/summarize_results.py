
from __future__ import annotations
import csv, json, re
from pathlib import Path
import yaml

RUN_ID = "dataset4_model_sweep_20260427_01"
CFG_DIR = Path("/home/xth/forest_agent_project/configs/generated/dataset4_model_sweep_20260427_01")
OUT_ROOT = Path("/home/xth/forest_agent_project/outputs/dataset4_model_sweep_20260427_01")
manifest = yaml.safe_load((CFG_DIR / 'manifest.yaml').read_text(encoding='utf-8'))['models']

AP_PATTERNS = [
    ('segm_AP50', re.compile(r"(?:coco/)?segm_mAP_50['\"=:\s]+([0-9.]+)")),
    ('segm_AP75', re.compile(r"(?:coco/)?segm_mAP_75['\"=:\s]+([0-9.]+)")),
    ('bbox_AP50', re.compile(r"(?:coco/)?bbox_mAP_50['\"=:\s]+([0-9.]+)")),
    ('bbox_AP75', re.compile(r"(?:coco/)?bbox_mAP_75['\"=:\s]+([0-9.]+)")),
    ('AP50', re.compile(r"\bAP50\b[^0-9]*([0-9.]+)")),
    ('AP75', re.compile(r"\bAP75\b[^0-9]*([0-9.]+)")),
]

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def extract_metrics_from_text(text: str):
    hits = {}
    for key, pat in AP_PATTERNS:
        vals = []
        for match in pat.finditer(text):
            try:
                vals.append(float(match.group(1)))
            except Exception:
                pass
        if vals:
            hits[key] = vals[-1]
    return hits

def collect_text(model_dir: Path, log_path: Path):
    chunks = []
    for path in [log_path, *model_dir.rglob('*.json'), *model_dir.rglob('*.log')]:
        if path.exists() and path.is_file():
            try:
                chunks.append(f"\n--- {path} ---\n" + path.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
    return '\n'.join(chunks)

rows = []
for key, meta in manifest.items():
    model_dir = Path(meta['output_dir'])
    log_path = OUT_ROOT / 'logs' / f'{key}.log'
    status_dir = OUT_ROOT / 'status'
    exitcode_path = status_dir / f'{key}.exitcode'
    prepare_summary = load_json(model_dir / 'external_segmentation_dataset' / 'prepare_summary.json') or {}
    train_summary = load_json(model_dir / 'segmentation_training' / 'train_summary.json') or {}
    test_summary_path = train_summary.get('test_summary_json') or str(model_dir / 'segmentation_training' / 'evaluation' / 'test_summary.json')
    text = collect_text(model_dir, log_path)
    metrics = extract_metrics_from_text(text)
    ap50 = metrics.get('segm_AP50') or metrics.get('AP50')
    ap75 = metrics.get('segm_AP75') or metrics.get('AP75')
    rows.append({
        'model_key': key,
        'model_name': meta['model_name'],
        'algorithm': meta['algorithm'],
        'status': 'done' if (status_dir / f'{key}.done').exists() else 'failed' if (status_dir / f'{key}.failed').exists() else 'running_or_pending',
        'exitcode': exitcode_path.read_text().strip() if exitcode_path.exists() else '',
        'ap50': ap50,
        'ap75': ap75,
        'best_ckpt': train_summary.get('best_ckpt', ''),
        'test_summary_json': test_summary_path if Path(test_summary_path).exists() else '',
        'train_images': ((prepare_summary.get('counts') or {}).get('train_images')),
        'val_images': ((prepare_summary.get('counts') or {}).get('val_images')),
        'test_images': ((prepare_summary.get('counts') or {}).get('test_images')),
        'holdout': json.dumps(prepare_summary.get('holdout_test') or {}, ensure_ascii=False),
    })

rows_sorted = sorted(rows, key=lambda r: (r['ap75'] is not None, r['ap75'] or -1, r['ap50'] or -1), reverse=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
with (OUT_ROOT / 'model_sweep_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    writer.writeheader(); writer.writerows(rows_sorted)

report = [
    '# Dataset_4 Segmentation Model Sweep Final Report',
    '',
    f'- run_id: `{RUN_ID}`',
    f'- output_root: `{OUT_ROOT}`',
    '- test split: 20% holdout from train + 20% holdout from validation, removed from train/val before training',
    '- primary ranking: AP75 first, AP50 second',
    '',
    '## Ranking',
    '',
    '| Rank | Model | Algorithm | Status | AP50 | AP75 | Test images | Best checkpoint |',
    '|---:|---|---|---|---:|---:|---:|---|',
]
for idx, row in enumerate(rows_sorted, 1):
    report.append(f"| {idx} | {row['model_name']} | `{row['algorithm']}` | {row['status']} | {row['ap50'] if row['ap50'] is not None else 'NA'} | {row['ap75'] if row['ap75'] is not None else 'NA'} | {row['test_images'] if row['test_images'] is not None else 'NA'} | `{row['best_ckpt']}` |")
report.extend(['', '## Split / Holdout Evidence', ''])
for row in rows_sorted:
    report.append(f"### {row['model_name']}")
    report.append(f"- train_images: `{row['train_images']}`, val_images: `{row['val_images']}`, test_images: `{row['test_images']}`")
    report.append(f"- holdout: `{row['holdout']}`")
    report.append(f"- test_summary_json: `{row['test_summary_json']}`")
    report.append('')
(OUT_ROOT / 'FINAL_REPORT.md').write_text('\n'.join(report).strip() + '\n', encoding='utf-8')
print(OUT_ROOT / 'FINAL_REPORT.md')
