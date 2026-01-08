
import json
import os
from pathlib import Path

files = [
    'a_team/data/evaluation/evaluation_results_V1_20260108_145522.json',
    'a_team/data/evaluation/evaluation_results_V2_20260107_224558.json',
    'a_team/data/evaluation/evaluation_results_V3_20260107_234511.json',
    'a_team/data/evaluation/evaluation_results_V3_20260108_094141.json',
    'a_team/data/evaluation/evaluation_results_V4_20260108_104637.json',
    'a_team/data/evaluation/evaluation_results_V4_20260108_105911.json',
    'a_team/data/evaluation/evaluation_results_V5_20260108_114126.json',
    'a_team/data/evaluation/evaluation_results_V7_20260108_121414.json',
    'a_team/data/evaluation/evaluation_results_V7_20260108_122811.json',
    'a_team/data/evaluation/evaluation_results_V8_20260108_141339.json',
    'a_team/data/evaluation/evaluation_results_V8_20260108_141855.json'
]

results = []

for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summary = data.get('summary', {})
            metrics = summary.get('metrics', {})
            timestamp = summary.get('timestamp', '')

            # extract version from filename if not in summary or as a fallback/display name
            filename = os.path.basename(fpath)
            # e.g. evaluation_results_V1_...
            version_part = filename.split('_')[2] if len(
                filename.split('_')) > 2 else "Unknown"

            # Clean up metrics keys (remove user_input.faithfulness etc if present, keep simple names)
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_metrics[k.split('.')[-1]] = v  # e.g. 'faithfulness'

            results.append({
                'version': version_part,
                'file': filename,
                'metrics': clean_metrics,
                'timestamp': timestamp
            })
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

# Deduplicate by picking the latest one for each version if multiple (Optional? User listed all, maybe wants all displayed?)
# User said "organize by version". If multiple V3s exist, I should probably show them or pick the best/latest.
# Let's show all and sort by version and timestamp.

results.sort(key=lambda x: (x['version'], x['timestamp']))

print("| Version | Faithfulness | relevancy | Context Precision | Context Recall | Timestamp |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")

for r in results:
    m = r['metrics']
    faith = m.get('faithfulness', 0)
    rel = m.get('answer_relevancy', 0)
    prec = m.get('context_precision', 0)
    recall = m.get('context_recall', 0)

    # Format to 4 decimal places
    print(f"| {r['version']} | {faith:.4f} | {rel:.4f} | {prec:.4f} | {recall:.4f} | {r['timestamp']} |")
