"""Write a lightweight Markdown report from an attention ablation summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


QUESTIONS = [
    "Local attention: check whether mean_peak_displacement_abs is smaller, local_mass_w1/w2 is larger, and adjacent transpositions rise without a wrong-block spike.",
    "Noisy attention: check whether accuracy degrades smoothly as noise increases and order-like errors rise more than random wrong-block errors.",
    "Decay/capacity: check for span cliffs in full_sequence_accuracy_len_6 to full_sequence_accuracy_len_7 and concentrated first-error positions.",
    "Response suppression: check whether repeat_error_rate drops while order_error_rate remains interpretable.",
    "Teacher-forced vs free-running: compare gaps to separate retrieval failure from rollout collapse.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="corsi_artifacts/visual_base/results/attention_ablation_summary.csv")
    parser.add_argument("--output-md", default="corsi_artifacts/visual_base/results/analysis_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    rows = []
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("model_name", "unknown")].append(row)

    lines = [
        "# Attention Ablation Analysis Report",
        "",
        "This report is generated from the collected evaluation summaries. Interpret accuracy together with error taxonomy and attention diagnostics.",
        "",
        "## Key Mechanism Questions",
        "",
    ]
    lines.extend(f"- {question}" for question in QUESTIONS)
    lines.extend(["", "## Runs", ""])
    for model_name, model_rows in sorted(grouped.items()):
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| eval | mode | token_acc | full_seq_acc | entropy | peak_abs | local_w1 | repeat | wrong | order | adj_swap | lcs |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in model_rows:
            lines.append(
                "| {eval_id} | {eval_mode} | {token_accuracy} | {full_sequence_accuracy} | {mean_attention_entropy} | "
                "{mean_peak_displacement_abs} | {local_mass_w1} | {repeat_error_rate} | "
                "{wrong_block_rate} | {order_error_rate} | {adjacent_transposition_rate} | {lcs_normalized_mean} |".format(
                    eval_id=row.get("eval_id", ""),
                    eval_mode=row.get("eval_mode", ""),
                    token_accuracy=row.get("token_accuracy", ""),
                    full_sequence_accuracy=row.get("full_sequence_accuracy", ""),
                    mean_attention_entropy=row.get("mean_attention_entropy", ""),
                    mean_peak_displacement_abs=row.get("mean_peak_displacement_abs", ""),
                    local_mass_w1=row.get("local_mass_w1", ""),
                    repeat_error_rate=row.get("repeat_error_rate", ""),
                    wrong_block_rate=row.get("wrong_block_rate", ""),
                    order_error_rate=row.get("order_error_rate", ""),
                    adjacent_transposition_rate=row.get("adjacent_transposition_rate", ""),
                    lcs_normalized_mean=row.get("lcs_normalized_mean", ""),
                )
            )
        lines.append("")

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
