#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a concise summary from estimate_cifar100_runtime.py output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days >= 1:
        return f"{int(days)}天 {int(hours):02d}小时 {int(minutes):02d}分"
    if hours >= 1:
        return f"{int(hours)}小时 {int(minutes):02d}分"
    if minutes >= 1:
        return f"{int(minutes)}分 {secs:04.1f}秒"
    return f"{secs:.2f}秒"


def safe_seconds(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    rows: list[dict[str, Any]] = []

    for item in payload.get("results", []):
        total = safe_seconds(item.get("estimated_seconds"))
        stages = item.get("stages") or []
        valid_stages = [
            stage for stage in stages
            if math.isfinite(safe_seconds(stage.get("estimated_seconds")))
        ]
        dominant = max(
            valid_stages,
            key=lambda stage: safe_seconds(stage.get("estimated_seconds")),
            default=None,
        )
        dominant_seconds = safe_seconds(dominant.get("estimated_seconds")) if dominant else 0.0
        share = dominant_seconds / total if dominant and math.isfinite(total) and total > 0 else float("nan")

        ratio = "共享"
        if dominant and dominant.get("ratio") is not None:
            ratio = f"{dominant['ratio']}%"

        rows.append({
            "method": str(item.get("method", "")),
            "status": str(item.get("status", "")),
            "seconds": total,
            "time": format_duration(total),
            "dominant_stage": str(dominant.get("stage", "—")) if dominant else "—",
            "dominant_ratio": ratio,
            "dominant_share": share,
            "warning": str(item.get("warning", "") or ""),
        })

    rows.sort(key=lambda row: row["seconds"] if math.isfinite(row["seconds"]) else float("inf"))
    return metadata, rows


def print_table(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print("=" * 112)
    print("CIFAR-100 数据选择方法运行时间预估：精简结论")
    print("=" * 112)
    print(
        f"GPU: {metadata.get('gpu_name', 'unknown')} | "
        f"机器数: {metadata.get('machines', 'unknown')} | "
        f"保留比例: {metadata.get('ratios', [])}"
    )
    print("说明：排名按生成 60%/70%/80%/90% 四组 mask 的总墙钟时间升序排列。")
    print()
    print(f"{'排名':<6}{'方法':<15}{'总时间':<18}{'主要耗时阶段':<43}{'比例':<8}{'占比':>7}{'状态':>9}")
    print("-" * 112)

    valid_rank = 0
    for row in rows:
        if math.isfinite(row["seconds"]):
            valid_rank += 1
            rank = str(valid_rank)
        else:
            rank = "—"
        share = f"{row['dominant_share']:.1%}" if math.isfinite(row["dominant_share"]) else "—"
        print(
            f"{rank:<6}{row['method']:<15}{row['time']:<18}"
            f"{row['dominant_stage'][:40]:<43}{row['dominant_ratio']:<8}{share:>7}{row['status']:>9}"
        )
        if row["warning"]:
            print(f"      警告：{row['warning']}")

    valid = [row for row in rows if math.isfinite(row["seconds"])]
    print()
    if valid:
        fastest = valid[0]
        slowest = valid[-1]
        multiple = slowest["seconds"] / fastest["seconds"] if fastest["seconds"] > 0 else float("nan")
        print(f"最快方法：{fastest['method']}，约 {fastest['time']}。")
        print(f"最慢方法：{slowest['method']}，约 {slowest['time']}。")
        if math.isfinite(multiple):
            print(f"最慢约为最快的 {multiple:.1f} 倍。")
    failed = [row["method"] for row in rows if row["status"] != "ok" or not math.isfinite(row["seconds"])]
    if failed:
        print("未成功完成预估的方法：" + "、".join(failed))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank", "method", "estimated_seconds", "estimated_time",
                "dominant_stage", "dominant_ratio", "dominant_share",
                "status", "warning",
            ],
        )
        writer.writeheader()
        rank = 0
        for row in rows:
            if math.isfinite(row["seconds"]):
                rank += 1
                current_rank: int | str = rank
            else:
                current_rank = ""
            writer.writerow({
                "rank": current_rank,
                "method": row["method"],
                "estimated_seconds": "" if not math.isfinite(row["seconds"]) else f"{row['seconds']:.6f}",
                "estimated_time": row["time"],
                "dominant_stage": row["dominant_stage"],
                "dominant_ratio": row["dominant_ratio"],
                "dominant_share": "" if not math.isfinite(row["dominant_share"]) else f"{row['dominant_share']:.6f}",
                "status": row["status"],
                "warning": row["warning"],
            })


def write_markdown(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CIFAR-100 运行时间预估摘要",
        "",
        f"- GPU：{metadata.get('gpu_name', 'unknown')}",
        f"- 机器数：{metadata.get('machines', 'unknown')}",
        f"- 保留比例：{metadata.get('ratios', [])}",
        "- 排名口径：生成四个目标保留比例 mask 的总墙钟时间",
        "",
        "| 排名 | 方法 | 总时间 | 主要耗时阶段 | 对应比例 | 阶段占比 | 状态 |",
        "|---:|---|---:|---|---:|---:|---|",
    ]
    rank = 0
    for row in rows:
        if math.isfinite(row["seconds"]):
            rank += 1
            rank_text = str(rank)
        else:
            rank_text = "—"
        share = f"{row['dominant_share']:.1%}" if math.isfinite(row["dominant_share"]) else "—"
        lines.append(
            f"| {rank_text} | {row['method']} | {row['time']} | "
            f"{row['dominant_stage']} | {row['dominant_ratio']} | {share} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CIFAR-100 runtime estimate JSON.")
    parser.add_argument("--input", type=Path, default=Path("runtime_estimate_cifar100.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("runtime_method_totals.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("runtime_summary.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"找不到结果文件：{args.input}")
    metadata, rows = load_rows(args.input)
    print_table(metadata, rows)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, metadata, rows)
    print(f"\n精简 CSV：{args.output_csv}")
    print(f"Markdown 摘要：{args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())