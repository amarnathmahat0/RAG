"""Generate Markdown report and matplotlib charts from benchmark results."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class ReportGenerator:
    def generate(self, report_dict: dict, run_id: str) -> tuple[str, list[str]]:
        """Return (markdown_path, list[chart_paths])."""
        md_path = self._write_markdown(report_dict, run_id)
        chart_paths = self._write_charts(report_dict, run_id)
        return md_path, chart_paths

    def _write_markdown(self, r: dict, run_id: str) -> str:
        lines = [
            f"# NexusRAG Evaluation Report — Run `{run_id}`\n",
            f"**Total cases:** {r['total_cases']}",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Context Precision | {r['mean_context_precision']:.4f} |",
            f"| Context Recall | {r['mean_context_recall']:.4f} |",
            f"| Faithfulness (LLM) | {r['mean_faithfulness']:.4f} |",
            f"| Relevancy (LLM) | {r['mean_relevancy']:.4f} |",
            f"| Completeness (LLM) | {r['mean_completeness']:.4f} |",
            "",
            "## Latency",
            "",
            "| Percentile | ms |",
            "|------------|-----|",
            f"| p50 | {r['latency']['p50']} |",
            f"| p95 | {r['latency']['p95']} |",
            f"| p99 | {r['latency']['p99']} |",
            f"| mean | {r['latency']['mean']} |",
            "",
            "## Tier Distribution",
            "",
        ]
        for tier, count in r["tier_distribution"].items():
            lines.append(f"- **{tier}**: {count} queries")
        lines += [
            "",
            "## Per-Case Results",
            "",
            "| # | Difficulty | Tier | Precision | Recall | Faithfulness | Latency (ms) |",
            "|---|-----------|------|-----------|--------|--------------|-------------|",
        ]
        for c in r["per_case"][:50]:
            lines.append(
                f"| {c['index']+1} | {c['difficulty']} | {c['actual_tier']} | "
                f"{c.get('det_context_precision', 0):.3f} | "
                f"{c.get('det_context_recall', 0):.3f} | "
                f"{c.get('llm_faithfulness_score', 0):.3f} | "
                f"{c['latency_ms']:.0f} |"
            )

        md = "\n".join(lines)
        out = Path(_settings.eval_runs_dir) / f"report_{run_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        logger.info("Markdown report: %s", out)
        return str(out)

    def _write_charts(self, r: dict, run_id: str) -> list[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping charts.")
            return []

        Path(_settings.metrics_dir).mkdir(parents=True, exist_ok=True)
        chart_paths: list[str] = []

        # ── Metrics bar chart ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics = {
            "Precision": r["mean_context_precision"],
            "Recall": r["mean_context_recall"],
            "Faithfulness": r["mean_faithfulness"],
            "Relevancy": r["mean_relevancy"],
            "Completeness": r["mean_completeness"],
        }
        bars = ax.bar(metrics.keys(), metrics.values(), color="#4C72B0")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"NexusRAG Metrics — {run_id}")
        ax.set_ylabel("Score")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", fontsize=9,
            )
        p = str(Path(_settings.metrics_dir) / f"metrics_{run_id}.png")
        fig.savefig(p, bbox_inches="tight", dpi=120)
        plt.close(fig)
        chart_paths.append(p)

        # ── Latency histogram ────────────────────────────────────────────────────
        latencies = [c["latency_ms"] for c in r["per_case"] if c.get("latency_ms")]
        if latencies:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(latencies, bins=20, color="#DD8452", edgecolor="white")
            ax.set_title("Latency Distribution (ms)")
            ax.set_xlabel("ms")
            ax.set_ylabel("Count")
            for pct, label in [(50, "p50"), (95, "p95"), (99, "p99")]:
                val = r["latency"][label.lower()]
                ax.axvline(val, linestyle="--", alpha=0.7, label=f"{label}={val:.0f}ms")
            ax.legend()
            p2 = str(Path(_settings.metrics_dir) / f"latency_{run_id}.png")
            fig.savefig(p2, bbox_inches="tight", dpi=120)
            plt.close(fig)
            chart_paths.append(p2)

        return chart_paths
