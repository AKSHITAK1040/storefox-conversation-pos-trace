"""Terminal demo for the CABS Engine."""

from __future__ import annotations

from typing import Iterable

from tabulate import tabulate

from cabs_engine import format_entities, run_demo


def _print_section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def _print_table(headers: list[str], rows: list[list[object]]) -> None:
    print(tabulate(rows, headers=headers, tablefmt="github"))


def _print_acoustic_events(acoustic_data: list[dict[str, object]]) -> None:
    rows = [
        [
            event["conversation_id"],
            event["timestamp"],
            event["speaker_context"],
            format_entities(event["spoken_entities"]),  # type: ignore[arg-type]
            event["upsell_offered"] or "-",
            event["expected_intent"],
        ]
        for event in acoustic_data
    ]
    _print_table(
        ["conversation_id", "timestamp", "speaker_context", "spoken_entities", "upsell_offered", "expected_intent"],
        rows,
    )


def _print_pos_receipts(pos_data: list[dict[str, object]]) -> None:
    rows = [
        [
            receipt["receipt_id"],
            receipt["timestamp"],
            format_entities(receipt["billed_skus"]),  # type: ignore[arg-type]
            receipt["total_amount"],
            receipt["source"],
        ]
        for receipt in pos_data
    ]
    _print_table(["receipt_id", "timestamp", "billed_skus", "total_amount", "source"], rows)


def _print_cost_matrix(acoustic_data: list[dict[str, object]], pos_data: list[dict[str, object]], cost_matrix) -> None:
    rows = []
    for row_idx, audio_event in enumerate(acoustic_data):
        rows.append(
            [audio_event["conversation_id"]]
            + [f"{cost_matrix[row_idx, col_idx]:.2f}" for col_idx in range(len(pos_data))]
        )
    _print_table(["audio\\receipt"] + [receipt["receipt_id"] for receipt in pos_data], rows)


def _print_matches(matched_pairs: list[dict[str, object]]) -> None:
    rows = [
        [
            pair["audio_id"],
            pair["receipt_id"],
            f"{pair['confidence']:.3f}",
            format_entities(pair["spoken_entities"]),  # type: ignore[arg-type]
            format_entities(pair["billed_skus"]),  # type: ignore[arg-type]
            pair["upsell_result"],
            pair["revenue_amount"],
        ]
        for pair in matched_pairs
    ]
    _print_table(
        ["audio_id", "receipt_id", "confidence", "spoken_entities", "billed_skus", "upsell_result", "revenue_amount"],
        rows,
    )


def _print_unmatched_audio(unmatched_audio: list[dict[str, object]]) -> None:
    rows = [
        [
            event["conversation_id"],
            event["timestamp"],
            format_entities(event["spoken_entities"]),  # type: ignore[arg-type]
            event["expected_intent"],
        ]
        for event in unmatched_audio
    ]
    _print_table(["conversation_id", "timestamp", "normalized_entities", "expected_intent"], rows)


def _print_unmatched_pos(unmatched_pos: list[dict[str, object]]) -> None:
    rows = [
        [
            receipt["receipt_id"],
            receipt["timestamp"],
            format_entities(receipt["billed_skus"]),  # type: ignore[arg-type]
            receipt["source"],
        ]
        for receipt in unmatched_pos
    ]
    _print_table(["receipt_id", "timestamp", "normalized_skus", "source"], rows)


def _print_summary(summary: dict[str, int]) -> None:
    rows = [
        ["total conversations", summary["total_conversations"]],
        ["total receipts", summary["total_receipts"]],
        ["matched pairs", summary["matched_pairs"]],
        ["walk-offs detected", summary["potential_walk_offs"]],
        ["online orders detected", summary["likely_online_orders"]],
        ["upsells converted", summary["upsells_converted"]],
        ["upsells missed", summary["upsells_missed"]],
        ["realized upsell revenue estimate", summary["realized_upsell_revenue"]],
    ]
    _print_table(["metric", "value"], rows)


def main() -> None:
    demo = run_demo()
    acoustic_data = demo["acoustic_data"]
    pos_data = demo["pos_data"]
    cost_matrix = demo["cost_matrix"]
    matched_pairs = demo["matched_pairs"]
    unmatched_audio = demo["unmatched_audio"]
    unmatched_pos = demo["unmatched_pos"]
    summary = demo["summary"]
    config = demo["config"]

    print("CABS Engine: Cart-Acoustic Bipartite Synchronization Engine")
    print("Confidence-based alignment of noisy counter conversations to POS receipts")
    print(
        "Weights: semantic={semantic_weight:.2f}, temporal={temporal_weight:.2f}, penalty={penalty}s, threshold={threshold:.2f}".format(
            semantic_weight=config["semantic_weight"],
            temporal_weight=config["temporal_weight"],
            penalty=config["max_temporal_penalty_seconds"],
            threshold=config["max_acceptable_cost"],
        )
    )

    _print_section("Acoustic Events")
    _print_acoustic_events(acoustic_data)

    _print_section("POS Receipts")
    _print_pos_receipts(pos_data)

    _print_section("Cost Matrix")
    print(
        "Weighted cost = semantic_distance * {:.2f} + normalized_time_distance * {:.2f}".format(
            config["semantic_weight"], config["temporal_weight"]
        )
    )
    print(f"Acceptance threshold: max acceptable cost <= {config['max_acceptable_cost']:.2f}")
    _print_cost_matrix(acoustic_data, pos_data, cost_matrix)

    _print_section("Matched Pairs")
    _print_matches(matched_pairs)

    _print_section("Potential Walk-offs")
    _print_unmatched_audio(unmatched_audio)

    _print_section("Likely Online / Non-counter Orders")
    _print_unmatched_pos(unmatched_pos)

    _print_section("Summary Metrics")
    _print_summary(summary)


if __name__ == "__main__":
    main()

