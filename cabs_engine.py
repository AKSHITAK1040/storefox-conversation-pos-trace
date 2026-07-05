"""Core logic for the CABS Engine demo.

This module keeps the matching logic separate from any user interface so the
same engine can drive both the terminal demo and the Streamlit prototype.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_SEMANTIC_WEIGHT = 0.75
DEFAULT_TEMPORAL_WEIGHT = 0.25
DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS = 180
DEFAULT_MAX_ACCEPTABLE_COST = 0.55

SCENARIO_NORMAL = "Normal store window"
SCENARIO_HIGH_CLOCK_DRIFT = "High clock drift"
SCENARIO_MORE_WALK_OFFS = "More walk-offs"
SCENARIO_MORE_ONLINE_ORDERS = "More online orders"

ENTITY_EXPANSIONS: dict[str, list[str]] = {
    "burger_combo_meal": ["burger", "fries", "coke"],
    "combo_upgrade": ["fries", "coke"],
    "cold_drink": ["coke"],
    "momos_combo": ["momos", "drink"],
    "large_fries": ["fries"],
}

UPSELL_EXPECTATIONS: dict[str, list[str]] = {
    "combo_upgrade": ["fries", "coke"],
}


def _clean_entity_name(item: str) -> str:
    """Normalize menu strings to a deterministic token form."""

    cleaned = re.sub(r"[^a-z0-9]+", "_", item.strip().lower())
    return cleaned.strip("_")


def normalize_entities(items: list[str]) -> list[str]:
    """Expand bundled SKUs and clean item names before matching."""

    normalized: list[str] = []
    for raw_item in items:
        item = _clean_entity_name(raw_item)
        expanded = ENTITY_EXPANSIONS.get(item)
        if expanded is not None:
            normalized.extend(expanded)
        elif item:
            normalized.append(item)
    return normalized


def format_entities(values: Iterable[str]) -> str:
    """Render a compact comma-separated list for tables and terminal output."""

    return ", ".join(values)


def _shift_timestamp(timestamp: str, seconds: int) -> str:
    """Shift an HH:MM:SS timestamp by a signed number of seconds."""

    parsed = datetime.strptime(timestamp, "%H:%M:%S")
    shifted = parsed + timedelta(seconds=seconds)
    return shifted.strftime("%H:%M:%S")


def _base_mock_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the default 15-minute store window with asynchronous streams."""

    base = datetime(2026, 7, 4, 12, 0, 0)

    def ts(minutes: int = 0, seconds: int = 0) -> str:
        return (base + timedelta(seconds=minutes * 60 + seconds)).strftime("%H:%M:%S")

    acoustic_data = [
        {
            "conversation_id": "AUDIO_001",
            "timestamp": ts(seconds=20),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["burger", "fries", "coke"],
            "upsell_offered": None,
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_002",
            "timestamp": ts(minutes=1, seconds=40),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["pizza", "large_fries", "cold_drink"],
            "upsell_offered": None,
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_003",
            "timestamp": ts(minutes=3),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["momos", "drink"],
            "upsell_offered": None,
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_004",
            "timestamp": ts(minutes=4, seconds=20),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["sandwich", "cold_drink"],
            "upsell_offered": None,
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_005",
            "timestamp": ts(minutes=5, seconds=40),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["wrap", "fries"],
            "upsell_offered": None,
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_006",
            "timestamp": ts(minutes=6, seconds=50),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["burger", "coke", "fries"],
            "upsell_offered": None,
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_007",
            "timestamp": ts(minutes=8, seconds=10),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["coffee"],
            "upsell_offered": None,
            "expected_intent": "walk_off",
        },
        {
            "conversation_id": "AUDIO_008",
            "timestamp": ts(minutes=10, seconds=20),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["burger", "combo_upgrade"],
            "upsell_offered": "combo_upgrade",
            "expected_intent": "purchase",
        },
        {
            "conversation_id": "AUDIO_009",
            "timestamp": ts(minutes=12, seconds=20),
            "speaker_context": "cashier_customer",
            "spoken_entities": ["burger"],
            "upsell_offered": "combo_upgrade",
            "expected_intent": "purchase",
        },
    ]

    pos_data = [
        {
            "receipt_id": "POS_001",
            "timestamp": ts(minutes=1, seconds=50),
            "billed_skus": ["burger", "fries", "coke"],
            "total_amount": 249,
            "source": "counter",
        },
        {
            "receipt_id": "POS_002",
            "timestamp": ts(minutes=3, seconds=10),
            "billed_skus": ["pizza", "fries", "coke"],
            "total_amount": 219,
            "source": "counter",
        },
        {
            "receipt_id": "POS_003",
            "timestamp": ts(minutes=4, seconds=20),
            "billed_skus": ["momos_combo"],
            "total_amount": 189,
            "source": "counter",
        },
        {
            "receipt_id": "POS_004",
            "timestamp": ts(minutes=5, seconds=40),
            "billed_skus": ["sandwich", "coke"],
            "total_amount": 179,
            "source": "counter",
        },
        {
            "receipt_id": "POS_005",
            "timestamp": ts(minutes=7),
            "billed_skus": ["wrap", "large_fries"],
            "total_amount": 159,
            "source": "counter",
        },
        {
            "receipt_id": "POS_006",
            "timestamp": ts(minutes=8, seconds=20),
            "billed_skus": ["burger", "fries", "coke"],
            "total_amount": 249,
            "source": "counter",
        },
        {
            "receipt_id": "POS_007",
            "timestamp": ts(minutes=12),
            "billed_skus": ["burger_combo_meal"],
            "total_amount": 279,
            "source": "counter",
        },
        {
            "receipt_id": "POS_008",
            "timestamp": ts(minutes=13, seconds=50),
            "billed_skus": ["burger"],
            "total_amount": 149,
            "source": "counter",
        },
        {
            "receipt_id": "POS_009",
            "timestamp": ts(minutes=14, seconds=10),
            "billed_skus": ["ramen", "bubble_tea"],
            "total_amount": 399,
            "source": "online",
        },
    ]

    return acoustic_data, pos_data


def _apply_scenario_modifiers(
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically adjust the base data for the selected demo scenario."""

    acoustic_copy = [dict(event) for event in acoustic_data]
    pos_copy = [dict(receipt) for receipt in pos_data]

    if scenario == SCENARIO_HIGH_CLOCK_DRIFT:
        for receipt in pos_copy:
            receipt["timestamp"] = _shift_timestamp(str(receipt["timestamp"]), 150)
        return acoustic_copy, pos_copy

    if scenario == SCENARIO_MORE_WALK_OFFS:
        acoustic_copy.extend(
            [
                {
                    "conversation_id": "AUDIO_010",
                    "timestamp": "12:15:00",
                    "speaker_context": "cashier_customer",
                    "spoken_entities": ["iced_tea"],
                    "upsell_offered": None,
                    "expected_intent": "walk_off",
                },
                {
                    "conversation_id": "AUDIO_011",
                    "timestamp": "12:16:20",
                    "speaker_context": "cashier_customer",
                    "spoken_entities": ["snack_box"],
                    "upsell_offered": None,
                    "expected_intent": "walk_off",
                },
            ]
        )
        return acoustic_copy, pos_copy

    if scenario == SCENARIO_MORE_ONLINE_ORDERS:
        pos_copy.extend(
            [
                {
                    "receipt_id": "POS_010",
                    "timestamp": "12:15:40",
                    "billed_skus": ["noodles", "cold_drink"],
                    "total_amount": 329,
                    "source": "online",
                },
                {
                    "receipt_id": "POS_011",
                    "timestamp": "12:16:50",
                    "billed_skus": ["salad_bowl", "sparkling_water"],
                    "total_amount": 289,
                    "source": "online",
                },
            ]
        )
        return acoustic_copy, pos_copy

    return acoustic_copy, pos_copy


def generate_mock_data(scenario: str = SCENARIO_NORMAL) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a realistic 15-minute store window with asynchronous streams."""

    acoustic_data, pos_data = _base_mock_data()
    return _apply_scenario_modifiers(acoustic_data, pos_data, scenario)


def build_entity_vocab(acoustic_data: list[dict[str, Any]], pos_data: list[dict[str, Any]]) -> dict[str, int]:
    """Build a stable token-to-id mapping from all observed menu entities."""

    unique_entities: set[str] = set()
    for event in acoustic_data:
        unique_entities.update(normalize_entities(list(event["spoken_entities"])))
    for receipt in pos_data:
        unique_entities.update(normalize_entities(list(receipt["billed_skus"])))

    vocab: dict[str, int] = {"<UNK>": 0}
    for idx, entity in enumerate(sorted(unique_entities), start=1):
        vocab[entity] = idx
    return vocab


def encode_sequence(items: Iterable[str], vocab: dict[str, int]) -> list[int]:
    """Convert a menu entity sequence into integer ids."""

    normalized = normalize_entities(list(items))
    return [vocab.get(item, vocab["<UNK>"]) for item in normalized]


def _dtw_distance(seq_a: list[int], seq_b: list[int]) -> float:
    """Simple DTW with unit substitution cost and insertion/deletion cost."""

    if not seq_a and not seq_b:
        return 0.0
    if not seq_a or not seq_b:
        return float(max(len(seq_a), len(seq_b)))

    n, m = len(seq_a), len(seq_b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = 0.0 if seq_a[i - 1] == seq_b[j - 1] else 1.0
            dp[i, j] = local_cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m])


def semantic_distance(audio_entities: list[str], receipt_skus: list[str], vocab: dict[str, int]) -> float:
    """Compare normalized entity sequences with DTW and a reorder-tolerant fallback."""

    audio_encoded = encode_sequence(audio_entities, vocab)
    receipt_encoded = encode_sequence(receipt_skus, vocab)

    raw_distance = _dtw_distance(audio_encoded, receipt_encoded)
    raw_distance /= max(len(audio_encoded), len(receipt_encoded), 1)

    # DTW handles local stretch/compression. A canonical fallback softens mild
    # ASR / SKU reorder noise without pretending order never matters.
    canonical_distance = _dtw_distance(sorted(audio_encoded), sorted(receipt_encoded))
    canonical_distance /= max(len(audio_encoded), len(receipt_encoded), 1)

    return float(min(raw_distance, canonical_distance))


def time_to_seconds(timestamp: str) -> int:
    """Convert HH:MM:SS timestamps to absolute seconds since midnight."""

    parsed = datetime.strptime(timestamp, "%H:%M:%S")
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def time_distance_seconds(audio_timestamp: str, pos_timestamp: str) -> int:
    """Return absolute timestamp drift in seconds."""

    return abs(time_to_seconds(audio_timestamp) - time_to_seconds(pos_timestamp))


def build_cost_matrix(
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    vocab: dict[str, int],
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    max_temporal_penalty_seconds: int = DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS,
) -> np.ndarray:
    """Build a weighted cost matrix across the two event streams."""

    cost_matrix = np.zeros((len(acoustic_data), len(pos_data)), dtype=float)

    for i, audio_event in enumerate(acoustic_data):
        audio_entities = list(audio_event["spoken_entities"])
        for j, receipt in enumerate(pos_data):
            receipt_items = list(receipt["billed_skus"])
            # Timestamp-only matching fails when the edge device and POS clock drift.
            sem_distance = semantic_distance(audio_entities, receipt_items, vocab)
            time_distance = time_distance_seconds(audio_event["timestamp"], receipt["timestamp"])
            normalized_time = min(time_distance / max_temporal_penalty_seconds, 1.0)
            total_cost = (semantic_weight * sem_distance) + (temporal_weight * normalized_time)
            cost_matrix[i, j] = total_cost

    return cost_matrix


def detect_upsell_result(audio_event: dict[str, Any], pos_receipt: dict[str, Any]) -> str:
    """Classify whether an offered upsell was converted, missed, or not applicable."""

    upsell_offer = audio_event.get("upsell_offered")
    if not upsell_offer:
        return "not_applicable"

    expected_items = UPSELL_EXPECTATIONS.get(str(upsell_offer), [])
    if not expected_items:
        return "not_applicable"

    receipt_items = set(normalize_entities(list(pos_receipt["billed_skus"])))
    if set(expected_items).issubset(receipt_items):
        return "converted"
    return "missed"


def build_match_explanation(
    audio_event: dict[str, Any],
    pos_receipt: dict[str, Any],
    confidence: float,
) -> str:
    """Generate a short human-readable explanation for a matched pair."""

    upsell_result = detect_upsell_result(audio_event, pos_receipt)
    raw_receipt_items = [_clean_entity_name(item) for item in list(pos_receipt["billed_skus"])]
    bundle_expanded = any(item in ENTITY_EXPANSIONS for item in raw_receipt_items)

    if upsell_result == "converted":
        return "Upsell offer converted because expected combo items appeared in the receipt."
    if bundle_expanded:
        return "Bundled SKU expanded before matching, which improved the semantic alignment."
    if confidence >= 0.8:
        return "Matched because menu entities align strongly despite timestamp drift."
    if confidence >= 0.65:
        return "Needs review because confidence is moderate."
    return "Matched conservatively, but the link is weak and should be reviewed."


def match_conversations_to_receipts(
    cost_matrix: np.ndarray,
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    max_acceptable_cost: float = DEFAULT_MAX_ACCEPTABLE_COST,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use global assignment instead of greedy nearest-time matching."""

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: list[dict[str, Any]] = []
    matched_audio_rows: set[int] = set()
    matched_pos_cols: set[int] = set()

    for row_idx, col_idx in zip(row_ind, col_ind):
        cost = float(cost_matrix[row_idx, col_idx])
        if cost <= max_acceptable_cost:
            audio_event = acoustic_data[row_idx]
            receipt = pos_data[col_idx]
            matched_pairs.append(
                {
                    "audio_index": row_idx,
                    "receipt_index": col_idx,
                    "audio_id": audio_event["conversation_id"],
                    "receipt_id": receipt["receipt_id"],
                    "cost": cost,
                    "confidence": round(max(0.0, 1.0 - cost), 3),
                    "spoken_entities": normalize_entities(list(audio_event["spoken_entities"])),
                    "billed_skus": normalize_entities(list(receipt["billed_skus"])),
                    "upsell_result": detect_upsell_result(audio_event, receipt),
                    "revenue_amount": receipt["total_amount"],
                }
            )
            matched_audio_rows.add(row_idx)
            matched_pos_cols.add(col_idx)

    # Unmatched nodes are first-class signals: audio gaps often mean walk-offs,
    # and receipt-only nodes often mean online or otherwise non-counter orders.
    unmatched_audio = [event for idx, event in enumerate(acoustic_data) if idx not in matched_audio_rows]
    unmatched_pos = [receipt for idx, receipt in enumerate(pos_data) if idx not in matched_pos_cols]

    return matched_pairs, unmatched_audio, unmatched_pos


def build_timestamp_only_baseline(
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    max_temporal_penalty_seconds: int = DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS,
) -> list[dict[str, Any]]:
    """Match each audio event to the nearest receipt by timestamp only."""

    baseline_rows: list[dict[str, Any]] = []
    for audio_idx, audio_event in enumerate(acoustic_data):
        closest_idx = min(
            range(len(pos_data)),
            key=lambda idx: time_distance_seconds(audio_event["timestamp"], pos_data[idx]["timestamp"]),
        )
        receipt = pos_data[closest_idx]
        drift_seconds = time_distance_seconds(audio_event["timestamp"], receipt["timestamp"])
        baseline_rows.append(
            {
                "audio_index": audio_idx,
                "audio_id": audio_event["conversation_id"],
                "receipt_index": closest_idx,
                "receipt_id": receipt["receipt_id"],
                "drift_seconds": drift_seconds,
                "normalized_time_distance": round(min(drift_seconds / max_temporal_penalty_seconds, 1.0), 3),
                "timestamp_only_confidence": round(max(0.0, 1.0 - min(drift_seconds / max_temporal_penalty_seconds, 1.0)), 3),
                "timestamp_only_reason": "Nearest receipt by clock time only.",
            }
        )
    return baseline_rows


def compare_with_timestamp_baseline(
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    matched_pairs: list[dict[str, Any]],
    unmatched_audio: list[dict[str, Any]],
    unmatched_pos: list[dict[str, Any]],
    max_temporal_penalty_seconds: int = DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS,
) -> dict[str, Any]:
    """Compare CABS output against a timestamp-only nearest-receipt baseline."""

    baseline_rows = build_timestamp_only_baseline(
        acoustic_data,
        pos_data,
        max_temporal_penalty_seconds=max_temporal_penalty_seconds,
    )
    cabs_by_audio_id = {pair["audio_id"]: pair for pair in matched_pairs}
    baseline_by_audio_id = {row["audio_id"]: row for row in baseline_rows}

    comparison_rows: list[dict[str, Any]] = []
    avoided_matches = 0
    for audio_event in acoustic_data:
        audio_id = audio_event["conversation_id"]
        baseline_row = baseline_by_audio_id[audio_id]
        baseline_receipt = pos_data[baseline_row["receipt_index"]]
        cabs_match = cabs_by_audio_id.get(audio_id)
        cabs_receipt_id = cabs_match["receipt_id"] if cabs_match is not None else None
        avoided = cabs_receipt_id != baseline_row["receipt_id"]
        if avoided:
            avoided_matches += 1

        explanation = "Timestamp-only and CABS agree."
        if avoided:
            if cabs_match is None:
                explanation = "CABS left this event unmatched while timestamp-only would have forced a pair."
            else:
                baseline_upsell = detect_upsell_result(audio_event, baseline_receipt)
                baseline_raw_items = [_clean_entity_name(item) for item in list(baseline_receipt["billed_skus"])]
                bundle_expanded = any(item in ENTITY_EXPANSIONS for item in baseline_raw_items)
                if baseline_upsell == "converted" and cabs_match["upsell_result"] != "converted":
                    explanation = (
                        "Timestamp-only would have overstated upsell conversion, but CABS kept the tighter semantic match."
                    )
                elif baseline_row["drift_seconds"] >= 120:
                    explanation = "Timestamp-only was dragged by clock drift, while CABS leaned on the item sequence."
                elif bundle_expanded:
                    explanation = "Timestamp-only landed on a bundled receipt, but CABS expanded the SKU and matched the underlying items."
                else:
                    explanation = "Nearest timestamp picked a different receipt, but CABS used semantics plus assignment to choose better."

        comparison_rows.append(
            {
                "audio_id": audio_id,
                "baseline_receipt_id": baseline_row["receipt_id"],
                "cabs_receipt_id": cabs_receipt_id or "-",
                "baseline_drift_seconds": baseline_row["drift_seconds"],
                "baseline_confidence": baseline_row["timestamp_only_confidence"],
                "cabs_confidence": round(cabs_match["confidence"], 3) if cabs_match is not None else 0.0,
                "result": "Avoided" if avoided else "Same",
                "explanation": explanation,
            }
        )

    baseline_matched_count = len(baseline_rows)

    return {
        "baseline_rows": baseline_rows,
        "comparison_rows": comparison_rows,
        "baseline_matched_count": baseline_matched_count,
        "avoided_matches": avoided_matches,
        "baseline_walk_off_like_events": sum(
            1 for event in unmatched_audio if event.get("expected_intent") == "walk_off"
        ),
        "baseline_online_order_like_receipts": sum(
            1 for receipt in unmatched_pos if receipt.get("source") == "online"
        ),
    }


def summarize_results(
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    matched_pairs: list[dict[str, Any]],
    unmatched_audio: list[dict[str, Any]],
    unmatched_pos: list[dict[str, Any]],
) -> dict[str, int]:
    """Compute the business-facing summary metrics for the demo."""

    walk_offs_detected = sum(1 for event in unmatched_audio if event.get("expected_intent") == "walk_off")
    online_orders_detected = sum(1 for receipt in unmatched_pos if receipt.get("source") == "online")
    upsells_offered = sum(1 for event in acoustic_data if event.get("upsell_offered"))
    upsells_converted = sum(1 for pair in matched_pairs if pair["upsell_result"] == "converted")
    upsells_missed = sum(1 for pair in matched_pairs if pair["upsell_result"] == "missed")
    realized_upsell_revenue = sum(
        pair["revenue_amount"] for pair in matched_pairs if pair["upsell_result"] == "converted"
    )

    return {
        "total_conversations": len(acoustic_data),
        "total_receipts": len(pos_data),
        "matched_pairs": len(matched_pairs),
        "potential_walk_offs": walk_offs_detected,
        "likely_online_orders": online_orders_detected,
        "upsells_offered": upsells_offered,
        "upsells_converted": upsells_converted,
        "upsells_missed": upsells_missed,
        "realized_upsell_revenue": realized_upsell_revenue,
    }


def run_demo(
    scenario: str = SCENARIO_NORMAL,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    max_temporal_penalty_seconds: int = DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS,
    max_acceptable_cost: float = DEFAULT_MAX_ACCEPTABLE_COST,
) -> dict[str, Any]:
    """Generate mock data, score matches, and return a complete demo payload."""

    acoustic_data, pos_data = generate_mock_data(scenario=scenario)
    vocab = build_entity_vocab(acoustic_data, pos_data)
    cost_matrix = build_cost_matrix(
        acoustic_data,
        pos_data,
        vocab,
        semantic_weight=semantic_weight,
        temporal_weight=temporal_weight,
        max_temporal_penalty_seconds=max_temporal_penalty_seconds,
    )
    matched_pairs, unmatched_audio, unmatched_pos = match_conversations_to_receipts(
        cost_matrix,
        acoustic_data,
        pos_data,
        max_acceptable_cost=max_acceptable_cost,
    )
    summary = summarize_results(acoustic_data, pos_data, matched_pairs, unmatched_audio, unmatched_pos)
    baseline = compare_with_timestamp_baseline(
        acoustic_data,
        pos_data,
        matched_pairs,
        unmatched_audio,
        unmatched_pos,
        max_temporal_penalty_seconds=max_temporal_penalty_seconds,
    )

    for pair in matched_pairs:
        audio_event = acoustic_data[pair["audio_index"]]
        receipt = pos_data[pair["receipt_index"]]
        pair["explanation"] = build_match_explanation(audio_event, receipt, float(pair["confidence"]))

    return {
        "acoustic_data": acoustic_data,
        "pos_data": pos_data,
        "vocab": vocab,
        "cost_matrix": cost_matrix,
        "matched_pairs": matched_pairs,
        "unmatched_audio": unmatched_audio,
        "unmatched_pos": unmatched_pos,
        "summary": summary,
        "baseline": baseline,
        "config": {
            "scenario": scenario,
            "semantic_weight": semantic_weight,
            "temporal_weight": temporal_weight,
            "max_temporal_penalty_seconds": max_temporal_penalty_seconds,
            "max_acceptable_cost": max_acceptable_cost,
        },
    }
