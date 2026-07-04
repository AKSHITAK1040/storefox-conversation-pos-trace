"""Cart-Acoustic Bipartite Synchronization Engine (CABS Engine).

This proof of concept aligns edge microphone conversations with POS receipts
in a confidence-based way. It avoids relying on perfect clock sync and instead
combines menu semantics, temporal drift tolerance, and global assignment.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate


SEMANTIC_WEIGHT = 0.75
TEMPORAL_WEIGHT = 0.25
MAX_TEMPORAL_PENALTY_SECONDS = 180  # 3 minutes of drift should be a soft penalty only.
MAX_ACCEPTABLE_COST = 0.55


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


def generate_mock_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a realistic 15-minute store window with asynchronous streams."""

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
            normalized_time = min(time_distance / MAX_TEMPORAL_PENALTY_SECONDS, 1.0)
            total_cost = (SEMANTIC_WEIGHT * sem_distance) + (TEMPORAL_WEIGHT * normalized_time)
            cost_matrix[i, j] = total_cost

    return cost_matrix


def match_conversations_to_receipts(
    cost_matrix: np.ndarray,
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use global assignment instead of greedy nearest-time matching.

    Greedy matching can lock in a locally cheap pair and make the remaining
    assignments worse. Hungarian optimization minimizes the total cost across
    the full bipartite graph, which is what we want for noisy retail data.
    """

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: list[dict[str, Any]] = []
    matched_audio_rows: set[int] = set()
    matched_pos_cols: set[int] = set()

    for row_idx, col_idx in zip(row_ind, col_ind):
        cost = float(cost_matrix[row_idx, col_idx])
        if cost <= MAX_ACCEPTABLE_COST:
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


def _fmt_entities(values: Iterable[str]) -> str:
    return ", ".join(values)


def _print_section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def _print_cost_matrix(acoustic_data: list[dict[str, Any]], pos_data: list[dict[str, Any]], cost_matrix: np.ndarray) -> None:
    headers = ["audio\\receipt"] + [receipt["receipt_id"] for receipt in pos_data]
    rows = []
    for row_idx, audio_event in enumerate(acoustic_data):
        rows.append(
            [audio_event["conversation_id"]]
            + [f"{cost_matrix[row_idx, col_idx]:.2f}" for col_idx in range(len(pos_data))]
        )
    print(tabulate(rows, headers=headers, tablefmt="github"))


def _print_acoustic_events(acoustic_data: list[dict[str, Any]]) -> None:
    acoustic_rows = [
        [
            event["conversation_id"],
            event["timestamp"],
            event["speaker_context"],
            _fmt_entities(event["spoken_entities"]),
            event["upsell_offered"] or "-",
            event["expected_intent"],
        ]
        for event in acoustic_data
    ]
    print(tabulate(
        acoustic_rows,
        headers=["conversation_id", "timestamp", "speaker_context", "spoken_entities", "upsell_offered", "expected_intent"],
        tablefmt="github",
    ))


def _print_pos_receipts(pos_data: list[dict[str, Any]]) -> None:
    pos_rows = [
        [
            receipt["receipt_id"],
            receipt["timestamp"],
            _fmt_entities(receipt["billed_skus"]),
            receipt["total_amount"],
            receipt["source"],
        ]
        for receipt in pos_data
    ]

    print(tabulate(
        pos_rows,
        headers=["receipt_id", "timestamp", "billed_skus", "total_amount", "source"],
        tablefmt="github",
    ))


def _print_matches(matched_pairs: list[dict[str, Any]]) -> None:
    rows = [
        [
            pair["audio_id"],
            pair["receipt_id"],
            f"{pair['confidence']:.3f}",
            _fmt_entities(pair["spoken_entities"]),
            _fmt_entities(pair["billed_skus"]),
            pair["upsell_result"],
            pair["revenue_amount"],
        ]
        for pair in matched_pairs
    ]

    print(tabulate(
        rows,
        headers=[
            "audio_id",
            "receipt_id",
            "confidence",
            "spoken_entities",
            "billed_skus",
            "upsell_result",
            "revenue_amount",
        ],
        tablefmt="github",
    ))


def _print_unmatched_audio(unmatched_audio: list[dict[str, Any]]) -> None:
    rows = [
        [
            event["conversation_id"],
            event["timestamp"],
            _fmt_entities(normalize_entities(list(event["spoken_entities"]))),
            event["expected_intent"],
        ]
        for event in unmatched_audio
    ]

    print(tabulate(rows, headers=["conversation_id", "timestamp", "normalized_entities", "expected_intent"], tablefmt="github"))


def _print_unmatched_pos(unmatched_pos: list[dict[str, Any]]) -> None:
    rows = [
        [
            receipt["receipt_id"],
            receipt["timestamp"],
            _fmt_entities(normalize_entities(list(receipt["billed_skus"]))),
            receipt["source"],
        ]
        for receipt in unmatched_pos
    ]

    print(tabulate(rows, headers=["receipt_id", "timestamp", "normalized_skus", "source"], tablefmt="github"))


def _print_summary(
    acoustic_data: list[dict[str, Any]],
    pos_data: list[dict[str, Any]],
    matched_pairs: list[dict[str, Any]],
    unmatched_audio: list[dict[str, Any]],
    unmatched_pos: list[dict[str, Any]],
) -> None:
    walk_offs_detected = sum(1 for event in unmatched_audio if event.get("expected_intent") == "walk_off")
    online_orders_detected = sum(1 for receipt in unmatched_pos if receipt.get("source") == "online")
    upsells_converted = sum(1 for pair in matched_pairs if pair["upsell_result"] == "converted")
    upsells_missed = sum(1 for pair in matched_pairs if pair["upsell_result"] == "missed")
    realized_upsell_revenue = sum(
        pair["revenue_amount"] for pair in matched_pairs if pair["upsell_result"] == "converted"
    )

    rows = [
        ["total conversations", len(acoustic_data)],
        ["total receipts", len(pos_data)],
        ["matched pairs", len(matched_pairs)],
        ["walk-offs detected", walk_offs_detected],
        ["online orders detected", online_orders_detected],
        ["upsells converted", upsells_converted],
        ["upsells missed", upsells_missed],
        ["realized upsell revenue estimate", realized_upsell_revenue],
    ]
    print(tabulate(rows, headers=["metric", "value"], tablefmt="github"))


def main() -> None:
    acoustic_data, pos_data = generate_mock_data()
    vocab = build_entity_vocab(acoustic_data, pos_data)
    cost_matrix = build_cost_matrix(acoustic_data, pos_data, vocab)
    matched_pairs, unmatched_audio, unmatched_pos = match_conversations_to_receipts(cost_matrix, acoustic_data, pos_data)

    print("CABS Engine: Cart-Acoustic Bipartite Synchronization Engine")
    print("Confidence-based alignment of edge audio events to POS receipts")

    _print_section("Acoustic Events")
    _print_acoustic_events(acoustic_data)

    _print_section("POS Receipts")
    _print_pos_receipts(pos_data)

    _print_section("Cost Matrix")
    print(
        "Weighted cost = semantic_distance * {:.2f} + normalized_time_distance * {:.2f}".format(
            SEMANTIC_WEIGHT, TEMPORAL_WEIGHT
        )
    )
    print(f"Acceptance threshold: max acceptable cost <= {MAX_ACCEPTABLE_COST:.2f}")
    _print_cost_matrix(acoustic_data, pos_data, cost_matrix)

    _print_section("Matched Pairs")
    _print_matches(matched_pairs)

    _print_section("Unmatched Audio Events")
    _print_unmatched_audio(unmatched_audio)

    _print_section("Unmatched POS Receipts")
    _print_unmatched_pos(unmatched_pos)

    _print_section("Summary Metrics")
    _print_summary(acoustic_data, pos_data, matched_pairs, unmatched_audio, unmatched_pos)


if __name__ == "__main__":
    main()
