"""Streamlit demo for the CABS Engine."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from cabs_engine import (
    DEFAULT_MAX_ACCEPTABLE_COST,
    DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_TEMPORAL_WEIGHT,
    SCENARIO_HIGH_CLOCK_DRIFT,
    SCENARIO_MORE_ONLINE_ORDERS,
    SCENARIO_MORE_WALK_OFFS,
    SCENARIO_NORMAL,
    format_entities,
    run_demo,
)


st.set_page_config(
    page_title="CABS Engine",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _pluralize(value: int, singular: str, plural: str | None = None) -> str:
    return singular if value == 1 else (plural or f"{singular}s")


def _currency(value: int) -> str:
    return f"₹{value:,}"


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return "Strong match"
    if confidence >= 0.65:
        return "Review"
    return "Rejected / unmatched"


def _band_style(label: str) -> str:
    if label == "Strong match":
        return "background-color: rgba(16, 185, 129, 0.16); color: #065f46; font-weight: 700;"
    if label == "Review":
        return "background-color: rgba(245, 158, 11, 0.16); color: #92400e; font-weight: 700;"
    return "background-color: rgba(239, 68, 68, 0.16); color: #7f1d1d; font-weight: 700;"


def _confidence_style(value: float) -> str:
    return _band_style(_confidence_band(value))


def _to_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty and not frame.columns.size:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = frame.reindex(columns=columns).copy()

    for column in ("spoken_entities", "billed_skus"):
        if column in frame.columns:
            frame[column] = frame[column].apply(
                lambda values: format_entities(values) if isinstance(values, list) else (values or "")
            )
    return frame


def _df_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _render_metric_grid(metrics: list[tuple[str, Any]], columns_per_row: int = 4) -> None:
    for start in range(0, len(metrics), columns_per_row):
        row = metrics[start : start + columns_per_row]
        cols = st.columns(len(row))
        for column, (label, value) in zip(cols, row):
            with column:
                st.metric(label, value)


@st.cache_data(show_spinner=False)
def load_demo(
    scenario: str,
    semantic_weight: float,
    temporal_weight: float,
    max_temporal_penalty_seconds: int,
    max_acceptable_cost: float,
) -> dict[str, Any]:
    return run_demo(
        scenario=scenario,
        semantic_weight=semantic_weight,
        temporal_weight=temporal_weight,
        max_temporal_penalty_seconds=max_temporal_penalty_seconds,
        max_acceptable_cost=max_acceptable_cost,
    )


def _render_result_card(summary: dict[str, int]) -> None:
    result_text = (
        f"In this mock 15-minute store window, CABS matched {summary['matched_pairs']} audio conversations to receipts, "
        f"detected {summary['potential_walk_offs']} likely {_pluralize(summary['potential_walk_offs'], 'walk-off')}, "
        f"identified {summary['likely_online_orders']} likely online/non-counter {_pluralize(summary['likely_online_orders'], 'order')}, "
        f"and estimated {_currency(summary['realized_upsell_revenue'])} in realized upsell revenue."
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Demo result</div>
            <div class="result-text">{result_text}</div>
            <div class="result-pills">
                <span class="pill">Matched pairs: {summary["matched_pairs"]}</span>
                <span class="pill">Walk-offs: {summary["potential_walk_offs"]}</span>
                <span class="pill">Online orders: {summary["likely_online_orders"]}</span>
                <span class="pill">Upsell revenue: {_currency(summary["realized_upsell_revenue"])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_matches(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    styled = frame.style.format(
        {
            "confidence": "{:.0%}",
            "revenue_amount": "{:,.0f}",
        }
    )
    styled = styled.applymap(_confidence_style, subset=["confidence"])
    styled = styled.applymap(lambda value: _band_style(str(value)), subset=["match_band"])
    return styled


def _download_row(label: str, frame: pd.DataFrame, file_name: str) -> None:
    st.download_button(
        label,
        data=_df_to_csv_bytes(frame),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #111827 58%, #1f2937 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.4rem 1.5rem 1.2rem 1.5rem;
            color: #f8fafc;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
        }
        .hero .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #7dd3fc;
            font-size: 0.72rem;
            margin-bottom: 0.65rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.45rem;
            line-height: 1.05;
        }
        .hero .subtitle {
            margin: 0.35rem 0 0.8rem 0;
            font-size: 1.05rem;
            color: #cbd5e1;
        }
        .hero .summary {
            margin: 0;
            color: #e2e8f0;
            max-width: 60rem;
        }
        .result-card {
            margin-top: 0.9rem;
            margin-bottom: 0.9rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.98));
            padding: 1rem 1.1rem 0.95rem 1.1rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }
        .result-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #2563eb;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .result-text {
            color: #0f172a;
            font-size: 1.03rem;
            line-height: 1.55;
        }
        .result-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }
        .pill {
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid rgba(37, 99, 235, 0.16);
            color: #1e3a8a;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.8rem;
        }
        .subtle-note {
            color: #475569;
            margin-top: 0.35rem;
            margin-bottom: 0.2rem;
        }
        .comparison-example {
            border-left: 3px solid #2563eb;
            padding-left: 0.9rem;
            color: #0f172a;
            margin: 0.25rem 0 0.25rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Independent mock-data prototype</div>
            <h1>CABS Engine</h1>
            <p class="subtitle">Cart-Acoustic Bipartite Synchronization Engine</p>
            <p class="summary">
                Match noisy counter conversations to receipts, detect walk-offs, flag online-order mismatches,
                and estimate upsell conversion without assuming perfect clock sync.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "This is an independent mock-data prototype. It does not connect to any real company data, POS system, or microphone feed."
    )

    with st.sidebar:
        st.header("Engine controls")
        scenario = st.selectbox(
            "Scenario",
            [SCENARIO_NORMAL, SCENARIO_HIGH_CLOCK_DRIFT, SCENARIO_MORE_WALK_OFFS, SCENARIO_MORE_ONLINE_ORDERS],
            index=0,
            help="Scenario changes are deterministic and only adjust the mock data.",
        )
        semantic_weight = st.slider(
            "Semantic weight",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_SEMANTIC_WEIGHT,
            step=0.05,
        )
        temporal_weight = st.slider(
            "Temporal weight",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPORAL_WEIGHT,
            step=0.05,
        )
        max_acceptable_cost = st.slider(
            "Max acceptable cost",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_MAX_ACCEPTABLE_COST,
            step=0.01,
        )
        max_temporal_penalty_seconds = st.slider(
            "Max temporal penalty seconds",
            min_value=30,
            max_value=600,
            value=DEFAULT_MAX_TEMPORAL_PENALTY_SECONDS,
            step=15,
        )
        st.caption("Lower cost means a stronger possible match. Weak pairs stay unmatched on purpose.")

    demo = load_demo(
        scenario=scenario,
        semantic_weight=semantic_weight,
        temporal_weight=temporal_weight,
        max_temporal_penalty_seconds=max_temporal_penalty_seconds,
        max_acceptable_cost=max_acceptable_cost,
    )

    acoustic_data = demo["acoustic_data"]
    pos_data = demo["pos_data"]
    cost_matrix = demo["cost_matrix"]
    matched_pairs = demo["matched_pairs"]
    unmatched_audio = demo["unmatched_audio"]
    unmatched_pos = demo["unmatched_pos"]
    summary = demo["summary"]
    baseline = demo["baseline"]

    _render_result_card(summary)

    st.subheader("Problem")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("Timestamp-only joins are brittle when the store floor is messy.")
        st.markdown(
            """
            - POS receipts may be created after the conversation ends
            - microphone and POS clocks can drift
            - bundled items may appear in different forms
            - some audio events are walk-offs with no receipt
            - some receipts are online or non-counter orders
            """
        )
    with right:
        st.info(
            "The engine combines semantic similarity, a soft time penalty, and global assignment so the full set of matches is better than a greedy timestamp join."
        )

    st.divider()
    st.subheader("Input streams")
    acoustic_frame = _to_frame(
        acoustic_data,
        ["conversation_id", "timestamp", "speaker_context", "spoken_entities", "upsell_offered", "expected_intent"],
    )
    pos_frame = _to_frame(pos_data, ["receipt_id", "timestamp", "billed_skus", "total_amount", "source"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Acoustic Events**")
        st.dataframe(acoustic_frame, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**POS Receipts**")
        st.dataframe(pos_frame, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Matching results")
    st.caption("High confidence = strong match, medium confidence = review, weak links are left unmatched.")

    matched_rows: list[dict[str, Any]] = []
    for pair in matched_pairs:
        confidence = float(pair["confidence"])
        matched_rows.append(
            {
                "audio_id": pair["audio_id"],
                "receipt_id": pair["receipt_id"],
                "confidence": confidence,
                "match_band": _confidence_band(confidence),
                "spoken_entities": format_entities(pair["spoken_entities"]),
                "billed_skus": format_entities(pair["billed_skus"]),
                "upsell_result": str(pair["upsell_result"]).replace("_", " ").title(),
                "revenue_amount": int(pair["revenue_amount"]),
                "explanation": pair["explanation"],
            }
        )
    matched_frame = pd.DataFrame(
        matched_rows,
        columns=[
            "audio_id",
            "receipt_id",
            "confidence",
            "match_band",
            "spoken_entities",
            "billed_skus",
            "upsell_result",
            "revenue_amount",
            "explanation",
        ],
    )

    if matched_frame.empty:
        st.warning("No matches passed the current threshold. Try lowering max acceptable cost or increasing semantic weight.")
    else:
        st.dataframe(_style_matches(matched_frame), use_container_width=True, hide_index=True)

    download_left, download_middle, download_right = st.columns(3)
    with download_left:
        _download_row("Download matched pairs CSV", matched_frame, "cabs_matched_pairs.csv")
    with download_middle:
        unmatched_audio_frame = _to_frame(
            unmatched_audio,
            ["conversation_id", "timestamp", "spoken_entities", "expected_intent"],
        )
        _download_row("Download unmatched audio CSV", unmatched_audio_frame, "cabs_unmatched_audio.csv")
    with download_right:
        unmatched_pos_frame = _to_frame(
            unmatched_pos,
            ["receipt_id", "timestamp", "billed_skus", "source"],
        )
        _download_row("Download unmatched receipts CSV", unmatched_pos_frame, "cabs_unmatched_receipts.csv")

    st.divider()
    st.subheader("Why not just nearest timestamp?")

    comparison_rows = baseline["comparison_rows"]
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame = comparison_frame[
        ["audio_id", "baseline_receipt_id", "cabs_receipt_id", "baseline_drift_seconds", "baseline_confidence", "cabs_confidence", "result", "explanation"]
    ]
    avoided_rows = comparison_frame[comparison_frame["result"] == "Avoided"].copy()

    _render_metric_grid(
        [
            ("Timestamp-only matched pairs", baseline["baseline_matched_count"]),
            ("CABS matched pairs", summary["matched_pairs"]),
            ("Weak/wrong matches avoided", baseline["avoided_matches"]),
            ("Potential walk-offs detected", summary["potential_walk_offs"]),
            ("Online receipts detected", summary["likely_online_orders"]),
        ]
    )

    st.markdown("Timestamp-only matching can look reasonable until receipt timing or bundles distort the order of events.")
    if avoided_rows.empty:
        st.info("In this scenario, CABS and the timestamp baseline agree on the visible matches, but CABS still keeps weak links out of the final set.")
    else:
        st.dataframe(
            avoided_rows[["audio_id", "baseline_receipt_id", "cabs_receipt_id", "baseline_drift_seconds", "explanation"]],
            use_container_width=True,
            hide_index=True,
        )
        examples = avoided_rows.head(2)
        for _, row in examples.iterrows():
            st.markdown(
                f"<div class='comparison-example'><strong>{row['audio_id']}</strong>: {row['explanation']} "
                f"(timestamp-only picked {row['baseline_receipt_id']}, CABS picked {row['cabs_receipt_id']}).</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Examples like these usually come from delayed receipts, clock drift, or bundled SKUs that are easier to interpret when semantics are part of the score."
        )

    if scenario == SCENARIO_HIGH_CLOCK_DRIFT:
        st.markdown(
            "- This scenario shifts POS timestamps later by about 2.5 minutes, so timestamp-only matching gets less trustworthy even though the item semantics stay the same."
        )
    elif scenario == SCENARIO_MORE_WALK_OFFS:
        st.markdown("- This scenario adds extra audio events with no receipts, which makes walk-off detection more visible.")
    elif scenario == SCENARIO_MORE_ONLINE_ORDERS:
        st.markdown("- This scenario adds extra online receipts with no matching audio, which helps show non-counter order detection.")

    st.divider()
    st.subheader("Unmatched signals")
    walkoff_col, online_col = st.columns(2)

    unmatched_audio_frame = _to_frame(
        unmatched_audio,
        ["conversation_id", "timestamp", "spoken_entities", "expected_intent"],
    )
    unmatched_pos_frame = _to_frame(
        unmatched_pos,
        ["receipt_id", "timestamp", "billed_skus", "source"],
    )

    with walkoff_col:
        st.markdown("**Potential walk-offs**")
        if unmatched_audio_frame.empty:
            st.info("No potential walk-offs surfaced in this scenario.")
        else:
            st.dataframe(unmatched_audio_frame, use_container_width=True, hide_index=True)
    with online_col:
        st.markdown("**Likely online / non-counter orders**")
        if unmatched_pos_frame.empty:
            st.info("No likely online or non-counter orders surfaced in this scenario.")
        else:
            st.dataframe(unmatched_pos_frame, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Upsell intelligence")
    st.markdown(
        "This is not final revenue attribution; it shows how aligned audio and receipt streams can support upsell conversion analysis."
    )
    _render_metric_grid(
        [
            ("Upsells offered", summary["upsells_offered"]),
            ("Upsells converted", summary["upsells_converted"]),
            ("Upsells missed", summary["upsells_missed"]),
            ("Realized upsell revenue", _currency(summary["realized_upsell_revenue"])),
        ]
    )

    st.divider()
    st.subheader("Summary metrics")
    _render_metric_grid(
        [
            ("Total conversations", summary["total_conversations"]),
            ("Total receipts", summary["total_receipts"]),
            ("Matched pairs", summary["matched_pairs"]),
            ("Potential walk-offs", summary["potential_walk_offs"]),
            ("Likely online orders", summary["likely_online_orders"]),
            ("Upsells converted", summary["upsells_converted"]),
            ("Upsells missed", summary["upsells_missed"]),
            ("Realized upsell revenue", _currency(summary["realized_upsell_revenue"])),
        ]
    )

    st.divider()
    st.subheader("Cost matrix")
    st.caption(
        "Lower cost means a stronger possible match. The Hungarian algorithm chooses the globally best assignment, instead of greedily matching nearest timestamps."
    )
    with st.expander("Show weighted cost matrix", expanded=False):
        cost_frame = pd.DataFrame(
            cost_matrix,
            index=[event["conversation_id"] for event in acoustic_data],
            columns=[receipt["receipt_id"] for receipt in pos_data],
        ).round(2)
        st.dataframe(cost_frame, use_container_width=True)

    with st.expander("How the engine thinks", expanded=False):
        st.markdown(
            """
            - Normalize menu entities
            - Expand bundled SKUs
            - Compare item sequences with DTW
            - Add soft timestamp penalty
            - Build weighted cost matrix
            - Run Hungarian global assignment
            - Reject weak links using confidence threshold
            - Treat unmatched nodes as useful signals
            """
        )

    st.divider()
    st.subheader("Architecture")
    st.code(
        """
Audio transcripts        POS receipts
|                        |
Entity normalization     SKU expansion
|                        |
Semantic + temporal scoring -> Cost matrix
           |
    Hungarian assignment
           |
   Confidence threshold
           |
 Matches + walk-offs + online orders + upsell metrics
        """.strip(),
        language="text",
    )

    st.divider()
    st.subheader("What this demo proves")
    st.markdown(
        """
        - This is not claiming perfect matching.
        - It shows a confidence-based alignment layer for noisy retail data.
        - Weak links are intentionally left unmatched instead of forcing a bad join.
        - The same pattern can later connect to real ASR transcripts and POS APIs.
        """
    )


if __name__ == "__main__":
    main()
