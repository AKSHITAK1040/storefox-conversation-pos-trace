# CABS Engine

Cart-Acoustic Bipartite Synchronization Engine

## Live Demo

Live demo link: `https://your-streamlit-app-url-here`

## Screenshot

Screenshot placeholder: `./assets/cabs-demo-screenshot.png`

## What the Demo Shows

The Streamlit demo presents CABS as a practical retail/QSR intelligence layer, not a classroom assignment. In under two minutes, a viewer can see:

- noisy counter conversations matched to POS receipts
- potential walk-offs
- likely online or non-counter receipt mismatches
- upsell conversion and realized upsell revenue
- why timestamp-only matching is not enough

All demo data is mock data generated locally. The app does not connect to real company systems, POS feeds, microphone streams, or transcripts.

## What This Demo Proves

The main idea is that audio conversations and POS receipts can be aligned with a confidence-based pipeline instead of a brittle timestamp join.

The engine:

1. normalizes menu entities
2. expands bundled SKUs
3. scores semantic similarity with DTW
4. adds a soft timestamp penalty
5. solves the global assignment problem with the Hungarian algorithm
6. rejects weak links when confidence is too low
7. treats unmatched events as useful signals

That makes the output useful for offline retail intelligence, QSR analytics, and operational anomaly detection.

## Repository Layout

- [`app.py`](./app.py): Streamlit demo
- [`cabs_engine.py`](./cabs_engine.py): reusable matching logic
- [`main.py`](./main.py): terminal demo

## Run Locally

### Streamlit demo

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Terminal demo

```bash
python main.py
```

## Mock-Data Disclaimer

This project uses synthetic data only.

The mock data is intentional so the demo can show the product concept without claiming access to real operational systems or customer data.

## Why This Is Useful for Retail/QSR Intelligence

This kind of alignment layer can support:

- receipt-to-conversation reconciliation
- walk-off detection
- online-order mismatch detection
- upsell conversion analysis
- revenue attribution in noisy store environments

It is especially useful when timestamps drift or when bundled items make simple joins unreliable.

## How the Engine Thinks

The demo includes an expandable explanation of the pipeline, but the short version is:

1. normalize the spoken and billed item streams
2. score semantic similarity between the sequences
3. add a soft time penalty for clock drift
4. build a weighted cost matrix
5. use Hungarian assignment to choose the best global alignment
6. leave weak links unmatched on purpose

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Community Cloud app from the repo.
3. Set the main file path to `app.py`.
4. Let Streamlit install dependencies from `requirements.txt`.
5. Deploy.

The app is self-contained, so there are no API keys or external data connections to configure.

## Future Improvements

- connect to real ASR transcripts and POS APIs
- add labeled evaluation data for calibration
- improve confidence scoring with learned embeddings
- add session-level context for tougher matching cases
- persist matches to a database or warehouse
- add export and reporting views for operators

## Docker

The included Dockerfile runs the terminal demo by default.

```bash
docker build -t cabs-engine .
docker run --rm cabs-engine
```

