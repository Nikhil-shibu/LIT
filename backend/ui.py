import logging
import sys
import time

import streamlit as st
import requests
import pandas as pd

# ── Logging setup (prints to the terminal running `streamlit run ui.py`) ────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | ui | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ui")

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide", page_title="TruthLens")
st.title("TruthLens — Fake News Detector")

col_input, col_results = st.columns([1, 1])

with col_input:
    source_type_display = st.radio(
        "Select Input Type:",
        ["Paste text / WhatsApp forward", "Article URL", "YouTube link"],
    )

    input_text = st.text_area("Paste content here", height=200)
    analyze_btn = st.button("Analyse")

    source_map = {
        "Paste text / WhatsApp forward": "text",
        "Article URL": "url",
        "YouTube link": "youtube",
    }
    source_type = source_map[source_type_display]

if analyze_btn:
    log.info(f"Analyse clicked | source_type={source_type!r} | input length={len(input_text)}")

    if not input_text.strip():
        log.warning("Empty input submitted — aborting request")
        with col_results:
            st.warning("Please enter some content before clicking Analyse.")
    else:
        with st.spinner("Analyzing with TruthLens pipeline…"):
            payload = {"text": input_text, "source_type": source_type}

            try:
                # ── Main analysis ──────────────────────────────────────────
                log.debug(f"POST {API_URL}/analyze | payload preview: {input_text[:80]!r}")
                t0 = time.time()
                response = requests.post(
                    f"{API_URL}/analyze", json=payload, timeout=60
                )
                log.info(f"/analyze responded {response.status_code} in {time.time() - t0:.2f}s")
                response.raise_for_status()
                data = response.json()
                log.debug(
                    f"/analyze data keys: {list(data.keys())} | "
                    f"bert={data.get('bert')} | xgb={data.get('xgb')} | "
                    f"claims_used={len(data.get('claims_used', []))}"
                )

                # ── LIME HTML ──────────────────────────────────────────────
                log.debug(f"POST {API_URL}/lime_html")
                t0 = time.time()
                html_response = requests.post(
                    f"{API_URL}/lime_html", json=payload, timeout=60
                )
                log.info(f"/lime_html responded {html_response.status_code} in {time.time() - t0:.2f}s")
                html_response.raise_for_status()
                lime_html_str = html_response.json()["html"]
                log.debug(f"/lime_html returned {len(lime_html_str)} chars of HTML")

                with col_results:
                    # ── Row 1: Verdict + Confidence ───────────────────────
                    bert_result = data.get("bert", {})
                    label = bert_result.get("label", "Unknown")
                    confidence = bert_result.get("confidence", 0.0)
                    log.debug(f"Rendering verdict: label={label!r}, confidence={confidence}")

                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        if label == "Misleading":
                            st.error("🚨 Verdict: MISLEADING")
                        else:
                            st.success("✅ Verdict: GENUINE")
                    with m_col2:
                        st.metric("Confidence", f"{confidence * 100:.0f}%")

                    # ── Row 2: LIME AI Reasoning ──────────────────────────
                    st.subheader("AI reasoning (LIME)")
                    st.components.v1.html(lime_html_str, height=380, scrolling=True)

                    # ── Row 3: Fact-check results ─────────────────────────
                    st.subheader("Fact-check results")
                    fact_checks = data.get("fact_checks", [])
                    log.debug(f"Rendering {len(fact_checks)} fact-check result(s)")
                    if not fact_checks:
                        st.warning("No claims were extracted for fact-checking.")
                    for fc in fact_checks:
                        if fc.get("source"):
                            st.info(
                                f"**Source:** {fc['source']}  |  "
                                f"**Verdict:** {fc['verdict']}\n\n"
                                f"[Read more]({fc['url']})"
                            )
                        else:
                            st.warning("No match found in fact-check databases.")

                    # ── Row 4: Baseline model comparison ─────────────────
                    st.subheader("Baseline model comparison")
                    xgb_result = data.get("xgb", {})
                    comparison_data = [
                        {
                            "Model": "BERT (Fine-Tuned)",
                            "Prediction": label,
                            "Probability": f"{confidence * 100:.1f}%",
                        },
                        {
                            "Model": "XGBoost + TF-IDF (Baseline)",
                            "Prediction": xgb_result.get("label", "—"),
                            "Probability": f"{xgb_result.get('probability', 0) * 100:.1f}%",
                        },
                    ]
                    st.dataframe(
                        pd.DataFrame(comparison_data),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ── Row 5: Extracted claims (debug / transparency) ────
                    with st.expander("Claims sent to Fact-Check API"):
                        for i, c in enumerate(data.get("claims_used", []), 1):
                            st.write(f"{i}. {c}")

                log.info("Render complete — request cycle finished successfully")

            except requests.exceptions.ConnectionError as e:
                log.error(f"ConnectionError — backend unreachable: {e}")
                with col_results:
                    st.error(
                        "Cannot reach the TruthLens API on port 8000. "
                        "Start it with:  `uvicorn app:app --reload`"
                    )
            except requests.exceptions.Timeout as e:
                log.error(f"Timeout — backend took too long: {e}")
                with col_results:
                    st.error("Request timed out. The model may still be loading — try again.")
            except requests.exceptions.HTTPError as e:
                log.error(f"HTTPError — backend returned {e.response.status_code}: {e.response.text}")
                with col_results:
                    st.error(f"Backend returned an error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                log.exception(f"Unexpected error during analysis: {e}")
                with col_results:
                    st.error(f"Unexpected error: {e}")
