import logging
import sys
import time

import streamlit as st
import streamlit.components.v1 as components
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
st.caption("BERT classification + live web double-check, combined into one verdict.")

col_input, col_results = st.columns([1, 1])

with col_input:
    source_type_display = st.radio(
        "Select Input Type:",
        ["Paste text / WhatsApp forward", "YouTube link"],
    )

    placeholder = (
        "Paste the article text, message, or claim here..."
        if source_type_display == "Paste text / WhatsApp forward"
        else "Paste a YouTube video URL (e.g. https://www.youtube.com/watch?v=...)"
    )
    input_text = st.text_area("Paste content here", height=200, placeholder=placeholder)
    analyze_btn = st.button("Analyse")

    source_map = {
        "Paste text / WhatsApp forward": "text",
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
        with st.spinner("Analyzing with TruthLens pipeline (model + live web check)…"):
            payload = {"text": input_text, "source_type": source_type}

            try:
                # ── Main analysis ──────────────────────────────────────────
                log.debug(f"POST {API_URL}/analyze | payload preview: {input_text[:80]!r}")
                t0 = time.time()
                response = requests.post(
                    f"{API_URL}/analyze", json=payload, timeout=90
                )
                log.info(f"/analyze responded {response.status_code} in {time.time() - t0:.2f}s")
                response.raise_for_status()
                data = response.json()
                log.debug(
                    f"/analyze data keys: {list(data.keys())} | "
                    f"final_verdict={data.get('final_verdict')} | "
                    f"claims_used={len(data.get('claims_used', []))}"
                )

                # ── LIME HTML ──────────────────────────────────────────────
                log.debug(f"POST {API_URL}/lime_html")
                t0 = time.time()
                html_response = requests.post(
                    f"{API_URL}/lime_html", json=payload, timeout=90
                )
                log.info(f"/lime_html responded {html_response.status_code} in {time.time() - t0:.2f}s")
                html_response.raise_for_status()
                lime_html_str = html_response.json()["html"]
                log.debug(f"/lime_html returned {len(lime_html_str)} chars of HTML")

                with col_results:
                    # ── Row 1: FINAL VERDICT (combined signal) ────────────
                    final = data.get("final_verdict", {})
                    final_label = final.get("label", "Unknown")
                    final_conf = final.get("confidence", 0.0)
                    log.debug(f"Rendering final verdict: {final_label} ({final_conf:.3f})")

                    st.subheader("Final Verdict")
                    fv_col1, fv_col2 = st.columns(2)
                    with fv_col1:
                        if final_label == "Misleading":
                            st.error("🚨 LIKELY MISLEADING")
                        else:
                            st.success("✅ LIKELY GENUINE")
                    with fv_col2:
                        st.metric("Overall Confidence", f"{final_conf * 100:.0f}%")

                    with st.expander("Why this verdict? (reasoning trail)", expanded=True):
                        for line in final.get("reasoning", []):
                            st.write(f"• {line}")

                    st.divider()

                    # ── Row 2: Individual model scores ─────────────────────
                    st.subheader("Model comparison")
                    bert_result = data.get("bert", {})
                    label = bert_result.get("label", "Unknown")
                    confidence = bert_result.get("confidence", 0.0)
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

                    # ── Row 3: LIME AI Reasoning ───────────────────────────
                    st.subheader("AI reasoning (LIME)")
                    components.html(lime_html_str, height=380, scrolling=True)

                    # ── Row 4: Live web double-check ───────────────────────
                    st.subheader("Live web double-check")
                    web_checks = data.get("web_checks", [])
                    entity_checks = data.get("entity_checks", [])
                    log.debug(f"Rendering {len(web_checks)} web-check result(s)")
                    if not web_checks:
                        st.info("No claims were searched on the web.")
                    for idx, wc in enumerate(web_checks):
                        entity_check = entity_checks[idx] if idx < len(entity_checks) else None
                        with st.container(border=True):
                            st.markdown(f"**Claim:** {wc['claim']}")
                            st.caption(wc["corroboration"])
                            if entity_check and entity_check.get("order_mismatch"):
                                st.error(f"⚠️ Possible swap detected: {entity_check['detail']}")
                            for ev in wc.get("evidence", [])[:3]:
                                title = ev.get("title") or "Untitled"
                                url = ev.get("url") or "#"
                                snippet = ev.get("snippet", "")
                                st.markdown(f"- [{title}]({url})  \n  {snippet[:150]}")

                    # ── Row 5: Fact-check database results ─────────────────
                    st.subheader("Fact-check database results")
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

                    # ── Row 6: Extracted claims (debug / transparency) ─────
                    with st.expander("Claims sent for verification"):
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
                    st.error("Request timed out. The model or web search may be slow — try again.")
            except requests.exceptions.HTTPError as e:
                log.error(f"HTTPError — backend returned {e.response.status_code}: {e.response.text}")
                with col_results:
                    st.error(f"Backend returned an error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                log.exception(f"Unexpected error during analysis: {e}")
                with col_results:
                    st.error(f"Unexpected error: {e}")
