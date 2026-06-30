import streamlit as st
import cv2
import numpy as np
import hashlib
import imagehash
from PIL import Image
import io
import tempfile
import os
import logging
import moviepy.editor as mp
from datetime import datetime
import json
from settings import Config
from config_manager import config_manager
import base64
import torch
import time
import pandas as pd

# Load .env so GOOGLE_API_KEY is available for the Fact-Check API
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
Config.configure_logging()

# Import custom modules
from detection.ai_image_detector import AIImageDetector
from detection.ai_video_detector import AIVideoDetector
from detection.face_extractor import FaceExtractor
from models.xception_net import load_xception_model
from models.meso_net import load_meso_model
from detection.deepfake_detector import DeepfakeDetector
from detection.duplicate_detector import DuplicateDetector
from utils.media_processor import MediaProcessor
from utils.visualization import create_result_card, create_confidence_chart
from reports.analysis_summary import AnalysisSummaryGenerator
from reports.advanced_pdf_generator import AdvancedPDFGenerator
from reports.technical_exporter import TechnicalDetailsExporter
from reports.advanced_batch_processor import AdvancedBatchProcessor

# ── TruthLens imports (lazy — only loaded when Fake News mode is selected) ───
_truthlens_loaded = False
_bert_tokenizer = None
_bert_model = None
_nlp = None
_lime_explainer = None
_xgb_pipeline = None

def _load_truthlens():
    """Load all TruthLens NLP models on first use (cached in session state)."""
    global _truthlens_loaded, _bert_tokenizer, _bert_model, _nlp, _lime_explainer, _xgb_pipeline

    if _truthlens_loaded:
        return True

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import spacy
        from truthlens.explainer import get_explainer
        from truthlens.baseline import train_xgb_pipeline

        _PRIMARY_MODEL = "hamzab/roberta-fake-news-classification"
        _FALLBACK_MODEL = "bert-base-uncased"

        try:
            _bert_tokenizer = AutoTokenizer.from_pretrained(_PRIMARY_MODEL)
            _bert_model = AutoModelForSequenceClassification.from_pretrained(_PRIMARY_MODEL)
        except Exception:
            _bert_tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_MODEL)
            _bert_model = AutoModelForSequenceClassification.from_pretrained(_FALLBACK_MODEL)

        _bert_model.eval()

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")

        _lime_explainer = get_explainer()
        _xgb_pipeline = train_xgb_pipeline()
        _truthlens_loaded = True
        return True
    except Exception as e:
        logging.error(f"Failed to load TruthLens models: {e}")
        return False


# Configure Streamlit page
st.set_page_config(
    page_title="TrueLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102,126,234,0.35);
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
    .main-header p { font-size: 1.05rem; opacity: 0.9; margin: 0.5rem 0 0; }

    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .real-result { background: linear-gradient(135deg, #d4edda, #c3e6cb); border-left-color: #28a745; }
    .fake-result { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border-left-color: #dc3545; }
    .warning-result { background: linear-gradient(135deg, #fff3cd, #ffeeba); border-left-color: #ffc107; }

    .news-card {
        background: linear-gradient(135deg, #e8f4fd, #d6eaf8);
        border-left: 5px solid #3498db;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(52,152,219,0.15);
    }
    .misleading-news {
        background: linear-gradient(135deg, #fdedec, #fadbd8);
        border-left-color: #e74c3c;
    }
    .genuine-news {
        background: linear-gradient(135deg, #eafaf1, #d5f5e3);
        border-left-color: #27ae60;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-red { background: #e74c3c; color: white; }
    .badge-green { background: #27ae60; color: white; }
    .badge-blue { background: #3498db; color: white; }

    .stProgress .st-bo { background-color: #667eea; }
    .section-divider { border: none; border-top: 2px solid #f0f2f6; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Log system status at startup
Config.log_system_status()

class MediaForensicsApp:
    def __init__(self):
        self.ai_detector = AIImageDetector()
        self.ai_video_detector = AIVideoDetector()
        self.deepfake_detector = DeepfakeDetector()
        self.duplicate_detector = DuplicateDetector()
        self.media_processor = MediaProcessor()
        self.face_extractor = FaceExtractor()
        # Load XceptionNet for deepfake detection
        self.xception_model = load_xception_model()
        # Load MesoNet for deepfake detection
        self.meso_model = load_meso_model()

        # Initialize comprehensive report generators
        self.summary_generator = AnalysisSummaryGenerator()
        self.pdf_generator = AdvancedPDFGenerator()
        self.tech_exporter = TechnicalDetailsExporter()
        self.batch_processor = AdvancedBatchProcessor()

    def run(self):
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🔍 TrueLens</h1>
            <p>Advanced AI-powered detection for synthetic media, deepfakes, and fake news</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")

            # Load user preferences
            user_prefs = config_manager.load_preferences()

            ALL_MODES = [
                "Detect AI-Generated Image",
                "Detect AI-Generated Video",
                "Detect Deepfake Video",
                "Detect Duplicate Image/Video",
                "Detect Fake News",
            ]

            detection_mode = st.selectbox(
                "🎯 Select Detection Mode:",
                ALL_MODES,
                index=ALL_MODES.index(user_prefs.get('detection_mode', 'Detect AI-Generated Image'))
                      if user_prefs.get('detection_mode') in ALL_MODES else 0
            )

            # Advanced options — only show media sliders for non-news modes
            if detection_mode != "Detect Fake News":
                st.subheader("🔧 Advanced Options")
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.1, max_value=1.0,
                    value=config_manager.get_preference('confidence_threshold', Config.CONFIDENCE_THRESHOLD),
                    step=0.01
                )
                video_frame_threshold = st.slider(
                    "Video Frame Fake % Threshold",
                    min_value=0.1, max_value=1.0,
                    value=config_manager.get_preference('video_frame_threshold', 0.3),
                    step=0.05
                )
                enable_visualization = st.checkbox(
                    "Enable Visual Analysis",
                    value=config_manager.get_preference('enable_visualization', Config.ENABLE_VISUALIZATION)
                )
                save_results = st.checkbox(
                    "Save Results",
                    value=config_manager.get_preference('save_results', Config.SAVE_RESULTS)
                )
            else:
                confidence_threshold = 0.51
                video_frame_threshold = 0.3
                enable_visualization = False
                save_results = False
                st.subheader("📰 Fake News Options")
                st.info("Paste article text or a YouTube link below. The system will use BERT + XGBoost + live fact-checking to analyse the content.")
                st.markdown("**Optional:** Set your [Google Fact Check API key](https://developers.google.com/fact-check/tools/api) as `GOOGLE_API_KEY` in your environment for live fact-check results.")

            # Set fixed default models
            config_manager.set_model_preference('deepfake_detection', 'xception')
            config_manager.set_model_preference('ai_image_detection', 'efficientnet')

            # Save preferences
            current_prefs = {
                'detection_mode': detection_mode,
                'confidence_threshold': confidence_threshold,
                'video_frame_threshold': video_frame_threshold,
                'enable_visualization': enable_visualization,
                'save_results': save_results
            }
            if current_prefs != {k: user_prefs.get(k) for k in current_prefs.keys()}:
                user_prefs.update(current_prefs)
                config_manager.save_preferences(user_prefs)

            with st.expander("📊 Configuration Status"):
                status = config_manager.get_config_status()
                st.json(status)

        # ── FAKE NEWS MODE ───────────────────────────────────────────────────
        if detection_mode == "Detect Fake News":
            self.run_fake_news_detector()
            return

        # ── MEDIA DETECTION MODES ────────────────────────────────────────────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📁 Upload Media")

            if "Image" in detection_mode:
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                    help="Upload JPG, PNG, or other image formats"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Choose a video file",
                    type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                    help="Upload MP4, AVI, MOV, or other video formats"
                )

        with col2:
            st.header("👁️ Preview")

            if uploaded_file is not None:
                if "image" in uploaded_file.type:
                    try:
                        image_bytes = uploaded_file.read()
                        uploaded_file.seek(0)
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image preview: {str(e)}")
                else:
                    st.video(uploaded_file)
                    st.info("Video preview shown above. Analysis will process key frames.")

        if uploaded_file is not None:
            st.header("🔍 Analysis Results")

            with st.spinner("🧠 Analyzing media... This may take a moment."):
                results = self.process_media(
                    uploaded_file,
                    detection_mode,
                    confidence_threshold,
                    enable_visualization,
                    video_frame_threshold
                )

            self.display_results(results, enable_visualization)

            if save_results:
                st.subheader("📄 Report Generation")

                report_col1, report_col2, report_col3 = st.columns(3)

                with report_col1:
                    if st.button("📊 Generate JSON Report"):
                        with st.spinner("Generating comprehensive JSON report..."):
                            detailed_report = self.summary_generator.generate_detailed_report(results, results['file_info'])
                            json_report = json.dumps(detailed_report, indent=2, default=str)
                        st.download_button(
                            label="📊 Download JSON Report",
                            data=json_report,
                            file_name=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                with report_col2:
                    if st.button("📋 Generate PDF Report"):
                        with st.spinner("Generating professional PDF report..."):
                            try:
                                pdf_path = f"temp_pdf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                self.pdf_generator.generate_comprehensive_pdf_report(results, results['file_info'], pdf_path)
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_data = pdf_file.read()
                                st.download_button(
                                    label="📋 Download PDF Report",
                                    data=pdf_data,
                                    file_name=f"forensics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                if os.path.exists(pdf_path):
                                    os.remove(pdf_path)
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")

                with report_col3:
                    export_format = st.selectbox("Technical Export Format:", ['json', 'csv', 'xml', 'excel', 'txt'])
                    if st.button("⚙️ Export Technical Details"):
                        with st.spinner(f"Exporting technical details as {export_format.upper()}..."):
                            try:
                                temp_path = f"temp_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                                self.tech_exporter.export_technical_details(results, temp_path, export_format)
                                with open(temp_path, "rb") as export_file:
                                    export_data = export_file.read()
                                st.download_button(
                                    label=f"⚙️ Download {export_format.upper()} Export",
                                    data=export_data,
                                    file_name=f"technical_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                    mime=f"application/{export_format}"
                                )
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            except Exception as e:
                                st.error(f"Error exporting technical details: {str(e)}")

                st.subheader("📈 Executive Summary")
                exec_summary = self.summary_generator.generate_executive_summary(results)
                with st.expander("View Executive Summary", expanded=True):
                    st.json(exec_summary)

    # ── FAKE NEWS DETECTOR UI ────────────────────────────────────────────────
    def run_fake_news_detector(self):
        st.header("📰 Fake News Detector")
        st.markdown("Powered by **BERT**, **XGBoost**, **Google Fact-Check API**, and **live web search**.")

        col_input, col_results = st.columns([1, 1])

        with col_input:
            source_type_display = st.radio(
                "Select Input Type:",
                ["Paste text / Article", "YouTube link"],
                key="fn_source_type"
            )
            source_map = {
                "Paste text / Article": "text",
                "YouTube link": "youtube",
            }
            source_type = source_map[source_type_display]

            input_text = st.text_area(
                "Paste your content here",
                height=220,
                placeholder="E.g. 'The COVID-19 vaccine contains microchips for tracking.' or a YouTube URL",
                key="fn_input_text"
            )

            analyse_btn = st.button("🔍 Analyse", type="primary", key="fn_analyse_btn")

        if analyse_btn:
            if not input_text.strip():
                with col_results:
                    st.warning("⚠️ Please enter some content before clicking Analyse.")
                return

            with col_results:
                with st.spinner("🤖 Loading AI models and analysing... (first run may take ~30s)"):
                    loaded = _load_truthlens()

                if not loaded:
                    st.error("❌ Failed to load TruthLens models. Please check that `transformers`, `spacy`, and `xgboost` are installed.")
                    return

                with col_results:
                    with st.spinner("🔍 Running full analysis pipeline..."):
                        t_start = time.time()
                        data = self._run_truthlens_pipeline(input_text.strip(), source_type)
                        elapsed = time.time() - t_start

                    if "error" in data:
                        st.error(f"❌ Analysis failed: {data['error']}")
                        return

                    final_verdict = data["final_verdict"]
                    bert_result = data["bert"]
                    xgb_result = data["xgb"]
                    fact_checks = data["fact_checks"]
                    web_checks = data["web_checks"]
                    claims_used = data["claims_used"]
                    lime_words = data["lime_words"]

                    label = final_verdict.get("label", "Unknown")
                    confidence = final_verdict.get("confidence", 0.0)
                    reasoning = final_verdict.get("reasoning", [])

                    # ── Verdict card ──────────────────────────────────────────
                    if label == "Misleading":
                        card_cls = "fake-result"
                        icon = "🚨"
                        headline = "MISLEADING / FAKE NEWS DETECTED"
                    else:
                        card_cls = "real-result"
                        icon = "✅"
                        headline = "CONTENT APPEARS GENUINE"

                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                        <h2>{icon} {headline}</h2>
                        <h3>Overall Confidence: {confidence:.1%}</h3>
                        <p style="font-size:0.85rem; opacity:0.7;">Analysis completed in {elapsed:.1f}s</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Metrics ───────────────────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Final Verdict", label)
                    with m2:
                        bert_label_val = bert_result.get('label', '?')
                        bert_conf_val = bert_result.get('confidence', 0)
                        st.metric("BERT Says", f"{bert_label_val} ({bert_conf_val:.0%})")
                    with m3:
                        xgb_prob = xgb_result.get("probability", 0) if xgb_result else 0
                        xgb_label_val = xgb_result.get("label", "?") if xgb_result else "?"
                        st.metric("XGBoost Says", f"{xgb_label_val} ({xgb_prob:.0%})")
                    with m4:
                        raw_score = final_verdict.get("raw_score", confidence)
                        st.metric("Raw Risk Score", f"{raw_score:.2f}", help="≥0.38 = Misleading. Closer to 1.0 = more suspicious.")

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # ── AI Reasoning ──────────────────────────────────────────
                    st.subheader("🧠 AI Reasoning Breakdown")
                    for r in reasoning:
                        st.markdown(f"- {r}")

                    # ── LIME word weights ─────────────────────────────────────
                    if lime_words:
                        st.subheader("🔬 Key Words That Influenced the Decision")
                        lime_df = pd.DataFrame(lime_words, columns=["Word", "Weight"])
                        lime_df["Impact"] = lime_df["Weight"].apply(
                            lambda w: "🔴 Misleading Signal" if w > 0 else "🟢 Genuine Signal"
                        )
                        lime_df["Weight"] = lime_df["Weight"].apply(lambda w: f"{w:.4f}")
                        st.dataframe(lime_df, use_container_width=True, hide_index=True)

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # ── Fact-check results ────────────────────────────────────
                    st.subheader("📋 Fact-Check Results")
                    if not fact_checks:
                        st.info("No claims were extracted for fact-checking.")
                    else:
                        for fc in fact_checks:
                            if fc.get("source") and fc.get("source") != "System":
                                verdict_lower = str(fc.get("verdict", "")).lower()
                                is_false = any(k in verdict_lower for k in ["false", "pants on fire", "misleading", "fake"])
                                badge = "🔴" if is_false else "🟢"
                                link = f"[Read more]({fc['url']})" if fc.get("url") else ""
                                st.info(f"{badge} **{fc['source']}** — *{fc['verdict']}* {link}")
                            else:
                                verdict = fc.get("verdict", "No match found")
                                st.warning(f"⚠️ {verdict}")

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # ── Web corroboration ─────────────────────────────────────
                    st.subheader("🌐 Live Web Corroboration")
                    if not web_checks:
                        st.info("No web search was performed.")
                    else:
                        for w in web_checks:
                            match_count = w.get("match_count", 0)
                            corroboration = w.get("corroboration", "")
                            icon_w = "✅" if match_count >= 3 else ("⚠️" if match_count > 0 else "❌")
                            st.markdown(f"**{icon_w} {corroboration}**")
                            if w.get("evidence"):
                                with st.expander(f"View {match_count} source(s) for this claim"):
                                    for ev in w["evidence"][:3]:
                                        st.markdown(f"- [{ev.get('title','No title')}]({ev.get('url','#')}) — *{ev.get('source','')}*")

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # ── Model comparison table ────────────────────────────────
                    st.subheader("📊 Model Comparison")
                    comp_rows = [
                        {
                            "Model": "BERT (Fine-Tuned)",
                            "Prediction": bert_result.get("label", "—"),
                            "Confidence": f"{bert_result.get('confidence', 0) * 100:.1f}%",
                        },
                    ]
                    if xgb_result:
                        comp_rows.append({
                            "Model": "XGBoost + TF-IDF (Baseline)",
                            "Prediction": xgb_result.get("label", "—"),
                            "Confidence": f"{xgb_result.get('probability', 0) * 100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                    # ── Extracted claims ──────────────────────────────────────
                    with st.expander("🔎 Claims Extracted from Content"):
                        if claims_used:
                            for i, c in enumerate(claims_used, 1):
                                st.write(f"{i}. {c}")
                        else:
                            st.write("No claims were extracted.")

    def _run_truthlens_pipeline(self, text: str, source_type: str) -> dict:
        """Run the full TruthLens analysis pipeline and return structured results."""
        try:
            from truthlens.ingestor import ingest
            from truthlens.claims import extract_claims
            from truthlens.classifier import bert_classify
            from truthlens.explainer import lime_explain
            from truthlens.fact_check import fact_check
            from truthlens.web_verify import web_verify_claims
            from truthlens.baseline import xgb_pipeline_predict
            from truthlens.aggregator import compute_final_verdict

            # Step 1: Ingest
            ingested_text = ingest(text, source_type)
            if not ingested_text or ingested_text.startswith("Error during ingestion"):
                return {"error": f"Ingestion failed: {ingested_text}"}

            # Step 2: Extract claims
            claims = extract_claims(ingested_text, _nlp)

            # Step 3: BERT classification
            bert_result = bert_classify(ingested_text, _bert_tokenizer, _bert_model)
            logging.info(f"BERT result: {bert_result}")

            # Step 4: LIME explanation
            try:
                lime_words = lime_explain(ingested_text, _bert_tokenizer, _bert_model, _lime_explainer)
            except Exception:
                lime_words = []

            # Step 5: Fact-check
            fact_results = [fact_check(c) for c in claims]

            # Step 6: Web verification
            try:
                web_results = web_verify_claims(claims)
            except Exception:
                web_results = []

            # Step 7: XGBoost baseline
            xgb_result = xgb_pipeline_predict(ingested_text, _xgb_pipeline)
            logging.info(f"XGBoost result: {xgb_result}")

            # Step 8: Aggregate
            final_verdict = compute_final_verdict(bert_result, xgb_result, fact_results, web_results)
            logging.info(f"Final verdict: {final_verdict}")

            return {
                "final_verdict": final_verdict,
                "bert": bert_result,
                "xgb": xgb_result,
                "lime_words": lime_words,
                "fact_checks": fact_results,
                "web_checks": web_results,
                "claims_used": claims,
            }
        except Exception as e:
            logging.error(f"TruthLens pipeline error: {e}", exc_info=True)
            return {"error": str(e)}

    def process_media(self, uploaded_file, detection_mode, threshold, enable_viz, frame_fake_threshold=0.3):
        """Process uploaded media based on selected detection mode"""
        logging.info(f"Processing media: {uploaded_file.name} with mode: {detection_mode}")

        results = {
            "mode": detection_mode,
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "type": uploaded_file.type
            }
        }

        try:
            uploaded_file.seek(0)

            if detection_mode == "Detect AI-Generated Image":
                detection_result = self.ai_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("AI detector returned None result")
                if 'error' in detection_result:
                    raise ValueError(f"AI detection failed: {detection_result['error']}")
                results.update(detection_result)

            elif detection_mode == "Detect AI-Generated Video":
                detection_result = self.ai_video_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("AI video detector returned None result")
                if 'error' in detection_result:
                    raise ValueError(f"AI video detection failed: {detection_result['error']}")
                results.update(detection_result)

            elif detection_mode == "Detect Deepfake Video":
                detection_result = self.deepfake_detector.detect(uploaded_file, threshold, enable_viz, frame_fake_threshold)
                if detection_result is None:
                    raise ValueError("Deepfake detector returned None result")
                if 'error' in detection_result:
                    raise ValueError(f"Deepfake detection failed: {detection_result['error']}")
                results.update(detection_result)

            elif detection_mode == "Detect Duplicate Image/Video":
                detection_result = self.duplicate_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("Duplicate detector returned None result")
                if 'error' in detection_result:
                    raise ValueError(f"Duplicate detection failed: {detection_result['error']}")
                results.update(detection_result)

        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
            logging.error(f"Detection failed for {uploaded_file.name}: {error_msg}", exc_info=True)
            results["error"] = error_msg
            results["status"] = "error"

        return results

    def display_results(self, results, enable_visualization):
        """Display analysis results with visual components"""
        if "error" in results:
            logging.error(f"Analysis error: {results['error']}")
            st.error(f"❌ Analysis failed: {results['error']}")
            return

        if results.get("is_fake", False):
            card_class = "fake-result"
            icon = "❌"
            status = "SYNTHETIC/FAKE DETECTED"
        elif results.get("is_duplicate", False):
            card_class = "warning-result"
            icon = "⚠️"
            status = "DUPLICATE DETECTED"
        else:
            card_class = "real-result"
            icon = "✅"
            status = "AUTHENTIC/ORIGINAL"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <h2>{icon} {status}</h2>
            <h3>Confidence: {results.get('confidence', 0):.1%}</h3>
            <p><strong>Analysis:</strong> {results.get('explanation', 'No explanation available')}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{results.get('confidence', 0):.1%}", delta=f"{results.get('confidence', 0) - 0.5:.1%}")
        with col2:
            st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
        with col3:
            st.metric("Model Accuracy", f"{results.get('model_accuracy', 0.85):.1%}")

        if enable_visualization and results.get('visualizations'):
            st.subheader("📊 Visual Analysis")
            if 'heatmap' in results['visualizations']:
                st.image(results['visualizations']['heatmap'], caption="Attention Heatmap - Red areas indicate suspicious regions")
            if 'confidence_chart' in results['visualizations']:
                st.plotly_chart(results['visualizations']['confidence_chart'], use_container_width=True)

        with st.expander("🔬 Technical Details"):
            st.json(results.get('technical_details', {}))

    def generate_report(self, results, filename):
        """Generate downloadable JSON report"""
        logging.info(f"Generating report for {filename}")
        logging.info("Error handling is active.")
        report = {
            "media_forensics_report": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "file_analyzed": filename,
                "results": results
            }
        }
        return json.dumps(report, indent=2)


# Initialize and run the app
if __name__ == "__main__":
    app = MediaForensicsApp()
    app.run()
