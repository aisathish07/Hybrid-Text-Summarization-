import sys
import os

# Ensure the parent directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pdfplumber
import requests
from utils.export_utils import create_docx, create_pdf

# Page Config
st.set_page_config(
    page_title="Hybrid Summarizer",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def get_local_pipeline_runner():
    from main import run_summarization_pipeline

    return run_summarization_pipeline


def is_torch_oom_error(exc):
    exc_type = type(exc)
    return exc_type.__name__ == "OutOfMemoryError" and exc_type.__module__.startswith("torch")

# ---- Custom CSS Styling ----
st.markdown("""
<style>
    /* Custom Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 5px 0 0 0;
    }

    /* Sidebar Section Headers */
    .sidebar-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #667eea;
    }

    /* Custom Button Styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
    }

    /* Result Card Styling */
    .result-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }

    /* Metric Card Styling */
    .metric-container {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Pipeline Stage Headers */
    .stage-header {
        color: #667eea;
        font-weight: 600;
        padding: 10px 0;
        border-bottom: 2px solid #e4e8ec;
        margin-bottom: 15px;
    }

    /* Download Button Styling */
    .download-btn {
        background: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        text-decoration: none;
    }

    /* Info Box Styling */
    .info-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d4ecf9 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 40px;
        color: #6c757d;
    }
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Custom CSS Styling ----
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main font family */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Custom Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
    }

    /* Sidebar sections */
    .sidebar-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* Custom card for results */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }

    /* Stage indicator */
    .stage-indicator {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Download buttons */
    .download-btn > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f5f7fa;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background: #f0f2f6;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #e0e7ff 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
    }

    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #dcfce7 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        background: #f8f9fa;
        border-radius: 16px;
        border: 2px dashed #dee2e6;
    }

    .empty-state h3 {
        color: #6c757d;
        margin-bottom: 1rem;
    }

    /* Sample text button */
    .sample-btn {
        background: #f0f2f6 !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def create_download_buttons(summary_text, language, title="Generated Summary", file_prefix="summary"):
    col1, col2 = st.columns(2)
    with col1:
        docx_bytes = create_docx(summary_text, title=title)
        st.download_button(
            label="📄 Download as Word (.docx)",
            data=docx_bytes,
            file_name=f"{file_prefix}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    with col2:
        try:
            pdf_bytes = create_pdf(summary_text, title=title, language=language)
            st.download_button(
                label="📕 Download as PDF (.pdf)",
                data=pdf_bytes,
                file_name=f"{file_prefix}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"PDF generation is currently limited without custom fonts: {e}")

# Sample text for empty state
SAMPLE_TEXT = """Artificial intelligence has revolutionized the way we process and understand large amounts of text. Text summarization, in particular, has become an essential tool for extracting key information from lengthy documents. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization works by selecting and combining existing sentences directly from the source text, while abstractive summarization generates new sentences that capture the essence of the original content. The hybrid approach combines the strengths of both methods, using extractive techniques to identify key sentences and abstractive models to rewrite and polish the final summary. Recent advances in transformer models have significantly improved the quality of abstractive summaries, making them more coherent and contextually accurate. Natural Language Processing continues to evolve, with new models achieving state-of-the-art results on various summarization benchmarks."""

# ---- Sidebar Config ----
with st.sidebar:
    st.header("⚙️ Configuration")

    # Language Section
    with st.expander("🌐 Language", expanded=True):
        language_opts = {"English": "en", "Gujarati": "gu", "Hindi": "hi"}
        selected_lang_name = st.selectbox("Select Language:", options=list(language_opts.keys()), label_visibility="collapsed")
        language_code = language_opts[selected_lang_name]

    # Settings Section
    with st.expander("📊 Summarization Settings", expanded=True):
        top_n = st.slider(
            "Extractive Sentences (Top N)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of key sentences to extract initially using TextRank algorithm."
        )
        clusters = st.slider(
            "Semantic Clusters",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of core ideas to narrow down to after clustering."
        )
        max_len = st.slider(
            "Abstractive Max Length",
            min_value=50,
            max_value=500,
            value=150,
            help="Maximum length of the final generated AI summary in words."
        )

    # Info Section
    with st.expander("💡 How It Works"):
        st.markdown("""
        **Hybrid Approach:**
        1. **TextRank** - Extracts the most important sentences from your text
        2. **Semantic Clustering** - Groups similar ideas and removes redundancy
        3. **Abstractive Ensemble** - Multiple AI models generate summary candidates
        4. **Meta-Selection** - Best summary is selected using quality metrics
        """)

    st.markdown("---")

    # Backend Selection
    with st.expander("🖥️ Backend", expanded=True):
        backend_mode = st.radio(
            "Run models on:",
            ["💻 Local Machine", "☁️ Google Colab (Remote)"],
            index=0,
            help="Choose where to run the AI models. Use Colab for faster GPU inference."
        )
        use_remote = backend_mode.startswith("☁️")

        colab_url = ""
        if use_remote:
            colab_url = st.text_input(
                "Colab API URL:",
                placeholder="https://xxxx.ngrok-free.app",
                help="Paste the ngrok URL from your Colab notebook here."
            )
            if colab_url:
                # Quick health check
                try:
                    resp = requests.get(f"{colab_url.rstrip('/')}/health", timeout=5)
                    if resp.status_code == 200:
                        st.success("✅ Connected to Colab backend!")
                    else:
                        st.warning("⚠️ Server responded but may not be ready.")
                except Exception:
                    st.error("❌ Cannot reach Colab server. Check the URL and that the notebook is running.")
            else:
                st.info("📋 See `COLAB_SETUP.md` for setup instructions.")

    st.markdown("---")
    st.markdown("### 📈 Supported Languages")
    st.markdown("- 🇬🇧 **English** - Full support")
    st.markdown("- 🇮🇳 **Gujarati** - Full support")
    st.markdown("- 🇮🇳 **Hindi** - Full support")

# ---- Main Content Area ----
# Custom Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Hybrid Text Summarization System</h1>
    <p>Upload a document or paste your text to generate a concise AI-powered summary</p>
</div>
""", unsafe_allow_html=True)

# Empty State / Welcome Message
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

col_left, col_right = st.columns([3, 1])

with col_left:
    tab_input, tab_file = st.tabs(["📝 Paste Text", "📂 Upload Document"])

    input_text = ""

    with tab_input:
        user_text = st.text_area(
            "Enter your text here:",
            height=300,
            placeholder="Paste a news article, report, or any long text...",
            label_visibility="collapsed",
            key="text_input"
        )
        if user_text:
            input_text = user_text

        # Sample text button
        if not input_text:
            st.markdown("---")
            col_s1, col_s2 = st.columns([1, 4])
            with col_s1:
                if st.button("📋 Load Sample Text", use_container_width=True):
                    input_text = SAMPLE_TEXT
                    st.rerun()
            with col_s2:
                st.caption("Try it with sample text to see how it works")

    with tab_file:
        uploaded_file = st.file_uploader("Upload a TXT or PDF file:", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.pdf'):
                with st.spinner("Extracting text from PDF..."):
                    input_text = extract_text_from_pdf(uploaded_file)
            else:
                input_text = str(uploaded_file.read(), "utf-8")

            st.success(f"✅ File uploaded: {uploaded_file.name}")
            with st.expander("👁️ Show extracted text"):
                st.write(input_text[:1000] + ("..." if len(input_text) > 1000 else ""))

with col_right:
    st.markdown("### 📋 Quick Tips")
    st.info("💡 **Tip:** For best results, use texts with 3+ paragraphs.")
    st.info("📄 **Supported:** TXT and PDF files up to 10MB")
    st.info("⏱️ **Processing:** Takes 30-60 seconds depending on text length")

# ---- Summarization Button ----
st.markdown("---")
if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
    if not input_text.strip():
        st.error("⚠️ Please provide text or upload a file first.")
    else:
        # Progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Define progress callback (local only)
        def update_progress(stage_name, percent):
            status_text.markdown(f"**{stage_name}**")
            progress_bar.progress(min(percent, 100))

        try:
            if use_remote and colab_url:
                # ---- Remote Colab Backend ----
                status_text.markdown("**☁️ Sending to Colab backend...**")
                progress_bar.progress(30)

                api_endpoint = f"{colab_url.rstrip('/')}/summarize"
                payload = {
                    "text": input_text,
                    "top_n": top_n,
                    "clusters": clusters,
                    "max_length": max_len,
                    "language": language_code
                }

                response = requests.post(api_endpoint, json=payload, timeout=300)

                if response.status_code != 200:
                    raise Exception(f"Colab API returned error {response.status_code}: {response.text}")

                progress_bar.progress(90)
                status_text.markdown("**☁️ Processing response...**")

                results = response.json()
                progress_bar.progress(100)

            elif use_remote and not colab_url:
                raise ValueError("Please enter the Colab API URL in the sidebar first.")

            else:
                # ---- Local Backend ----
                status_text.markdown("**Preparing local pipeline...**")
                progress_bar.progress(5)
                run_summarization_pipeline = get_local_pipeline_runner()
                results = run_summarization_pipeline(
                    text=input_text,
                    top_n=top_n,
                    clusters=clusters,
                    max_length=max_len,
                    language=language_code,
                    progress_callback=update_progress
                )

            # Fallback: if all abstractive models produced empty output
            if not results.get('best_summary'):
                results['best_summary'] = results.get('extractive_text', 'No summary could be generated.')
                results['best_model'] = 'extractive_fallback'

            # Store results in session_state so they persist across reruns
            st.session_state['results'] = results
            st.session_state['result_language'] = language_code

            # Clear progress
            status_text.empty()
            progress_bar.empty()

        except ValueError as e:
            progress_bar.empty()
            status_text.empty()
            st.error("⚠️ **Input Error:** The text you provided is too short or could not be processed.")
            st.info("💡 **Suggestion:** Try using a longer text with at least 3–5 sentences for best results.")
            with st.expander("🔍 Technical details"):
                st.exception(e)

        except OSError as e:
            progress_bar.empty()
            status_text.empty()
            if "model" in str(e).lower() or "checkpoint" in str(e).lower():
                st.error("📥 **Model Download Error:** A required AI model could not be loaded or downloaded.")
                st.info("💡 **Suggestion:** Check your internet connection and run `python download_models.py` to pre-download all models.")
            else:
                st.error(f"💾 **File System Error:** {e}")
            with st.expander("🔍 Technical details"):
                st.exception(e)

        except ImportError as e:
            progress_bar.empty()
            status_text.empty()
            st.error("📦 **Missing Dependency:** A required library is not installed.")
            st.code(f"pip install -r requirements.txt", language="bash")
            with st.expander("🔍 Technical details"):
                st.exception(e)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            error_msg = str(e)

            if is_torch_oom_error(e):
                st.error("💾 **Out of Memory:** Your GPU ran out of memory while processing the text.")
                st.info("💡 **Suggestion:** Try shortening your input text, or reduce the **Abstractive Max Length** slider in the sidebar.")
            elif "device" in error_msg.lower() and "cuda" in error_msg.lower():
                st.error("🖥️ **GPU/CPU Device Mismatch:** A model component is on the wrong device.")
                st.info("💡 **Suggestion:** Restart the Streamlit server and try again.")
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                st.error("🌐 **Network Error:** Could not connect to HuggingFace to download a model.")
                st.info("💡 **Suggestion:** Run `python download_models.py` once to pre-cache all models locally.")
            else:
                st.error(f"❌ **Unexpected Error:** Something went wrong in the pipeline.")

            with st.expander("🔍 Technical details — click to report this bug"):
                st.exception(e)

# ---- Display Results from Session State (persists across reruns) ----
if 'results' in st.session_state:
    results = st.session_state['results']
    result_language = st.session_state.get('result_language', 'en')

    st.success("✅ Summarization Complete!")

    res_tab_final, res_tab_stages, res_tab_metrics = st.tabs(["⭐ Final Summary", "🔄 Pipeline Stages", "📊 Meta-Selection Metrics"])

    with res_tab_final:
        st.markdown(f"""
        <div class="result-card">
            <p style="margin-bottom: 0.5rem; color: #6c757d;">Selected Model Phase</p>
            <span class="model-badge">{results['best_model']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Original Summary")
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size: 1.1rem; line-height: 1.8;">{results['best_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📥 Export Original")
        create_download_buttons(
            results['best_summary'],
            language=result_language,
            title="Generated Summary",
            file_prefix="summary_original"
        )

        if results.get('english_translation'):
            st.markdown("#### English Translation")
            st.markdown(f"""
            <div class="result-card">
                <p style="font-size: 1.1rem; line-height: 1.8;">{results['english_translation']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📥 Export English Translation")
            create_download_buttons(
                results['english_translation'],
                language='en',
                title="English Translation",
                file_prefix="summary_english"
            )

    with res_tab_stages:
        # Stage 1
        st.markdown("""
        <div class="stage-indicator">
            <h4 style="margin: 0; color: #667eea;">📖 Stage 1: Extractive Selection (TextRank)</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Identified Key Sentences:</p>
        </div>
        """, unsafe_allow_html=True)
        for idx, s in enumerate(results['extractive_list']):
            st.markdown(f"**{idx+1}.** {s}")

        st.markdown("---")

        # Stage 2
        st.markdown("""
        <div class="stage-indicator">
            <h4 style="margin: 0; color: #667eea;">🔍 Stage 2: Semantic Clustering</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Clustered Core Ideas:</p>
        </div>
        """, unsafe_allow_html=True)
        for idx, s in enumerate(results['clustered_list']):
            st.markdown(f"**{idx+1}.** {s}")

        st.markdown("---")

        # Stage 3
        st.markdown("""
        <div class="stage-indicator">
            <h4 style="margin: 0; color: #667eea;">✍️ Stage 3: Abstractive Ensemble Candidates</h4>
        </div>
        """, unsafe_allow_html=True)
        for model, candidate in results['candidates'].items():
            with st.expander(f"Candidate: {model}"):
                st.write(candidate if candidate else "_No output generated by this model._")

    with res_tab_metrics:
        st.markdown("""
        <div class="info-box">
            <strong>Why was this summary chosen?</strong><br>
            The system automatically scored all candidates against the Semantic Clustering reference using coverage, length adequacy, coherence, and overlap metrics.
        </div>
        """, unsafe_allow_html=True)

        if results.get('scores'):
            for model, scores in results['scores'].items():
                st.markdown(f"**Model:** `{model}`")

                # Score breakdown
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Final Score", f"{scores['final_score']:.4f}")
                with col2:
                    st.metric("ROUGE-L", f"{scores['raw_metrics'].get('rougeL', 0):.4f}")
                with col3:
                    st.metric("Semantic", f"{scores['raw_metrics'].get('semantic_coverage', 0):.4f}")
                with col4:
                    st.metric("Length", f"{scores['raw_metrics'].get('length_adequacy', 0):.4f}")
                with col5:
                    st.metric("Coherence", f"{scores['raw_metrics'].get('coherence', 0):.4f}")

                bert_score_value = scores['raw_metrics'].get('bert_score', 0)
                if bert_score_value:
                    st.caption(f"BERTScore: {bert_score_value:.4f}")

                st.divider()
        else:
            st.warning("No metrics available. The summary was generated using the extractive fallback.")

    # Button to clear results
    if st.button("🗑️ Clear Results", use_container_width=True):
        del st.session_state['results']
        if 'result_language' in st.session_state:
            del st.session_state['result_language']
        st.rerun()
