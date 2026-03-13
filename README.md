# Hybrid Text Summarization System

Hybrid text summarization pipeline that combines:

- `TextRank` sentence extraction
- semantic clustering with `sentence-transformers`
- an abstractive ensemble using `mBART`, `mT5`, and `PEGASUS`
- metric-based meta-selection to choose the final summary

The repo currently supports:

- CLI summarization via `main.py`
- a Streamlit UI via `app.py`
- an optional FastAPI backend via `api_server.py`
- benchmark scripts for CNN/DailyMail and XSum

## Setup

Python `3.11` is the safest target for this project.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Large model downloads happen on first use. To pre-cache them:

```powershell
python download_models.py
```

## Run The App

Streamlit UI:

```powershell
streamlit run app.py
```

CLI:

```powershell
python main.py --input_file path\to\article.txt
python main.py --text "Your raw text here"
python main.py --input_file sample_gujarati.txt --language gu
```

Local API server:

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Remote Colab backend:

- Follow the steps in `COLAB_SETUP.md`
- Paste the public Colab/ngrok URL into the Streamlit sidebar

## Benchmarks

Text report:

```powershell
python run_benchmark.py --dataset cnn_dailymail --samples 10
```

JSON report:

```powershell
python test_benchmarks.py --dataset both --num_samples 10 --output_dir results
```

Quick dependency smoke check:

```powershell
python test_setup.py
```

## Notes

- The benchmark scripts are development tools, not fast unit tests.
- PDF export for non-Latin languages still depends on a compatible font file in `utils/fonts/`.
- The first full run can be slow because several transformer models and scorers are loaded lazily.
