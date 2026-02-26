# Hybrid Text Summarization System

A robust text summarization system combining extractive (TextRank) and abstractive (Transformer Ensemble) methods, with support for English and Gujarati.

## Features
- **Hybrid Approach**: Filters key sentences first, then generates a concise abstractive summary.
- **Multilingual**: Supports English and Indian languages (Gujarati, Hindi, etc.).
- **Ensemble Models**: Uses mBART, mT5, and PEGASUS for high-quality generation.
- **Meta-Selection**: Automatically picks the best summary based on quality metrics.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK Data** (First run only):
    The script will automatically download necessary NLTK data.

## Usage

### English Summarization
```bash
python main.py --input_file path/to/article.txt
```

### Gujarati Summarization
```bash
python main.py --input_file sample_gujarati.txt --language gu
```

### Options
- `--input_file`: Path to the text file.
- `--text`: Raw text string (alternative to file).
- `--top_n`: Number of sentences for extractive step (default: 5).
- `--clusters`: Number of semantic clusters (default: 3).
- `--max_length`: Max length of the generated summary (default: 150).
- `--language`: Language code (`en`, `gu`, `hi`, etc.) (default: `en`).
