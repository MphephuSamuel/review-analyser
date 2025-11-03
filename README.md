# review-analyser

Lightweight toolkit for analyzing textual reviews — sentiment, simple topic extraction, and metadata extraction. This repository provides data preprocessing utilities, model training/evaluation scripts, and example notebooks to help you analyze customer/product reviews quickly.

Status: Prototype / experimental

Table of contents
- About
- Features
- Quick Start
- Examples
- Installation
- Usage
- Project structure
- Testing & CI
- Contributing
- License
- Contact

About
-----
review-analyser is a small, modular project to help researchers and engineers preprocess, explore, and build lightweight models for review text analysis (sentiment classification, topic/tag extraction, and simple metadata parsing). The goal is to be easy to run locally, easy to extend, and friendly for contributors.

Features
--------
- Data loaders for common CSV/JSON review formats
- Text preprocessing pipeline (normalization, tokenization, stop-words)
- Baseline sentiment classifier and evaluation scripts
- Utilities for simple topic or keyword extraction
- Example notebook and demo script for quick experiments

Quick start
-----------
Clone the repo, create a virtual environment, install dependencies, and run the demo:

1) Clone
   git clone https://github.com/MphephuSamuel/review-analyser.git
   cd review-analyser

2) Create env & install
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt

3) Run demo
   python examples/run_demo.py --input examples/sample_reviews.csv --output out/results.json

If you prefer Docker:
   docker build -t review-analyser .
   docker run --rm -v $(pwd)/examples:/app/examples review-analyser python examples/run_demo.py --input examples/sample_reviews.csv

Installation
------------
- Python 3.8+
- Install from requirements.txt: pip install -r requirements.txt
- (Optional) Use the provided Dockerfile for reproducible environments.

Usage
-----
Basic CLI examples:
- Preprocess a CSV of reviews:
  python bin/preprocess.py --input data/reviews.csv --output data/preprocessed.csv

- Train a baseline model:
  python bin/train.py --data data/preprocessed.csv --model out/baseline_model.pkl

- Predict with a trained model:
  python bin/predict.py --model out/baseline_model.pkl --input examples/sample_reviews.csv --output out/predictions.csv

API example (Python):
```python
from review_analyser import loader, preprocess, model

df = loader.load_csv("examples/sample_reviews.csv")
df_clean = preprocess.clean_text(df, text_column="review")
clf = model.train_baseline(df_clean, label_column="label")
preds = clf.predict(df_clean["review"])
