<div align="center">

# 🔑 Keyword Extraction with Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF-green?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge"/>
</p>

<p align="center">
  An intelligent keyword extraction system built with <strong>TF-IDF (Term Frequency–Inverse Document Frequency)</strong> using scikit-learn.
  Automatically identifies the most relevant and significant keywords from any text — with pre-trained serialized models for instant inference.
</p>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Model Files](#-model-files)
- [Example Output](#-example-output)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

Keyword extraction is a core **Natural Language Processing (NLP)** task that surfaces the most meaningful terms in a piece of text — without the need for labelled training data. This project uses a **TF-IDF pipeline** trained and saved with scikit-learn, enabling:

- Fast keyword extraction on new documents
- Reproducible results via serialized `.pkl` model files
- Easy integration via `app.py` or the Jupyter Notebook

**Use cases:**
- 📰 News article summarization
- 🔍 SEO keyword analysis
- 📚 Document classification & tagging
- 🤖 Preprocessing for downstream NLP tasks

---

## ⚙️ How It Works

```
Raw Text Input
      │
      ▼
┌─────────────────────┐
│  CountVectorizer    │  ← Tokenizes text & builds term-document matrix
│  (count_vectorizer  │
│      .pkl)          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Transformer │  ← Weights terms by frequency & rarity across docs
│  (tfidf_transformer │
│      .pkl)          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Names      │  ← Maps indices back to actual words
│  (feature_names.pkl)│
└────────┬────────────┘
         │
         ▼
   Top-N Keywords (ranked by TF-IDF score)
```

**TF-IDF Formula:**

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

Where:
- `TF(t, d)` = frequency of term `t` in document `d`
- `N` = total number of documents
- `DF(t)` = number of documents containing term `t`

---

## 🗂️ Project Structure

```
Keyword-Extraction-with-python/
│
├── 📓 keyword_extraction.ipynb   # Full training pipeline & EDA notebook
├── 🐍 app.py                     # Inference script — run keyword extraction
│
├── 🤖 count_vectorizer.pkl       # Saved CountVectorizer model
├── 🤖 tfidf_transformer.pkl      # Saved TF-IDF Transformer model
├── 🤖 feature_names.pkl          # Saved vocabulary / feature names
│
├── 📄 LICENSE                    # Apache 2.0
└── 📄 README.md                  # You are here
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| `Python 3.8+` | Core language |
| `scikit-learn` | CountVectorizer + TF-IDF Transformer |
| `pickle` | Model serialization / deserialization |
| `Jupyter Notebook` | Experimentation & pipeline development |
| `NumPy` | Numerical operations |
| `Pandas` | Data handling |

---

## 📦 Installation

**1. Clone the repository:**
```bash
git clone https://github.com/eddiebrock911/Keyword-Extraction-with-python.git
cd Keyword-Extraction-with-python
```

**2. Create & activate a virtual environment:**
```bash
# Create
python -m venv venv

# Activate — Linux/Mac
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install scikit-learn numpy pandas jupyter
```

---

## ▶️ Usage

### Option 1 — Run `app.py`

```bash
python app.py
```

The script loads the pre-trained `.pkl` models and extracts keywords from your input text.

### Option 2 — Use in your own code

```python
import pickle
import numpy as np

# Load saved models
with open("count_vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

with open("tfidf_transformer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Extract keywords
def extract_keywords(text, top_n=10):
    tf_matrix = cv.transform([text])
    tfidf_matrix = tfidf.transform(tf_matrix)
    scores = tfidf_matrix.toarray()[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(feature_names[i], round(scores[i], 4)) for i in top_indices if scores[i] > 0]

text = "Machine learning models are trained on large datasets to make predictions."
keywords = extract_keywords(text, top_n=5)
print(keywords)
```

### Option 3 — Jupyter Notebook

```bash
jupyter notebook keyword_extraction.ipynb
```

---

## 📓 Notebook Walkthrough

The `keyword_extraction.ipynb` notebook covers:

| Step | Description |
|---|---|
| **1. Data Loading** | Load and inspect the text corpus |
| **2. Preprocessing** | Tokenization, stopword removal, lowercasing |
| **3. Vectorization** | Build term-document matrix with `CountVectorizer` |
| **4. TF-IDF Weighting** | Apply `TfidfTransformer` to score terms |
| **5. Keyword Extraction** | Rank and extract top-N keywords per document |
| **6. Model Saving** | Serialize models to `.pkl` for reuse |
| **7. Visualization** | Plot keyword scores (bar charts, word clouds) |

---

## 🤖 Model Files

Pre-trained models are included in the repository for **zero-setup inference**:

| File | Description |
|---|---|
| `count_vectorizer.pkl` | Fitted `CountVectorizer` — tokenizes & builds vocabulary |
| `tfidf_transformer.pkl` | Fitted `TfidfTransformer` — computes TF-IDF scores |
| `feature_names.pkl` | List of all vocabulary words mapped to matrix indices |

> ⚠️ These `.pkl` files must be loaded with the same version of scikit-learn used during training.

---

## 📋 Example Output

**Input:**
```
Deep learning is a subset of machine learning that uses neural networks 
with many layers to model complex patterns in data.
```

**Output:**
```
┌─────────────────────────┬─────────┐
│ Keyword                 │ Score   │
├─────────────────────────┼─────────┤
│ deep learning           │ 0.4821  │
│ neural networks         │ 0.4103  │
│ machine learning        │ 0.3897  │
│ complex patterns        │ 0.3542  │
│ model                   │ 0.2914  │
└─────────────────────────┴─────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repo
# 2. Create your branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "feat: add your feature"

# 4. Push to your branch
git push origin feature/your-feature-name

# 5. Open a Pull Request
```

Please follow [PEP 8](https://pep8.org/) style guidelines for Python code.

---

## 📄 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [eddiebrock911](https://github.com/eddiebrock911)

⭐ Star this repo if you found it helpful!

</div>
