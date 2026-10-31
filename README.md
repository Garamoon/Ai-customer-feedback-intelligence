# 🛍️ AI Customer Feedback Intelligence System

> End-to-end NLP project that classifies customer sentiment, discovers recurring complaint themes, and serves everything through a live **Gradio** web app.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/AI_Customer_Feedback_Gradio_Deployment.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Gradio](https://img.shields.io/badge/Deployment-Gradio-ff7c00)
![License](https://img.shields.io/badge/License-MIT-green)

Graduation Project (Project 1) — **Artificial Intelligence Professional Diploma**.

<!-- After running the notebook, take a screenshot of the Gradio app, save it as images/gradio_demo.png, then uncomment the line below:
![Gradio demo](images/gradio_demo.png)
-->

---

## 📌 Problem

Online retailers collect thousands of written reviews. Reading each one to understand sentiment and root causes is slow, inconsistent, and doesn't scale. This project builds an AI system that:

1. **Classifies** each review as Positive / Neutral / Negative
2. **Clusters** negative reviews into complaint categories (sizing, fabric quality, shipping, price)
3. **Generates business insights** from the results
4. **Deploys** the model as an interactive web app

## 🚀 Quick Start

The whole project (data → training → deployment) lives in **one notebook**.

### Option A — Google Colab (easiest, gives a public link)
1. Click the **Open in Colab** badge above
2. **Runtime → Run all**
3. The last cell prints a public link like `https://xxxxx.gradio.live` — open it to use the app

### Option B — Run locally
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
jupyter notebook AI_Customer_Feedback_Gradio_Deployment.ipynb
```
Run all cells; the final cell launches the Gradio app. Open the local URL it prints.

## 🧠 Methodology

| Stage | What was done |
|---|---|
| **Data preparation** | Removed duplicates and missing values, lowercased text, stripped punctuation/numbers, tokenized, removed stopwords |
| **Labeling** | Rating 4–5 → Positive, 3 → Neutral, 1–2 → Negative |
| **NLP features** | TF-IDF (unigrams + bigrams, top 3,000 terms) |
| **Feature engineering** *(bonus)* | Review length, word count, exclamation count, positive/negative lexicon counts |
| **Classification** | Logistic Regression, Random Forest, plus Naive Bayes and Linear SVM *(bonus comparison)* |
| **Unsupervised learning** | K-Means (k=4) on TF-IDF of negative reviews, visualized with PCA |
| **Deployment** *(bonus)* | Gradio app with 3 tabs: Sentiment Classifier, Complaint Cluster Explorer, Insights Dashboard |

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.967 | 0.970 | 0.967 | 0.967 |
| Random Forest | 0.969 | 0.968 | 0.969 | 0.969 |
| Naive Bayes | 0.955 | 0.955 | 0.955 | 0.953 |
| **Linear SVM** | **0.981** | **0.981** | **0.981** | **0.981** |

<p align="center">
  <img src="images/model_comparison.png" width="48%" />
  <img src="images/confusion_matrix.png" width="40%" />
</p>

### Complaint clusters (K-Means on negative reviews)

| Cluster | Theme | Top terms |
|---|---|---|
| 0 | Price vs. quality | expensive, basic, material |
| 1 | Shipping & packaging | box, crushed, delayed, arrival |
| 2 | Sizing & fit | runs, small, return |
| 3 | Fabric quality | cheap, poor quality |

<p align="center">
  <img src="images/complaint_clusters.png" width="55%" />
</p>

### Exploratory analysis

<p align="center">
  <img src="images/sentiment_distribution.png" width="32%" />
  <img src="images/rating_distribution.png" width="32%" />
  <img src="images/common_words.png" width="32%" />
</p>

## 💡 Business Insights

- ~72% of reviews are positive and ~77% of customers recommend the product — a strong baseline to protect.
- Negative feedback concentrates around **sizing/fit** and **fabric quality**, giving buying and QA teams concrete priorities.
- The pipeline can run continuously on new reviews, turning a manual audit into a live monitoring feed.

## ⚠️ Dataset Note (please read)

The project targets the structure of the public
[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) dataset.
This version generates a **synthetic dataset with the same columns** (Review Text, Rating, Recommended IND, Division/Department/Class Name, Age, …) inside the notebook, so the project runs anywhere with no downloads.

Because the synthetic text is template-based, **the reported scores are much higher than what you should expect on real-world reviews**. To use the real data: download the CSV from Kaggle, save it as `data/reviews_raw.csv`, skip the data-generation cell, and re-run.

## 📁 Repository Structure

```
├── AI_Customer_Feedback_Gradio_Deployment.ipynb   # Full pipeline + Gradio app (single file)
├── requirements.txt
├── images/                                        # Charts used in this README
├── docs/
│   ├── AI_Customer_Feedback_Report.docx           # Project report
│   └── AI_Customer_Feedback_Presentation.pptx     # Presentation slides
├── LICENSE
└── README.md
```

## 🔭 Future Work

- Train on the full real Kaggle dataset and compare results
- Fine-tune a transformer model (e.g. DistilBERT) for sarcasm and mixed sentiment
- Permanent deployment on Hugging Face Spaces
- Drift monitoring as new product lines launch

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Gradio

## 👤 Author

**YOUR NAME** — [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

## 📄 License

Released under the [MIT License](LICENSE).
