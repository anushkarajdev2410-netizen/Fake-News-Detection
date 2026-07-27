# Fake News Detection Using Advanced NLP

> A content-based fake news classifier comparing TF-IDF and Word2Vec feature representations across Naive Bayes, Support Vector Machine, and Random Forest models, deployed as an interactive Streamlit app.

**Repository description (for GitHub "About"):** Classical ML + NLP pipeline for fake news detection \u2014 TF-IDF & Word2Vec features, Naive Bayes/SVM/Random Forest comparison, and a Streamlit demo app.

---

## Overview

This project builds a complete, reproducible pipeline that takes raw news article text and predicts whether it is **real** or **fake**, based purely on linguistic and statistical patterns in the text \u2014 no external fact-checking database is consulted. It was built as an internship project in Artificial Intelligence and Data Science, and is organized to industry-repository standard for portfolio use.

**What it does, end to end:**
1. Loads and cleans the ISOT Fake and Real News dataset
2. Extracts features two ways \u2014 TF-IDF (sparse, word-frequency based) and Word2Vec (dense, semantic embeddings)
3. Trains and compares Naive Bayes, SVM, and Random Forest on both feature sets (six models total)
4. Evaluates every model on accuracy, precision, recall, F1, and ROC-AUC
5. Saves the best-performing model and serves it through a Streamlit web app

## Project Structure

```
Fake-News-Detection/
├── dataset/           # True.csv, Fake.csv (download separately, see below), and generated intermediate files
├── notebooks/         # 01_EDA -> 08_Save_Best_Model, run in order
├── reports/           # Research report, action plan, model comparison results
├── models/            # Saved model, vectorizer/Word2Vec model, metadata
├── webapp/            # Streamlit application (app.py)
├── images/            # Saved plots and visualizations
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

**Prerequisites:** Python 3.10 or later, and a free [Kaggle](https://www.kaggle.com) account to download the dataset.

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Fake-News-Detection.git
   cd Fake-News-Detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from Kaggle \u2014 search "Fake and Real News Dataset" \u2014 and place `True.csv` and `Fake.csv` in `dataset/`.

## Usage

**Run the notebooks in order** (each one saves the artifacts the next one needs):

```
notebooks/01_EDA.ipynb
notebooks/02_Text_Preprocessing.ipynb
notebooks/03_TFIDF_Features.ipynb
notebooks/04_Word2Vec_Features.ipynb
notebooks/05_Train_Test_Split.ipynb
notebooks/06_Model_Training.ipynb
notebooks/07_Model_Evaluation.ipynb
notebooks/08_Save_Best_Model.ipynb
```

**Launch the web app** once `models/best_model.pkl` exists:

```bash
cd webapp
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), paste in article text, and click **Analyze Article**.

## Methodology Summary

The pipeline follows the standard NLP workflow: clean and normalize text (including removing a wire-service dateline pattern specific to this dataset, to avoid the model learning a source-formatting shortcut instead of genuine content signal), extract features via TF-IDF and Word2Vec, train three classical algorithms on both representations, and select the best of the six by F1-score on a held-out, stratified 80/20 test split. Full reasoning for every methodological choice is documented inline in each notebook and in the accompanying research report.

## Results

See `reports/model_comparison_results.csv` and the confusion matrices / ROC curves saved to `images/` after running `07_Model_Evaluation.ipynb` for the actual, data-driven results and the model recommendation produced from them.

## Limitations

- Trained and evaluated on a single English-language, U.S.-centric dataset; may not generalize to other languages, regions, or time periods
- Detects writing-style patterns associated with fake news in this dataset, not factual accuracy \u2014 it is a screening signal, not a fact-checking authority
- The real/fake split in the source dataset draws from a narrower set of outlets than a fully general news stream, so some of what the model learns may reflect source-specific style rather than fakeness as a general property (see the research report's Ethical Considerations section)
- Word2Vec vectors are trained on this project's corpus alone; a pretrained embedding (e.g. GloVe) might generalize better to unfamiliar vocabulary

## Future Scope

- Incorporate transformer-based models (e.g. BERT) for contextual understanding beyond bag-of-words and averaged embeddings
- Extend to multilingual datasets and non-U.S. news sources
- Add social-context features (sharing patterns, source reputation) alongside the current purely content-based approach
- Explore TF-IDF-weighted Word2Vec averaging as a lightweight improvement over simple mean-pooling

## References

Ahmed, H., Traore, I., & Saad, S. (2017). Detection of online fake news using N-gram analysis and machine learning techniques. *Proceedings of the International Conference on Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments*, 127\u2013138.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.

Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825\u20132830.

Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD Explorations Newsletter*, 19(1), 22\u201336.

Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. *ACM Computing Surveys*, 53(5), 1\u201340.

## License

This project is for educational and portfolio purposes. Add a license (e.g. MIT) here if you intend to open-source it.
