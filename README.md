# 📩 Spam SMS Classification using Machine Learning

## 📌 Project Overview

This project focuses on detecting whether an SMS message is **Spam** or **Ham (Not Spam)** using Machine Learning techniques.

The dataset contains SMS messages labeled as either `spam` or `ham`. The messages are cleaned, analyzed, transformed into numerical features, and then used to train multiple machine learning classification models.

The main goal of this project is to build a reliable spam detection system and compare the performance of different machine learning algorithms.

---

## 🎯 Objectives

* Detect spam SMS messages automatically.
* Perform exploratory data analysis on SMS messages.
* Analyze message length, number of words, and sentences.
* Perform text preprocessing and feature extraction.
* Train multiple machine learning classification models.
* Compare models using accuracy, precision, and confusion matrix.
* Identify the best-performing model for spam detection.

---

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**.

Each message belongs to one of two categories:

* **Ham** → Legitimate SMS message
* **Spam** → Unwanted/promotional/fraudulent SMS message

The dataset contains two important columns:

| Column   | Description                    |
| -------- | ------------------------------ |
| `target` | Class label: 0 = Ham, 1 = Spam |
| `text`   | SMS message                    |

Additional features created during analysis include:

* `num_characters`
* `num_words`
* `num_sentence`

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* WordCloud
* XGBoost

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Text Preprocessing
   ↓
Feature Extraction
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
```

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset.
2. Handle encoding issues while reading the CSV file.
3. Select the required columns.
4. Rename columns for easier access.
5. Encode the target labels.
6. Check for missing values.
7. Remove duplicate messages.
8. Calculate message statistics.
9. Clean and transform text.
10. Convert text into numerical features.

Example:

```python
df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['target', 'text']
```

---

## 📈 Exploratory Data Analysis

The project analyzes:

* Distribution of spam and ham messages
* Number of characters in each message
* Number of words
* Number of sentences
* Relationship between different numerical features
* Most frequently occurring words

### ☁️ WordCloud

Word clouds are generated separately for spam and ham messages to identify frequently occurring words.

Example:

```python
wc.generate(
    df[df['target'] == 1]['text'].str.cat(sep=" ")
)

plt.imshow(wc)
plt.axis('off')
plt.show()
```

---

## 🤖 Machine Learning Models

The following classification algorithms are implemented and compared:

### 1. Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
```

Gaussian Naive Bayes is generally suitable for continuous numerical features.

### 2. Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
```

Multinomial Naive Bayes is particularly useful for text classification problems involving word counts or TF-IDF features.

### 3. Bernoulli Naive Bayes

```python
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
```

Bernoulli Naive Bayes works well with binary features, such as whether a particular word is present or absent.

### 4. Extra Trees Classifier

```python
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
```

Extra Trees is an ensemble learning algorithm based on multiple randomized decision trees.

### 5. Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
```

Gradient Boosting builds models sequentially, with each new model attempting to improve upon the errors of previous models.

### 6. XGBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier()
```

XGBoost is a powerful gradient boosting algorithm commonly used for classification and other machine learning tasks.

---

## 📏 Model Evaluation

The models are evaluated using:

### Accuracy

Accuracy measures the percentage of correctly classified messages.

```python
accuracy_score(y_test, y_pred)
```

### Precision

Precision measures how many messages predicted as spam are actually spam.

```python
precision_score(y_test, y_pred)
```

### Confusion Matrix

The confusion matrix shows:

```text
                 Predicted
              Ham       Spam

Actual Ham     TN        FP
Actual Spam    FN        TP
```

It helps identify false positives and false negatives.

Example:

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
```

---

## 📊 Model Comparison

The trained models are compared based on their evaluation metrics.

| Model                   | Accuracy | Precision |
| ----------------------- | -------: | --------: |
| Gaussian Naive Bayes    |        — |         — |
| Multinomial Naive Bayes |        — |         — |
| Bernoulli Naive Bayes   |        — |         — |
| Extra Trees             |        — |         — |
| Gradient Boosting       |        — |         — |
| XGBoost                 |        — |         — |

> Replace the `—` values with the actual results obtained from the notebook.

---

## 📁 Project Structure

```text
Spam-SMS-Classification/
│
├── spam.csv
├── spam_detection.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd Spam-SMS-Classification
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud xgboost
```

Or install them using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Install Python.
2. Install the required dependencies.
3. Open Jupyter Notebook.

```bash
jupyter notebook
```

4. Open `spam_detection.ipynb`.
5. Run the cells sequentially.
6. Analyze the model performance.
7. Compare the classification algorithms.

---

## 💡 Key Learnings

Through this project, the following concepts were explored:

* Data cleaning and preprocessing
* Exploratory Data Analysis
* Text preprocessing
* Feature engineering
* WordCloud visualization
* Naive Bayes classification
* Ensemble learning
* Gradient Boosting
* XGBoost
* Model evaluation
* Accuracy and precision
* Confusion matrix
* Model comparison

---

## 🚀 Future Improvements

The project can be improved by:

* Using advanced NLP techniques.
* Experimenting with TF-IDF and n-gram features.
* Using word embeddings such as Word2Vec or GloVe.
* Applying deep learning models such as LSTM or Transformers.
* Building a web interface for real-time SMS classification.
* Deploying the final model using Flask, FastAPI, or Streamlit.

---

## 👩‍💻 Author

**Shruti**

Machine Learning / Data Science Project

---

## 📜 License

This project is intended for educational and academic purposes.
