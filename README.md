# 📘 **Question Pair Similarity Classification**

Semantic Duplicate Detection using TF-IDF, ANN, Siamese LSTM, and SBERT Embeddings

---

## 📌 **Overview**

This project builds a complete machine learning pipeline for **duplicate question detection**.
Given a pair of questions, the goal is to classify whether they have the **same meaning**.

Models implemented:

* **TF-IDF + Logistic Regression (Baseline)**
* **Neural Network (ANN) with Embedding Layers**
* **Siamese LSTM Network**
* **Sentence-BERT (SBERT) Embeddings + PCA + MLP Classifier**
* **(Optional) Additional analysis plots + model comparisons**

The project includes preprocessing, feature extraction, model training, hyperparameter tuning, evaluation, and final comparison.

---

# 🧠 **Features**

* Text cleaning & preprocessing
* Tokenization + padding
* TF-IDF vectorization
* Sentence-BERT embeddings (fast with batch encoding)
* PCA feature reduction for high-dimensional SBERT vectors
* Multiple model training
* Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
* Comparison bar chart visualization

---

# 📂 **Project Structure**

```
/project
│
│── Classifier_Selection.ipynb      # main working notebook
│── Task Report.pdf                             
└── README.md
```

---

# ⚙️ **Setup Instructions**

## ✔️ **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

## ✔️ **2. Create Virtual Environment (Recommended)**

### **Using Conda**

```bash
conda create -n qpair python=3.10 -y
conda activate qpair
```

### **Using venv**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

---

## ✔️ **3. Install Dependencies**

Install Python packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow
pip install sentence-transformers
pip install nltk
```

(Notebook automatically downloads stopwords + SBERT model.)

---

# ▶️ **Running the Project**

## **Option A — Run Notebook in Jupyter**

```bash
jupyter notebook notebooks/Classifier_Selection.ipynb
```

Then run each cell in order.

---

## **Option B — Run in Google Colab**

1. Upload the notebook
2. Upload the dataset to `/content`
3. Run all cells
4. SBERT & PCA steps will automatically optimize speed

---

# 🧪 **Model Training Workflow**

The notebook automatically performs:

### 🔹 Preprocessing

* lowercase, punctuation removal, stopword removal
* tokenization, padding
* lemmatization

### 🔹 Feature Engineering

* TF-IDF features
* Embedding-based representations
* SBERT embeddings
* PCA for dimensionality reduction

### 🔹 Model Training

* Logistic Regression
* ANN
* Siamese LSTM
* SBERT → PCA → MLP

### 🔹 Evaluation

* classification report
* accuracy, precision, recall, F1
* confusion matrix
* model comparison bar chart

---

# 📊 **Results Summary**

(You may update with your actual scores.)

| Model                 | Accuracy  | F1        |
| --------------------- | --------- | --------- |
| TF-IDF + LR           | ~0.78     | ~0.75     |
| ANN                   | ~0.82     | ~0.80     |
| Siamese LSTM          | ~0.83     | ~0.82     |
| **SBERT + PCA + MLP** | **~0.86** | **~0.85** |

SBERT-based models perform the best due to deep semantic representation.

---

# 🚀 **Future Improvements**

* Fine-tuning SBERT end-to-end
* Using DistilBERT or RoBERTa + classification head
* Adding attention mechanisms in Siamese network
* Deploying via FastAPI/Streamlit
* Model compression for mobile deployment

---

# 🤝 **Contributing**

Pull requests are welcome!
Please open an issue first to discuss any major changes.

---

# 📄 **License**

This project is licensed under the MIT License.

---

# 🙌 **Acknowledgements**

* Sentence-Transformers by UKPLab
* Scikit-learn
* TensorFlow / Keras
* NLTK


