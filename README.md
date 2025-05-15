# 🧠 Autism Prediction Model

A machine learning-based predictive system for the early detection of Autism Spectrum Disorder (ASD). This model leverages key behavioral and clinical features from a dataset of 1,000 individuals to provide accurate predictions that could support early intervention.

---

## 📌 Project Overview

Early diagnosis of Autism is crucial for timely support and intervention. This project focuses on building a robust machine learning model to classify whether an individual is likely to have Autism based on structured questionnaire data and clinical indicators.

---

## 🎯 Objectives

- Predict the likelihood of Autism using ML classifiers.
- Evaluate and compare multiple models.
- Maximize accuracy while maintaining model interpretability.

---

## 🧪 Dataset

- **Size**: 1,000 records
- **Source**: Publicly available Autism Screening datasets
- **Features**: Includes age, gender, ethnicity, questionnaire responses (Q1–Q10), family history, and more.
- **Target**: `ASD` (binary classification – Yes/No)

---

## 🛠️ Tech Stack

### 📊 Machine Learning:
- `Scikit-learn` – Core ML library used for training and evaluation
- `Random Forest Classifier`
- `Decision Tree Classifier`
- `Naive Bayes Classifier`

### 🧹 Data Processing:
- `Pandas`, `NumPy` – For data handling and preprocessing
- `Matplotlib`, `Seaborn` – For visualizing model performance and feature importance

---

## 🚀 Implementation Steps

1. **Data Cleaning** – Handle missing values, encode categorical variables.
2. **Exploratory Data Analysis (EDA)** – Visualize correlations and distributions.
3. **Model Training** – Apply and compare three models: Random Forest, Decision Tree, and Naive Bayes.
4. **Evaluation Metrics** – Accuracy, Confusion Matrix, Precision, Recall, and F1 Score.
5. **Best Model** – Random Forest achieved the highest accuracy of **85%**.

---

## 📈 Results

|      Model      | Train Accuracy |
|-----------------|----------------|
| Random Forest   |  **92%**       |
| Decision Tree   |    86%         |
| XG boost        |    90%         |

Overall Test Accuracy:83%(Random Forest Classifier)
---

## 📊 Sample Output

- Feature Importance Plot (from Random Forest)
- Confusion Matrix and Classification Report
- Accuracy Comparison Chart

*(Add plots and figures in your GitHub repo for better visuals)*

---

## 💡 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Deep learning implementation using TensorFlow or PyTorch
- Integration with a user-friendly web interface (Flask/Streamlit)
- Real-time API endpoint for prediction

---

## 📂 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/autism-prediction-model.git
cd autism-prediction-model

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
