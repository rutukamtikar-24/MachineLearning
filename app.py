import streamlit as st
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost import train_xgboost


# ===============================
# Helper: Load uploaded test data
# ===============================
def load_uploaded_test_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    if "url" in data.columns:
        data = data.drop(columns=["url"])

    X_test = data.drop(columns=["status"])
    y_test = data["status"].map({"legitimate": 0, "phishing": 1})

    return X_test, y_test


# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Phishing Website Detection",
    layout="centered"
)

st.title("🔐 Phishing Website Detection")
st.write("Machine Learning Assignment 2 – Model Evaluation Dashboard")

st.markdown("---")

# ===============================
# Dataset Upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ===============================
# Model Selection
# ===============================
model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]
)

st.markdown("---")

# ===============================
# Run Model
# ===============================
if st.button("Run Model"):

    if uploaded_file is not None:
        X_ext, y_ext = load_uploaded_test_data(uploaded_file)
    else:
        X_ext, y_ext = None, None

    # ===============================
    # Model Execution (UNCHANGED RETURNS)
    # ===============================
    if model_name == "Logistic Regression":
        metrics, y_test, y_pred = train_logistic_regression(X_ext, y_ext)

    elif model_name == "Decision Tree":
        metrics, y_test, y_pred = train_decision_tree(X_ext, y_ext)

    elif model_name == "KNN":
        metrics, y_test, y_pred = train_knn(X_ext, y_ext)

    elif model_name == "Naive Bayes":
        metrics, y_test, y_pred = train_naive_bayes(X_ext, y_ext)

    elif model_name == "Random Forest":
        metrics, y_test, y_pred = train_random_forest(X_ext, y_ext)

    elif model_name == "XGBoost":
        metrics, y_test, y_pred = train_xgboost(X_ext, y_ext)

    # ===============================
    # KPI METRICS (ALL SHOWN CLEARLY)
    # ===============================
    st.markdown("## 📊 Key Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    col2.metric("AUC", f"{metrics['AUC']:.3f}")
    col3.metric("Precision", f"{metrics['Precision']:.3f}")

    col4.metric("Recall", f"{metrics['Recall']:.3f}")
    col5.metric("F1 Score", f"{metrics['F1']:.3f}")
    col6.metric("MCC", f"{metrics['MCC']:.3f}")

    st.markdown("---")

    # ===============================
    # Confusion Matrix
    # ===============================
    st.markdown("## 🔢 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Phishing"],
        columns=["Predicted Legitimate", "Predicted Phishing"]
    )
    st.dataframe(cm_df)

    # ===============================
    # Classification Report (ALIGNED)
    # ===============================
    st.markdown("## 📄 Classification Report")

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Phishing"],
            output_dict=True
        )
    ).transpose().round(3)

    st.dataframe(report_df)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Developed for ML Assignment 2 | BITS Pilani")
