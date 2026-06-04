import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="ML Model Center",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Center")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# -----------------------------------------------------
# PREPARE DATA
# -----------------------------------------------------

X = df.drop("output", axis=1)
y = df["output"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------
# MODEL SELECTION
# -----------------------------------------------------

st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree"
    ]
)

# -----------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------

if model_name == "Logistic Regression":

    model = LogisticRegression(
        max_iter=1000
    )

elif model_name == "Random Forest":

    n_estimators = st.sidebar.slider(
        "Trees",
        50,
        500,
        200
    )

    max_depth = st.sidebar.slider(
        "Max Depth",
        2,
        20,
        8
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

else:

    max_depth = st.sidebar.slider(
        "Max Depth",
        2,
        20,
        6
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

# -----------------------------------------------------
# TRAINING
# -----------------------------------------------------

with st.spinner("Training model..."):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "Precision",
    f"{precision:.2%}"
)

col3.metric(
    "Recall",
    f"{recall:.2%}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)

col5.metric(
    "ROC AUC",
    f"{auc:.2%}"
)

# -----------------------------------------------------
# CONFUSION MATRIX
# -----------------------------------------------------

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions
)

fig = px.imshow(
    cm,
    text_auto=True,
    labels=dict(
        x="Predicted",
        y="Actual"
    ),
    title="Confusion Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# ROC CURVE
# -----------------------------------------------------

st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(
    y_test,
    probabilities
)

roc_df = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr
})

fig = px.line(
    roc_df,
    x="False Positive Rate",
    y="True Positive Rate",
    title=f"ROC Curve (AUC={auc:.3f})"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------

if hasattr(model, "feature_importances_"):

    st.subheader("🔥 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -----------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------

st.subheader("⚡ Compare Models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

comparison = []

for name, mdl in models.items():

    mdl.fit(X_train, y_train)

    pred = mdl.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    comparison.append([
        name,
        round(acc, 4)
    ])

comparison_df = pd.DataFrame(
    comparison,
    columns=[
        "Model",
        "Accuracy"
    ]
)

fig = px.bar(
    comparison_df,
    x="Model",
    y="Accuracy",
    text="Accuracy",
    title="Model Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# DOWNLOAD MODEL
# -----------------------------------------------------

st.subheader("💾 Save Trained Model")

joblib.dump(
    model,
    "heart_model.pkl"
)

with open(
    "heart_model.pkl",
    "rb"
) as file:

    st.download_button(
        label="Download Model",
        data=file,
        file_name="heart_model.pkl",
        mime="application/octet-stream"
    )

# -----------------------------------------------------
# AI INSIGHTS
# -----------------------------------------------------

st.subheader("🧠 AI Insights")

best_model = comparison_df.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]

st.success(
    f"""
    Best Performing Model: {best_model['Model']}

    Accuracy: {best_model['Accuracy']:.2%}

    Current Selected Model:
    {model_name}

    ROC-AUC Score:
    {auc:.2%}

    This model can be used for
    real-time patient risk prediction.
    """
)

# -----------------------------------------------------
# TEST PREDICTIONS
# -----------------------------------------------------

st.subheader("📋 Sample Predictions")

sample_results = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": predictions[:20]
})

st.dataframe(
    sample_results,
    use_container_width=True
)
