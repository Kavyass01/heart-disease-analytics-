import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

def disease_distribution_chart(df):

    fig = px.pie(
        df,
        names="output",
        title="Heart Disease Distribution",
        hole=0.5,
        color="output"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    return fig


# --------------------------------------------------
# AGE DISTRIBUTION
# --------------------------------------------------

def age_distribution_chart(df):

    fig = px.histogram(
        df,
        x="age",
        color="output",
        nbins=20,
        title="Age Distribution by Disease Status"
    )

    return fig


# --------------------------------------------------
# GENDER ANALYSIS
# --------------------------------------------------

def gender_analysis_chart(df):

    gender_df = (
        df.groupby("sex")["output"]
        .mean()
        .reset_index()
    )

    gender_df["sex"] = gender_df["sex"].replace(
        {
            0: "Female",
            1: "Male"
        }
    )

    fig = px.bar(
        gender_df,
        x="sex",
        y="output",
        title="Disease Rate by Gender",
        text_auto=".2f"
    )

    return fig


# --------------------------------------------------
# CHOLESTEROL ANALYSIS
# --------------------------------------------------

def cholesterol_boxplot(df):

    fig = px.box(
        df,
        x="output",
        y="chol",
        color="output",
        title="Cholesterol Distribution"
    )

    return fig


# --------------------------------------------------
# HEART RATE ANALYSIS
# --------------------------------------------------

def heart_rate_chart(df):

    fig = px.scatter(
        df,
        x="age",
        y="thalachh",
        color="output",
        size="chol",
        title="Age vs Maximum Heart Rate"
    )

    return fig


# --------------------------------------------------
# CHEST PAIN ANALYSIS
# --------------------------------------------------

def chest_pain_chart(df):

    cp_df = (
        df.groupby("cp")["output"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        cp_df,
        x="cp",
        y="output",
        title="Chest Pain Type vs Disease Risk",
        text_auto=".2f"
    )

    return fig


# --------------------------------------------------
# HISTOGRAM GENERATOR
# --------------------------------------------------

def feature_histogram(df, feature):

    fig = px.histogram(
        df,
        x=feature,
        nbins=25,
        title=f"{feature} Distribution"
    )

    return fig


# --------------------------------------------------
# BOXPLOT GENERATOR
# --------------------------------------------------

def feature_boxplot(df, feature):

    fig = px.box(
        df,
        y=feature,
        title=f"{feature} Outlier Analysis"
    )

    return fig


# --------------------------------------------------
# FEATURE VS TARGET
# --------------------------------------------------

def feature_vs_target(df, feature):

    fig = px.box(
        df,
        x="output",
        y=feature,
        color="output",
        title=f"{feature} vs Heart Disease"
    )

    return fig


# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

def correlation_heatmap(df):

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    return fig


# --------------------------------------------------
# TARGET CORRELATION
# --------------------------------------------------

def target_correlation_chart(df):

    target_corr = (
        df.corr(numeric_only=True)["output"]
        .sort_values(ascending=False)
        .reset_index()
    )

    target_corr.columns = [
        "Feature",
        "Correlation"
    ]

    fig = px.bar(
        target_corr,
        x="Feature",
        y="Correlation",
        title="Feature Correlation with Target"
    )

    return fig


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

def feature_importance_chart(model, features):

    importance_df = pd.DataFrame({
        "Feature": features,
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

    return fig


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

def roc_curve_chart(fpr, tpr, auc):

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {auc:.3f})"
    )

    fig.add_shape(
        type="line",
        line=dict(dash="dash"),
        x0=0,
        y0=0,
        x1=1,
        y1=1
    )

    return fig


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def confusion_matrix_chart(cm):

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Predicted",
            y="Actual"
        ),
        title="Confusion Matrix"
    )

    return fig


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

def model_comparison_chart(df):

    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Model Comparison"
    )

    return fig


# --------------------------------------------------
# KPI CARD HTML
# --------------------------------------------------

def create_kpi_card(title, value):

    return f"""
    <div style="
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.1);
        text-align:center;
    ">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """
