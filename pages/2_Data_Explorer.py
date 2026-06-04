import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 Data Explorer")
st.markdown("Explore, analyze, and understand the dataset.")

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# COLUMN INFORMATION
# --------------------------------------------------

st.subheader("📋 Feature Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

st.dataframe(info_df, use_container_width=True)

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

st.subheader("❗ Missing Value Analysis")

missing_df = pd.DataFrame({
    "Feature": df.columns,
    "Missing Count": df.isnull().sum().values
})

fig = px.bar(
    missing_df,
    x="Feature",
    y="Missing Count",
    title="Missing Values by Feature"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# STATISTICAL SUMMARY
# --------------------------------------------------

st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe().T,
    use_container_width=True
)

# --------------------------------------------------
# FEATURE DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Feature Distribution")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

selected_feature = st.selectbox(
    "Select Feature",
    numeric_columns
)

fig = px.histogram(
    df,
    x=selected_feature,
    nbins=25,
    title=f"{selected_feature} Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# BOXPLOT ANALYSIS
# --------------------------------------------------

st.subheader("📦 Outlier Detection")

box_feature = st.selectbox(
    "Select Feature for Boxplot",
    numeric_columns,
    key="boxplot"
)

fig = px.box(
    df,
    y=box_feature,
    title=f"{box_feature} Outlier Analysis"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# CORRELATION MATRIX
# --------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

corr = df.corr()

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

st.pyplot(fig)

# --------------------------------------------------
# TARGET ANALYSIS
# --------------------------------------------------

st.subheader("🎯 Target Variable Analysis")

target_counts = df["output"].value_counts()

fig = px.pie(
    values=target_counts.values,
    names=["No Disease", "Disease"],
    hole=0.5,
    title="Heart Disease Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# FEATURE VS TARGET
# --------------------------------------------------

st.subheader("⚡ Feature vs Target")

selected_feature_target = st.selectbox(
    "Select Feature for Comparison",
    numeric_columns,
    key="target_analysis"
)

fig = px.box(
    df,
    x="output",
    y=selected_feature_target,
    color="output",
    title=f"{selected_feature_target} vs Heart Disease"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# CORRELATION WITH TARGET
# --------------------------------------------------

st.subheader("🎯 Features Influencing Heart Disease")

target_corr = (
    df.corr()["output"]
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

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

st.subheader("⬇ Download Dataset")

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="heart_dataset.csv",
    mime="text/csv"
)

# --------------------------------------------------
# AUTO INSIGHTS
# --------------------------------------------------

st.subheader("🧠 Data Insights")

highest_corr = target_corr.iloc[1]["Feature"]
lowest_corr = target_corr.iloc[-1]["Feature"]

st.success(f"""
### Key Observations

✅ Total Patients: {len(df)}

✅ Features Available: {df.shape[1]}

✅ Missing Values Found: {df.isnull().sum().sum()}

✅ Strongest Positive Predictor:
**{highest_corr}**

✅ Strongest Negative Predictor:
**{lowest_corr}**

✅ Dataset is suitable for Machine Learning
Classification Models.

✅ Heart Disease prediction can be performed
using supervised learning algorithms.
""")
