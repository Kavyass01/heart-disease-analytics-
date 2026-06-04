import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("❤️ Heart Disease Analytics Dashboard")
st.markdown("---")

# ---------------------------------------------------
# KPI Cards
# ---------------------------------------------------

total_patients = len(df)
disease_cases = df["output"].sum()
healthy_cases = len(df[df["output"] == 0])
disease_rate = round((disease_cases / total_patients) * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Patients",
        value=f"{total_patients}"
    )

with col2:
    st.metric(
        label="Heart Disease Cases",
        value=f"{disease_cases}"
    )

with col3:
    st.metric(
        label="Healthy Patients",
        value=f"{healthy_cases}"
    )

with col4:
    st.metric(
        label="Disease Rate",
        value=f"{disease_rate}%"
    )

st.markdown("---")

# ---------------------------------------------------
# Charts Row 1
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        df,
        names="output",
        title="Heart Disease Distribution",
        hole=0.5
    )

    fig.update_traces(
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        df,
        x="age",
        color="output",
        nbins=20,
        title="Age Distribution by Disease Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Charts Row 2
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

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
        title="Heart Disease Rate by Gender",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        x="output",
        y="chol",
        color="output",
        title="Cholesterol Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------

st.subheader("📊 Correlation Analysis")

corr_matrix = df.corr()

fig, ax = plt.subplots(
    figsize=(12, 8)
)

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig)

# ---------------------------------------------------
# Risk Analysis
# ---------------------------------------------------

st.subheader("⚠️ Risk Factor Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        df,
        x="age",
        y="thalachh",
        color="output",
        size="chol",
        title="Age vs Maximum Heart Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    cp_analysis = (
        df.groupby("cp")["output"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        cp_analysis,
        x="cp",
        y="output",
        title="Chest Pain Type vs Disease Risk",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Key Insights
# ---------------------------------------------------

st.subheader("🧠 Automated Insights")

avg_age = round(df["age"].mean(), 1)
avg_chol = round(df["chol"].mean(), 1)
avg_hr = round(df["thalachh"].mean(), 1)

st.info(f"""
### Key Findings

✅ Average Patient Age: {avg_age} years

✅ Average Cholesterol Level: {avg_chol}

✅ Average Maximum Heart Rate: {avg_hr}

✅ Heart Disease Prevalence: {disease_rate}%

✅ Age, Chest Pain Type, and Maximum Heart Rate
appear among the strongest indicators.

✅ Cholesterol alone is not the most influential factor.

✅ Higher age groups show increased disease occurrence.
""")

# ---------------------------------------------------
# Raw Data
# ---------------------------------------------------

with st.expander("📄 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )
