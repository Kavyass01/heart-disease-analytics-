import pandas as pd
import streamlit as st

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data(filepath="data/heart.csv"):
    """
    Load dataset with Streamlit caching.
    """

    try:
        df = pd.read_csv(filepath)
        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


# ---------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------

def get_dataset_summary(df):
    """
    Returns dataset overview statistics.
    """

    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }

    return summary


# ---------------------------------------------------
# NUMERIC FEATURES
# ---------------------------------------------------

def get_numeric_columns(df):
    """
    Returns numeric columns.
    """

    return df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


# ---------------------------------------------------
# CATEGORICAL FEATURES
# ---------------------------------------------------

def get_categorical_columns(df):
    """
    Returns categorical columns.
    """

    return df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


# ---------------------------------------------------
# MISSING VALUE REPORT
# ---------------------------------------------------

def missing_value_report(df):
    """
    Returns missing value information.
    """

    report = pd.DataFrame({
        "Feature": df.columns,
        "Missing Values": df.isnull().sum(),
        "Missing %": (
            df.isnull().sum() /
            len(df)
        ) * 100
    })

    return report.sort_values(
        "Missing %",
        ascending=False
    )


# ---------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------

def get_correlation_matrix(df):
    """
    Returns correlation matrix.
    """

    return df.corr(numeric_only=True)


# ---------------------------------------------------
# TARGET CORRELATION
# ---------------------------------------------------

def target_correlation(
    df,
    target="output"
):
    """
    Correlation with target variable.
    """

    corr = (
        df.corr(numeric_only=True)[target]
        .sort_values(ascending=False)
        .reset_index()
    )

    corr.columns = [
        "Feature",
        "Correlation"
    ]

    return corr


# ---------------------------------------------------
# FEATURE STATISTICS
# ---------------------------------------------------

def feature_statistics(df):
    """
    Returns descriptive statistics.
    """

    return df.describe().T


# ---------------------------------------------------
# DATA QUALITY REPORT
# ---------------------------------------------------

def data_quality_report(df):
    """
    Generates quality report.
    """

    report = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique(),
        "Duplicates": df.duplicated().sum()
    })

    return report


# ---------------------------------------------------
# PREPARE ML DATA
# ---------------------------------------------------

def prepare_ml_data(
    df,
    target="output"
):
    """
    Splits dataset into X and y.
    """

    X = df.drop(
        target,
        axis=1
    )

    y = df[target]

    return X, y


# ---------------------------------------------------
# HEART DISEASE INSIGHTS
# ---------------------------------------------------

def get_health_insights(df):
    """
    Returns key insights from dataset.
    """

    insights = {
        "Average Age":
            round(df["age"].mean(), 1),

        "Average Cholesterol":
            round(df["chol"].mean(), 1),

        "Average Heart Rate":
            round(df["thalachh"].mean(), 1),

        "Disease Rate":
            round(
                df["output"].mean() * 100,
                2
            )
    }

    return insights


# ---------------------------------------------------
# FILTER DATA
# ---------------------------------------------------

def filter_data(
    df,
    min_age=None,
    max_age=None
):
    """
    Filter dataset by age.
    """

    filtered_df = df.copy()

    if min_age is not None:
        filtered_df = filtered_df[
            filtered_df["age"] >= min_age
        ]

    if max_age is not None:
        filtered_df = filtered_df[
            filtered_df["age"] <= max_age
        ]

    return filtered_df


# ---------------------------------------------------
# EXPORT CSV
# ---------------------------------------------------

def convert_to_csv(df):
    """
    Convert dataframe to CSV.
    """

    return df.to_csv(
        index=False
    ).encode("utf-8")
