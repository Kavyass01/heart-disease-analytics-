import pandas as pd


# --------------------------------------------------
# OVERALL DATASET INSIGHTS
# --------------------------------------------------

def generate_dataset_insights(df):

    total_patients = len(df)

    disease_cases = int(df["output"].sum())

    healthy_cases = total_patients - disease_cases

    disease_rate = round(
        (disease_cases / total_patients) * 100,
        2
    )

    insights = {
        "Total Patients": total_patients,
        "Disease Cases": disease_cases,
        "Healthy Cases": healthy_cases,
        "Disease Rate (%)": disease_rate
    }

    return insights


# --------------------------------------------------
# AGE INSIGHTS
# --------------------------------------------------

def age_insights(df):

    avg_age = round(df["age"].mean(), 1)

    disease_age = round(
        df[df["output"] == 1]["age"].mean(),
        1
    )

    healthy_age = round(
        df[df["output"] == 0]["age"].mean(),
        1
    )

    return {
        "Average Age": avg_age,
        "Disease Group Avg Age": disease_age,
        "Healthy Group Avg Age": healthy_age
    }


# --------------------------------------------------
# GENDER INSIGHTS
# --------------------------------------------------

def gender_insights(df):

    male_rate = round(
        df[df["sex"] == 1]["output"].mean() * 100,
        2
    )

    female_rate = round(
        df[df["sex"] == 0]["output"].mean() * 100,
        2
    )

    return {
        "Male Disease Rate (%)": male_rate,
        "Female Disease Rate (%)": female_rate
    }


# --------------------------------------------------
# CHOLESTEROL INSIGHTS
# --------------------------------------------------

def cholesterol_insights(df):

    avg_chol = round(
        df["chol"].mean(),
        1
    )

    disease_chol = round(
        df[df["output"] == 1]["chol"].mean(),
        1
    )

    healthy_chol = round(
        df[df["output"] == 0]["chol"].mean(),
        1
    )

    return {
        "Average Cholesterol": avg_chol,
        "Disease Group Cholesterol": disease_chol,
        "Healthy Group Cholesterol": healthy_chol
    }


# --------------------------------------------------
# HEART RATE INSIGHTS
# --------------------------------------------------

def heart_rate_insights(df):

    avg_hr = round(
        df["thalachh"].mean(),
        1
    )

    disease_hr = round(
        df[df["output"] == 1]["thalachh"].mean(),
        1
    )

    healthy_hr = round(
        df[df["output"] == 0]["thalachh"].mean(),
        1
    )

    return {
        "Average Heart Rate": avg_hr,
        "Disease Group HR": disease_hr,
        "Healthy Group HR": healthy_hr
    }


# --------------------------------------------------
# TOP RISK FACTORS
# --------------------------------------------------

def top_risk_factors(df):

    correlations = (
        df.corr(numeric_only=True)["output"]
        .drop("output")
        .abs()
        .sort_values(ascending=False)
    )

    top_factors = correlations.head(5)

    return top_factors.to_dict()


# --------------------------------------------------
# EXERCISE ANGINA INSIGHTS
# --------------------------------------------------

def exercise_angina_insights(df):

    risk = round(
        df.groupby("exng")["output"]
        .mean()
        .max() * 100,
        2
    )

    return {
        "Exercise Angina Highest Risk (%)": risk
    }


# --------------------------------------------------
# CHEST PAIN INSIGHTS
# --------------------------------------------------

def chest_pain_insights(df):

    cp = (
        df.groupby("cp")["output"]
        .mean()
        .sort_values(ascending=False)
    )

    return cp.to_dict()


# --------------------------------------------------
# TARGET CORRELATION INSIGHTS
# --------------------------------------------------

def target_correlation_insights(df):

    corr = (
        df.corr(numeric_only=True)["output"]
        .sort_values(ascending=False)
    )

    return corr.to_dict()


# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

def executive_summary(df):

    disease_rate = round(
        df["output"].mean() * 100,
        2
    )

    avg_age = round(
        df["age"].mean(),
        1
    )

    avg_chol = round(
        df["chol"].mean(),
        1
    )

    avg_hr = round(
        df["thalachh"].mean(),
        1
    )

    summary = f"""
HEART DISEASE ANALYTICS SUMMARY

• Total Patients: {len(df)}

• Disease Prevalence: {disease_rate}%

• Average Age: {avg_age} years

• Average Cholesterol: {avg_chol}

• Average Heart Rate: {avg_hr}

• Age, Chest Pain Type, Exercise Angina,
  and Maximum Heart Rate are among the
  strongest indicators of heart disease.

• Higher disease occurrence is observed
  among older patient groups.

• Machine learning models can effectively
  predict heart disease risk using available
  clinical attributes.
"""

    return summary


# --------------------------------------------------
# AI RECOMMENDATIONS
# --------------------------------------------------

def ai_recommendations(df):

    recommendations = [
        "Monitor patients above average age group.",
        "Evaluate chest pain symptoms carefully.",
        "Track exercise-induced angina indicators.",
        "Review heart rate abnormalities.",
        "Use predictive models for early screening.",
        "Conduct periodic cardiovascular assessments."
    ]

    return recommendations


# --------------------------------------------------
# DASHBOARD INSIGHT CARD
# --------------------------------------------------

def dashboard_insight_card(df):

    disease_rate = round(
        df["output"].mean() * 100,
        2
    )

    avg_age = round(
        df["age"].mean(),
        1
    )

    return f"""
🧠 Dashboard Insight

Heart Disease Rate: {disease_rate}%

Average Patient Age: {avg_age}

Patients with advanced age groups
show increased likelihood of
heart disease occurrence.

Early risk screening and
preventive healthcare measures
can improve outcomes.
"""
