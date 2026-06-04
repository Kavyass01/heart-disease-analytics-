import streamlit as st
import pandas as pd
import plotly.express as px

df=pd.read_csv("data/heart.csv")

st.title("📈 Deep Analytics")

fig=px.scatter(
df,
x="age",
y="thalachh",
color="output",
size="chol",
hover_data=["cp"]
)

st.plotly_chart(fig,use_container_width=True)
