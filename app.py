import streamlit as st

st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Analytics Platform")

st.markdown("""
### Deep Analytics & AI Insights

Analyze heart disease patterns, discover risk factors,
and predict patient outcomes using Machine Learning.
""")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Patients",303)

with col2:
    st.metric("Features",13)

with col3:
    st.metric("Target Classes",2)

st.image(
"https://images.unsplash.com/photo-1576091160550-2173dba999ef",
use_container_width=True
)
