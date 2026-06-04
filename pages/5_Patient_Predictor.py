age=st.slider("Age",20,90,50)
chol=st.slider("Cholesterol",100,600,200)
thalachh=st.slider("Max Heart Rate",60,220,150)

if st.button("Predict"):

    sample=pd.DataFrame({
        "age":[age],
        "chol":[chol],
        "thalachh":[thalachh]
    })

    prediction=model.predict(sample)

    if prediction[0]==1:
        st.error("High Risk")
    else:
        st.success("Low Risk")
