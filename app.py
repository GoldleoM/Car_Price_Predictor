import streamlit as st
import pickle
import pandas as pd

pipe= pickle.load(open("LinearRegressionModel.pkl", "rb"))
st.title("🚗 Car Price Prediction")

ohe = pipe.named_steps['columntransformer']['onehotencoder']

names = ohe.categories[0]
companies = ohe.categories[1]
fuel_types = ohe.categories[2]


name = st.selectbox("Car Name", names)


company = name.split()[0]
st.write(f"Company: {company}")

year = st.number_input("Year", min_value=1995, max_value=2019)
kms_driven = st.number_input("KMs Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", fuel_types)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "name": [name],
        "company": [company],
        "year": [year],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type]
    })

    prediction = pipe.predict(input_df)

    st.success(f"💰 Estimated Price: ₹ {int(prediction[0]):,}")
