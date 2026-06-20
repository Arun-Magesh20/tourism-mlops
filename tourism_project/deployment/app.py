
import streamlit as st
import pandas as pd
import pickle
model = pickle.load(
    open("../model_building/model.pkl", "rb")
)
features = pickle.load(
    open("../model_building/features.pkl", "rb")
)
st.title("Tourism Package Prediction")
age = st.number_input(
    "Age",
    min_value=18,
    max_value=80
)
income = st.number_input(
    "Monthly Income",
    min_value=1000
)
if st.button("Predict"):

    input_df = pd.DataFrame(
        [[age, income]],
        columns=["Age", "MonthlyIncome"]
    )
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    pred = model.predict(input_df)
    st.success(
        f"Prediction: {pred[0]}"
    )
