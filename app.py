import streamlit as st
import pandas as pd
import os
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling


with st.sidebar:
    st.image("../Auto-ML/ONE-POINT-01-1.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Classification Model", "Regression Model", "Download Model"])
    st.info("This aplication allows you to builed automated machine learning models ")


if os.path.exists("source_dara.csv"):
    df = pd.read_csv("source_dara.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your Data set for modeling")
    file = st.file_uploader("upload your data set here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("source_dara.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Explore through Atomated data analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Classification Model":
    from pycaret.classification import setup, compare_models, pull, save_model
    st.title("Classification Model")
    target = st.selectbox("Select your target", df.columns)
    if st.button("train model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML expierment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("this is the ML model")
        st.dataframe(compare_df)
        best_model 
        save_model(best_model, 'best_model')



if choice == "Regression Model":
    from pycaret.regression import setup, compare_models, pull, save_model
    st.title("Regression Model")
    target = st.selectbox("Select your target", df.columns)
    if st.button("train model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML expierment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("this is the ML model")
        st.dataframe(compare_df)
        best_model 
        save_model(best_model, 'best_model')

if choice == "Download Model":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the file", f, "best_model.pkl")

