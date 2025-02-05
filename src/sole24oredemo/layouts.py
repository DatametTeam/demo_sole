import streamlit as st
from datetime import datetime

def configure_sidebar() -> None:
    with st.sidebar:
        st.write("**Weather prediction**")    
        with st.form("weather_prediction_form"):
            start_date = st.date_input("Select a start date", value=datetime.now())
            end_date = st.date_input("Select an end date", value=datetime.now())
            model_name = st.selectbox("Select a model", ("ConvLSTM"))
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
        return start_date, end_date, model_name, submitted

def init_prediction_visualization_layout():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h3 style='text-align: center;'>Frame t0</h3>", unsafe_allow_html=True)
        gt_current = st.empty()
        pred_current = st.empty()
    with col2:
        st.markdown("<h3 style='text-align: center;'>Frame t+30</h3>", unsafe_allow_html=True)
        gt_plus_30 = st.empty()
        pred_plus_30 = st.empty()
    with col3:
        st.markdown("<h3 style='text-align: center;'>Frame t+60</h3>", unsafe_allow_html=True)
        gt_plus_60 = st.empty()
        pred_plus_60 = st.empty()
    return gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60
