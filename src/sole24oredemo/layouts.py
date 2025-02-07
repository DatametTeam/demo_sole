import streamlit as st
from datetime import datetime


def configure_sidebar():
    with st.sidebar:
        st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>NOWCASTING</h1>", unsafe_allow_html=True)
        with st.form("weather_prediction_form"):
            start_date = st.date_input("Select a start date", value=datetime.now())
            end_date = st.date_input("Select an end date", value=datetime.now())
            model_name = st.selectbox("Select a model", ("ConvLSTM", "SmAtUnet"))
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
        return start_date, end_date, model_name, submitted


def init_prediction_visualization_layout():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h3 style='text-align: center;'>Current Time</h3>", unsafe_allow_html=True)
        gt_current = st.empty()
        pred_current = st.empty()
    with col2:
        st.markdown("<h3 style='text-align: center;'>Time +30min</h3>", unsafe_allow_html=True)
        gt_plus_30 = st.empty()
        pred_plus_30 = st.empty()
    with col3:
        st.markdown("<h3 style='text-align: center;'>Time t+60min</h3>", unsafe_allow_html=True)
        gt_plus_60 = st.empty()
        pred_plus_60 = st.empty()
    return gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60
