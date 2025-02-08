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
    col1, col_small_1, col2, col3, col_small_2 = st.columns([3, 0.5, 3, 3, 0.5])
    with col1:
        st.markdown("<h3 style='text-align: center;'>Current Time</h3>", unsafe_allow_html=True)
        gt_current = st.empty()
        pred_current = st.empty()

    # Small column 1: Vertical text
    with col_small_1:
        # Row 1: +30 min
        st.markdown(
            """
            <div style="height: 100px; display: flex; justify-content: center; align-items: center; transform: rotate(270deg);">
                +30 min
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Row 2: +60 min
        st.markdown(
            """
            <div style="height: 100px; display: flex; justify-content: center; align-items: center; transform: rotate(270deg);">
                +60 min
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<h3 style='text-align: center;'>Grountruths</h3>", unsafe_allow_html=True)
        gt_plus_30 = st.empty()
        gt_plus_60 = st.empty()
    with col3:
        st.markdown("<h3 style='text-align: center;'>Predictions</h3>", unsafe_allow_html=True)
        pred_plus_30 = st.empty()
        pred_plus_60 = st.empty()
    # Small column 2 (colorbar placeholder)
    with col_small_2:
        st.markdown("<h4 style='text-align: center;'>Colorbar Placeholder</h4>", unsafe_allow_html=True)

    return gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60
