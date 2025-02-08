import streamlit as st
from datetime import datetime, time, timedelta


def configure_sidebar():
    with st.sidebar:
        st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>NOWCASTING</h1>", unsafe_allow_html=True)
        with st.form("weather_prediction_form"):
            # Date inputs
            start_date = st.date_input("Select a start date", value=datetime(2025, 1, 31).date(),
                                       format="DD/MM/YYYY")  # TODO: rimettere now
            # Time inputs
            start_time = st.time_input(
                "Select a start time",
                value=time(0, 0),
                step=timedelta(minutes=5)  # 5-minute intervals
            )
            end_date = st.date_input("Select an end date", value=datetime(2025, 1, 31).date(),
                                     format="DD/MM/YYYY")  # TODO: da rimettere now

            end_time = st.time_input(
                "Select an end time",
                value=time(1, 55),
                step=timedelta(minutes=5)  # 5-minute intervals
            )

            # Model selection
            model_name = st.selectbox("Select a model", ("ConvLSTM", "SmAtUnet"))

            # Form submission
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

        return {"start_date": start_date, "end_date": end_date, "start_time": start_time, "end_time": end_time,
                "model_name": model_name, "submitted": submitted}


def init_prediction_visualization_layout():
    row1 = st.columns([3, 0.3, 3, 3, 0.4], vertical_alignment='center')
    row2 = st.columns([3, 0.3, 3, 3, 0.4], vertical_alignment='center')
    row3 = st.columns([3, 0.3, 3, 3, 0.4], vertical_alignment='center')

    with row1[0]:
        st.markdown("<h3 style='text-align: center;'>Current Time</h3>", unsafe_allow_html=True)
    with row1[2]:
        st.markdown("<h3 style='text-align: center;'>Groundtruths</h3>", unsafe_allow_html=True)
    with row1[3]:
        st.markdown("<h3 style='text-align: center;'>Predictions</h3>", unsafe_allow_html=True)

    with row2[0]:
        gt_current = st.empty()

    with row3[0]:
        pred_current = st.empty()

    with row2[1]:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <div style="transform: rotate(-90deg); font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                    +30min
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with row3[1]:
        st.markdown(
            """
            <div style="position: relative; height: 100%; width: 100%;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(270deg);
                font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                    +60min
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row2[2]:
        gt_plus_30 = st.empty()
    with row3[2]:
        gt_plus_60 = st.empty()

    with row2[3]:
        pred_plus_30 = st.empty()
    with row3[3]:
        pred_plus_60 = st.empty()

    with row2[4]:
        colorbar30 = st.empty()
    with row3[4]:
        colorbar60 = st.empty()

    return gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60, colorbar30, colorbar60
