
import os
import tempfile
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from predict import MODEL_PATH, predict_sound


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="SonicSense | Sound Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom UI styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(0, 210, 255, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 95% 5%,
                rgba(139, 92, 246, 0.16),
                transparent 32%
            ),
            #080d1b;
        color: #edf2ff;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111a31, #0a1020);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 30px 28px;
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            rgba(0, 194, 255, 0.16),
            rgba(113, 78, 255, 0.20)
        );
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #60e5ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: clamp(34px, 5vw, 52px);
        font-weight: 800;
        line-height: 1.1;
        margin: 12px 0;
        background: linear-gradient(90deg, #ffffff, #83e8ff, #b9a2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #b7c5df;
        font-size: 16px;
        line-height: 1.7;
        max-width: 720px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        margin: 18px 0 12px;
        color: #f1f5ff;
    }

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 112px;
    }

    .metric-label {
        color: #9eafcc;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 23px;
        font-weight: 800;
        margin-top: 10px;
        overflow-wrap: anywhere;
    }

    .result-card {
        padding: 24px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(0, 210, 255, 0.12),
            rgba(139, 92, 246, 0.16)
        );
        border: 1px solid rgba(94, 234, 255, 0.24);
        margin: 12px 0 20px;
    }

    .result-label {
        color: #80eaff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .result-class {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }

    div.stButton > button {
        border: 0;
        border-radius: 12px;
        background: linear-gradient(90deg, #06b6d4, #7c3aed);
        color: white;
        font-weight: 750;
        min-height: 48px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        filter: brightness(1.15);
        border: 0;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.035);
        border: 1px dashed rgba(105, 210, 255, 0.40);
        border-radius: 16px;
        padding: 12px;
    }

    hr {
        border-color: rgba(255,255,255,0.10);
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning • Audio Intelligence</div>
        <h1>SonicSense</h1>
        <p>
            Turn environmental sounds into meaningful predictions.
            Upload an audio recording, inspect its waveform, and let
            the trained machine learning model identify the sound.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🎧 SonicSense")
    st.caption("Environmental Sound Classification")

    st.markdown("---")
    st.markdown("### About the model")
    st.markdown(
        """
        - **Algorithm:** Improved Support Vector Machine
        - **Dataset:** UrbanSound8K
        - **Features:** 29-feature pipeline
        - **Input:** Audio recordings
        """
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload an audio file.
        2. Listen to the recording.
        3. Inspect its waveform.
        4. Click **Classify Sound**.
        5. Review the predicted class and available scores.
        """
    )

    st.markdown("---")
    st.caption("SonicSense • ML Project")


# --------------------------------------------------
# Audio examples
# --------------------------------------------------
project_dir = Path(__file__).resolve().parent

example_files = {
    "Car ignition": project_dir / "car-ignition-1.wav",
    "Gun cocking": project_dir / "gun-cocking-01.wav",
    "Child laughing": project_dir / "mixkit-happy-child-laughing-2265.wav",
}

available_examples = {
    name: path
    for name, path in example_files.items()
    if path.is_file()
}


# --------------------------------------------------
# Input area
# --------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        '<div class="section-title">🎵 Audio input</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Choose your audio source",
        ["Upload audio", "Try an example"],
        horizontal=True,
    )

    audio_path = None
    uploaded_audio = None

    if input_mode == "Upload audio":
        uploaded_audio = st.file_uploader(
            "Upload an audio recording",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Choose an environmental sound recording.",
        )

    else:
        if available_examples:
            selected_example = st.selectbox(
                "Select an example recording",
                list(available_examples.keys()),
            )
            audio_path = available_examples[selected_example]
        else:
            st.warning(
                "No example recordings were found in the project folder. "
                "Upload a recording instead."
            )

with right:
    st.markdown(
        '<div class="section-title">✨ What you get</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Audio analysis</div>
            <div class="metric-value">Waveform preview</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">Model output</div>
            <div class="metric-value">Sound classification</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">Result details</div>
            <div class="metric-value">Scores when available</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Save uploaded audio temporarily
# --------------------------------------------------
temporary_path = None

if uploaded_audio is not None:
    suffix = Path(uploaded_audio.name).suffix.lower() or ".wav"

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_file.write(uploaded_audio.getvalue())
            temporary_path = temporary_file.name

        audio_path = Path(temporary_path)

    except OSError as exc:
        st.error(f"Could not prepare the uploaded audio: {exc}")


# --------------------------------------------------
# Audio preview and waveform
# --------------------------------------------------
if audio_path is not None and Path(audio_path).is_file():
    st.markdown("---")
    st.markdown(
        '<div class="section-title">📊 Audio preview</div>',
        unsafe_allow_html=True,
    )

    audio_col, info_col = st.columns([1.3, 0.7], gap="large")

    with audio_col:
        try:
            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read())

            waveform, sample_rate = librosa.load(
                str(audio_path),
                sr=None,
                mono=True,
            )

            duration = len(waveform) / sample_rate if sample_rate else 0

            st.markdown("#### Waveform")

            figure, axis = plt.subplots(figsize=(10, 2.6))
            time_axis = np.arange(len(waveform)) / sample_rate

            axis.plot(time_axis, waveform, linewidth=0.65)
            axis.set_xlabel("Time (seconds)")
            axis.set_ylabel("Amplitude")
            axis.set_xlim(0, max(duration, 0.01))
            axis.grid(alpha=0.20)
            figure.tight_layout()

            st.pyplot(figure, use_container_width=True)
            plt.close(figure)

        except Exception as exc:
            st.warning(
                "The audio preview or waveform could not be displayed. "
                f"Details: {exc}"
            )
            waveform = None
            sample_rate = None
            duration = None

    with info_col:
        st.markdown("#### Recording details")

        if sample_rate:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Sample rate</div>
                    <div class="metric-value">{sample_rate:,} Hz</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-label">Duration</div>
                    <div class="metric-value">{duration:.2f} sec</div>
                </div>
                <br>
                <div class="metric-card">
                    <div class="metric-label">Samples</div>
                    <div class="metric-value">{len(waveform):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    st.markdown("---")

    if st.button(
        "🚀 Classify Sound",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Analyzing audio with the Improved SVM..."):
            try:
                predicted_class, details = predict_sound(str(audio_path))

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted sound</div>
                        <div class="result-class">
                            🎯 {predicted_class}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.session_state["sonicsense_prediction"] = str(
                    predicted_class
                )
                st.session_state["sonicsense_details"] = details

            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)

    # --------------------------------------------------
    # Optional scores
    # --------------------------------------------------
    if "sonicsense_prediction" in st.session_state:
        # Clear old result if the user selects a different recording.
        current_key = str(audio_path)
        previous_key = st.session_state.get("sonicsense_audio_key")

        if previous_key != current_key:
            st.session_state.pop("sonicsense_prediction", None)
            st.session_state.pop("sonicsense_details", None)
        else:
            predicted_class = st.session_state["sonicsense_prediction"]
            details = st.session_state.get("sonicsense_details")

            st.markdown("### Prediction details")

            score_values = np.asarray(details)
            if score_values.shape == (1, 10):
                score_values = score_values[0]
            elif score_values.shape != (10,):
                score_values = None

            if (
                score_values is not None
                and np.issubdtype(score_values.dtype, np.number)
                and np.isfinite(score_values).all()
            ):
                try:
                    score_model = joblib.load(MODEL_PATH)
                    score_labels = np.asarray(
                        score_model.named_steps["svm"].classes_
                    )
                except (AttributeError, KeyError, OSError, TypeError, ValueError):
                    score_labels = None

                if score_labels is not None and len(score_labels) == len(score_values):
                    ranked_scores = sorted(
                        zip(score_labels, score_values),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                    st.caption(
                        "Raw SVM decision scores; these are not calibrated "
                        "probabilities or confidence percentages."
                    )

                    for label, score in ranked_scores:
                        st.write(f"**{label}** — {float(score):.4f}")
                else:
                    st.info(
                        f"Predicted class: **{predicted_class}**. "
                        "Class labels could not be matched to the score array."
                    )
            else:
                st.info(
                    f"Predicted class: **{predicted_class}**. "
                    "The model returned an unexpected or non-numeric "
                    "decision-score array."
                )

            st.session_state["sonicsense_audio_key"] = current_key

elif uploaded_audio is None and audio_path is None:
    st.info("Upload a recording or select an available example to begin.")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8494b3; padding:12px;">
        SonicSense · Environmental Sound Classification · Powered by ML
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Cleanup temporary uploaded file
# --------------------------------------------------
if temporary_path and os.path.exists(temporary_path):
    try:
        os.remove(temporary_path)
    except OSError:
        pass
