import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("🚦 Traffic Sign Recognition")

# st.divider()

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_cnn.keras")


@st.cache_data
def load_class_names():

    df = pd.read_csv("traffic_sign.csv")

    return dict(
        zip(
            df["ClassId"].astype(int),
            df["Name"].astype(str)
        )
    )


# ============================================================
# LOAD MODEL + CLASSES
# ============================================================

try:
    model = load_model()
    class_names = load_class_names()

except Exception as e:

    st.error("Unable to load the model or class-name file.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write("**Model:** CNN")

    st.write("**Dataset:** GTSRB")

    st.write("**Classes:** 59")

    st.write("**Input:** 32 × 32 × 3")

    st.write("**Normalization:** /255")

    


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📤 Upload Traffic Sign")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # HORIZONTAL LAYOUT
    # --------------------------------------------------------

    upload_col, prediction_col, top5_col = st.columns(
        [1.1, 1.2, 1.7],
        gap="large"
    )

    # ========================================================
    # COLUMN 1 - IMAGE
    # ========================================================

    with upload_col:

        st.subheader("🖼️ Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Image size: {image.size[0]} × {image.size[1]} px"
        )


    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    resized_image = image.resize(
        (32, 32)
    )

    img_array = np.array(
        resized_image
    ).astype("float32")

    # Same preprocessing as training
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with prediction_col:

        st.subheader("🎯 Prediction")

        if st.button(
            "🔍 Predict Traffic Sign",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                predictions = model.predict(
                    img_array,
                    verbose=0
                )[0]


            # ------------------------------------------------
            # BEST PREDICTION
            # ------------------------------------------------

            predicted_index = int(
                np.argmax(predictions)
            )

            confidence = float(
                predictions[predicted_index]
            )

            predicted_name = class_names.get(
                predicted_index,
                "Unknown Traffic Sign"
            )


            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            st.session_state["prediction"] = predicted_name

            st.session_state["confidence"] = confidence

            st.session_state["predictions"] = predictions


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if "prediction" in st.session_state:

            predicted_name = st.session_state["prediction"]

            confidence = st.session_state["confidence"]

            st.success(
                f"**{predicted_name}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.write("Confidence Level")

            st.progress(
                confidence
            )

            if confidence >= 0.80:

                st.success(
                    "🟢 High confidence"
                )

            elif confidence >= 0.60:

                st.warning(
                    "🟡 Moderate confidence"
                )

            else:

                st.error(
                    "🔴 Low confidence"
                )


    # ========================================================
    # TOP 5 PREDICTIONS
    # ========================================================

    with top5_col:

        st.subheader("🏆 Top 5 Predictions")

        if "predictions" in st.session_state:

            predictions = st.session_state["predictions"]

            top5_indices = np.argsort(
                predictions
            )[-5:][::-1]


            for rank, index in enumerate(
                top5_indices,
                start=1
            ):

                sign_name = class_names.get(
                    int(index),
                    "Unknown Traffic Sign"
                )

                probability = float(
                    predictions[index]
                )


                # --------------------------------------------
                # SIGN NAME
                # --------------------------------------------

                st.write(
                    f"**{rank}. {sign_name}**"
                )

                # --------------------------------------------
                # PROBABILITY
                # --------------------------------------------

                st.progress(
                    probability
                )

                st.caption(
                    f"{probability * 100:.2f}%"
                )

        else:

            st.info(
                "Top predictions will appear here "
                "after prediction."
            )


# ============================================================
# BEFORE IMAGE UPLOAD
# ============================================================

# else:

#     st.info(
#         "👆 Upload a traffic-sign image above to begin."
#     )


# ============================================================
# FOOTER
# ============================================================

# st.divider()

