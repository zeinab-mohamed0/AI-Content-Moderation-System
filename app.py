import streamlit as st

from PIL import Image

from image_captioning import generate_caption

from text_classifier import (
    load_distilbert,
    load_llama,
    predict_text,
    moderate_text
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Moderation Comparison",
    layout="centered"
)

st.title("🛡️ AI Content Moderation System")

st.write(
    "Compare DistilBERT and LLaMA moderation models"
)


# =========================
# INPUT TYPE
# =========================

input_type = st.radio(
    "Choose Input Type",
    ["Text", "Image"]
)


text_input = ""


# =========================
# TEXT INPUT
# =========================

if input_type == "Text":

    text_input = st.text_area(
        "Enter Text"
    )


# =========================
# IMAGE INPUT
# =========================

else:

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("Generating Caption..."):

            text_input = generate_caption(image)

        st.success(
            f"Generated Caption: {text_input}"
        )


# =========================
# MODEL SELECTION
# =========================

selected_models = st.multiselect(
    "Choose Models",
    [
        "DistilBERT",
        "LLaMA"
    ],
    default=["DistilBERT"]
)


# =========================
# RUN BUTTON
# =========================

if st.button("Run Moderation"):

    if text_input.strip() == "":

        st.warning(
            "Please enter text or upload image"
        )

    else:

        st.subheader("Results")


        # =====================
        # DISTILBERT
        # =====================

        if "DistilBERT" in selected_models:

            tokenizer, model = load_distilbert()

            label, confidence = predict_text(
                text_input,
                tokenizer,
                model
            )

            st.markdown("## DistilBERT")

            st.write(f"Prediction: {label}")

            st.progress(float(confidence))

            st.write(
                f"Confidence: {confidence:.4f}"
            )


    


        # =====================
        # LLAMA
        # =====================

        if "LLaMA" in selected_models:

            tokenizer, model = load_llama()

            response = moderate_text(
                text_input
            )

            st.markdown("## LLaMA")

            st.write(
                f"Prediction: {response}"
            )