# 🛡️ AI Content Moderation System

A Streamlit-based AI application that compares multiple NLP models for content moderation and image understanding.

The system supports:
- DistilBERT text classification
- Transformer-based classification (optional)
- LLaMA (optional, heavy model)
- BLIP image captioning

---

# 🚀 Features

## 📌 Text Moderation
1. Classifies input text into
-   Safe
-   Violent Crimes
-   Elections
-   Sex-Related Crimes
-   unsafe
-   Non-Violent Crimes
-   Child Sexual Exploitation
-   Unknown S-Type
-   Suicide & Self-Harm
-   Unsafe categories
2. Uses Transformer-based models (DistilBERT or others)

## 🖼️ Image Understanding
- Upload image → generate caption using BLIP model
- Caption is then classified by the moderation model

## ⚖️ Model Comparison
Users can compare outputs from:
- DistilBERT
- Transformer model (optional)
- LLaMA (optional)

---

# 🧠 Models Used

## 1. DistilBERT
- Lightweight transformer model
- Fast inference
- Used for classification

## 2. BLIP (Image Captioning)
- Converts image → text description

## 3. LLaMA (optional)
- Large language model for advanced classification
- Requires HuggingFace access + high RAM

---

# 📂 Project Structure
├── app.py # Streamlit UI

├── text_classifier.py # Text models (DistilBERT / Transformer / LLaMA)

├── image_captioning.py # BLIP image captioning

├── requirements.txt

├── README.md

└── saved_models/

├── distilBERT/

└── p2-llama/

---

# ⚙️ Installation

## Create environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows

2. Install dependencies
pip install -r requirements.txt


## Run the App
streamlit run app.py
