import pickle
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from peft import PeftModel

device = torch.device("cpu")


label_names = [
    "Safe",
    "Unsafe",
    "Violent Crimes",
    "Elections",
    "Sex-Related Crimes",
    "unsafe",
    "Non-Violent Crimes",
    "Child Sexual Exploitation",
    "Unknown S-Type",
    "Suicide & Self-Harm"
]


# DISTILBERT
def load_distilbert():

    BASE_MODEL = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        "./distilbert_tokenizer"
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=9
    )

    model = PeftModel.from_pretrained(
        base_model,
        "distilbert_model"
    )

    model.to(device)

    model.eval()

    return tokenizer, model

# ----------


# LLAMA
def load_llama():

    MODEL_ID = "meta-llama/Llama-Guard-3-1B"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID
    )

    model.to(device)

    model.eval()

    return tokenizer, model





max_len = 60


def predict_text(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)

    pred = torch.argmax(
        probs,
        dim=-1
    ).item()

    confidence = probs[0][pred].item()

    return label_names[pred], confidence



# LLAMA MODERATION
def moderate_text(user_text):

    prompt = f"""
You are a content moderation system.

Classify the following text as:
- Safe
- Violent Crimes
- Elections
- Sex-Related Crimes
- unsafe
- Non-Violent Crimes
- Child Sexual Exploitation
- Unknown S-Type
- Suicide & Self-Harm
- Unsafe

Text:
{user_text}

Answer with only one word.
"""
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=5
    )


    # remove prompt tokens
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


