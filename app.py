import os
import re
import tempfile
import streamlit as st
from PIL import Image
from gtts import gTTS
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Kids Storytelling App")
st.write("Upload an image and generate a children's story with audio.")


# --------------------------------------------------
# Load BLIP image captioning model
# Model: Salesforce/blip-image-captioning-base
# --------------------------------------------------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


# --------------------------------------------------
# Load FLAN-T5 story generation model
# Model: google/flan-t5-base
# --------------------------------------------------
@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


# --------------------------------------------------
# Function 1: Image to text
# --------------------------------------------------
def img2text(uploaded_image):
    processor, model = load_blip_model()

    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    inputs = processor(images=uploaded_image, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=5
    )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption


# --------------------------------------------------
# Helper: clean caption
# --------------------------------------------------
def clean_caption(caption):
    caption = caption.strip().lower()

    remove_words = {
        "illustration", "drawing", "cartoon", "painting",
        "art", "image", "picture"
    }

    words = [w for w in caption.split() if w not in remove_words]
    cleaned = " ".join(words).strip()

    if cleaned:
        return cleaned[0].upper() + cleaned[1:]

    return caption


# --------------------------------------------------
# Helper: repetition check
# --------------------------------------------------
def is_too_repetitive(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    if len(words) < 15:
        return True

    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.28


# --------------------------------------------------
# Helper: score story quality
# --------------------------------------------------
def score_story(story):
    words = re.findall(r"\b[a-zA-Z]+\b", story)
    word_count = len(words)

    score = 0

    # Prefer target range
    if 50 <= word_count <= 100:
        score += 3
    elif 45 <= word_count <= 105:
        score += 2
    elif 40 <= word_count <= 110:
        score += 1

    # Prefer non-repetitive text
    if not is_too_repetitive(story):
        score += 2

    # Prefer multiple sentences
    sentence_count = len(re.findall(r"[.!?]", story))
    if 4 <= sentence_count <= 7:
        score += 2
    elif 3 <= sentence_count <= 8:
        score += 1

    # Prefer stories with a simple ending cue
    ending_words = ["happy", "smile", "smiled", "joy", "home", "laughed", "thank"]
    if any(word in story.lower() for word in ending_words):
        score += 1

    return score, word_count


# --------------------------------------------------
# Function 2: Text to story
# --------------------------------------------------
def text2story(caption):
    tokenizer, model = load_story_model()
    clean_cap = clean_caption(caption)

    prompt = (
        f"Write a short children's story in simple English about this scene: {clean_cap}. "
        "Use 5 to 6 sentences and 50 to 100 words. "
        "Include a small adventure and a happy ending."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    candidates = []

    for _ in range(5):
        outputs = model.generate(
            **inputs,
            max_new_tokens=90,
            min_new_tokens=45,
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
            no_repeat_ngram_size=3
        )

        story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        score, word_count = score_story(story)
        candidates.append((story, score, word_count))

        # Good enough -> return immediately
        if score >= 6:
            return story, candidates, False

    # If none are excellent, return the best generated story
    if candidates:
        best_story, best_score, best_word_count = max(candidates, key=lambda x: x[1])

        # If at least somewhat acceptable, use model
