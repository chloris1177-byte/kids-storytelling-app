import os
import tempfile
from PIL import Image

import streamlit as st
from transformers import pipeline
from gtts import gTTS


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)


# -----------------------------
# Load Hugging Face pipelines
# Use cache_resource so models are loaded only once
# -----------------------------
@st.cache_resource
def load_image_caption_model():
    """
    Load the Hugging Face image-to-text pipeline.
    This model generates a caption from an uploaded image.
    """
    return pipeline(
        task="image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_model():
    """
    Load the Hugging Face text generation pipeline.
    This model expands a short caption into a short children's story.
    """
    return pipeline(
        task="text-generation",
        model="distilgpt2"
    )


# -----------------------------
# Function 1: image to caption
# -----------------------------
def img2text(image):
    """
    Generate a caption from the uploaded image using a Hugging Face model.

    Parameters:
        image: PIL image object

    Returns:
        str: generated caption
    """
    image_to_text_model = load_image_caption_model()
    result = image_to_text_model(image)
    caption = result[0]["generated_text"]
    return caption


# -----------------------------
# Function 2: caption to story
# -----------------------------
def text2story(caption):
    """
    Generate a short kid-friendly story based on the image caption.

    Parameters:
        caption (str): image caption generated from Hugging Face model

    Returns:
        str: generated story
    """
    story_generator = load_story_model()

    prompt = (
        f"Write a short, simple, happy children's story in 60 to 90 words "
        f"for kids aged 3 to 10 based on this scene: {caption}. "
        f"The story should be friendly, imaginative, and easy to understand."
    )

    result = story_generator(
        prompt,
        max_new_tokens=90,
        num_return_sequences=1,
        temperature=0.9,
        do_sample=True,
        truncation=True,
        pad_token_id=50256
    )

    generated_text = result[0]["generated_text"]

    # Remove the prompt part if it appears at the beginning
    if generated_text.startswith(prompt):
        story = generated_text[len(prompt):].strip()
    else:
        story = generated_text.strip()

    return story


# -----------------------------
# Function 3: story to audio
# -----------------------------
def text2audio(story_text):
    """
    Convert story text to speech using gTTS.

    Parameters:
        story_text (str): story text

    Returns:
        str: path to temporary mp3 file
    """
    tts = gTTS(text=story_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 Kids Storytelling App")
st.write("Upload an image and let Hugging Face create a fun story for kids!")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        with st.spinner("Analyzing image and creating story..."):
            # Step 1: image captioning
            caption = img2text(image)

            # Step 2: story generation
            story = text2story(caption)

            # Step 3: text to speech
            audio_path = text2audio(story)

        st.success("Story generated successfully!")

        st.subheader("🖼️ Image Caption")
        st.write(caption)

        st.subheader("✨ Story")
        st.write(story)

        word_count = len(story.split())
        st.write(f"**Word count:** {word_count}")

        st.subheader("🔊 Story Audio")
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="story.mp3",
                mime="audio/mpeg"
            )

        if os.path.exists(audio_path):
            os.remove(audio_path)
else:
    st.info("Please upload an image to begin.")
