import os
import tempfile
import requests
from PIL import Image
from gtts import gTTS
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# Hugging Face API configuration
# Store your Hugging Face token in Streamlit Secrets:
# HF_TOKEN = "your_token_here"
# --------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

IMAGE_MODEL_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
TEXT_MODEL_URL = "https://api-inference.huggingface.co/models/distilgpt2"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# --------------------------------------------------
# Function 1: Generate image caption
# --------------------------------------------------
def img2text(image_bytes):
    """
    Use a Hugging Face image captioning model to generate
    a caption from the uploaded image.

    Parameters:
        image_bytes (bytes): uploaded image in bytes

    Returns:
        str: generated image caption
    """
    response = requests.post(
        IMAGE_MODEL_URL,
        headers=HEADERS,
        data=image_bytes,
        timeout=60
    )
    response.raise_for_status()
    result = response.json()

    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]["generated_text"]

    raise ValueError(f"Unexpected image caption response: {result}")


# --------------------------------------------------
# Function 2: Generate story from caption
# --------------------------------------------------
def text2story(caption):
    """
    Use a Hugging Face text generation model to generate
    a short children's story based on the image caption.

    Parameters:
        caption (str): generated image caption

    Returns:
        str: generated story
    """
    prompt = (
        f"Write a happy and simple children's story in 50 to 100 words "
        f"for kids aged 3 to 10. "
        f"Base the story on this image description: {caption}. "
        f"Use easy words, a friendly tone, and a clear ending."
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.9,
            "return_full_text": False
        }
    }

    response = requests.post(
        TEXT_MODEL_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    result = response.json()

    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        story = result[0]["generated_text"].strip()
        return story

    raise ValueError(f"Unexpected text generation response: {result}")


# --------------------------------------------------
# Function 3: Convert story text to audio
# --------------------------------------------------
def text2audio(story_text):
    """
    Convert the generated story into speech using gTTS.

    Parameters:
        story_text (str): generated story text

    Returns:
        str: path to temporary audio file
    """
    tts = gTTS(text=story_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name


# --------------------------------------------------
# Streamlit user interface
# --------------------------------------------------
st.title("📚 Kids Storytelling App")
st.write("Upload an image and generate a fun story using Hugging Face!")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        try:
            with st.spinner("Generating caption, story, and audio..."):
                image_bytes = uploaded_file.getvalue()

                # Step 1: image captioning
                caption = img2text(image_bytes)

                # Step 2: story generation
                story = text2story(caption)

                # Step 3: text-to-speech
                audio_path = text2audio(story)

            st.success("Story generated successfully!")

            st.subheader("🖼️ Image Caption")
            st.write(caption)

            st.subheader("✨ Story")
            st.write(story)
            st.write(f"**Word count:** {len(story.split())}")

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

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an image to begin.")
