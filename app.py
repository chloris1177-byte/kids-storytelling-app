import os
import tempfile
from PIL import Image
from gtts import gTTS
import streamlit as st
from transformers import pipeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📚 Kids Storytelling App")
st.write("Upload an image and generate a fun children's story with audio!")


# --------------------------------------------------
# Cache models to avoid reloading every interaction
# --------------------------------------------------
@st.cache_resource
def load_image_caption_pipeline():
    return pipeline(
        task="image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_pipeline():
    return pipeline(
        task="text-generation",
        model="distilgpt2"
    )


# --------------------------------------------------
# Function 1: Generate caption from image
# --------------------------------------------------
def img2text(uploaded_image):
    """
    Generate a caption for the uploaded image using a Hugging Face image captioning pipeline.
    """
    image_to_text = load_image_caption_pipeline()
    result = image_to_text(uploaded_image)

    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]["generated_text"]

    raise ValueError("Could not generate image caption.")


# --------------------------------------------------
# Function 2: Generate story from caption
# --------------------------------------------------
def text2story(caption):
    """
    Generate a short children's story based on the image caption.
    """
    story_generator = load_story_pipeline()

    prompt = (
        f"Write a simple, happy children's story in 50 to 100 words. "
        f"The story is for children aged 3 to 10. "
        f"Use easy vocabulary, a warm tone, and a clear ending. "
        f"Image description: {caption}. Story:"
    )

    result = story_generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        full_text = result[0]["generated_text"]

        # Remove prompt if it is included in output
        if full_text.startswith(prompt):
            story = full_text[len(prompt):].strip()
        else:
            story = full_text.strip()

        return story

    raise ValueError("Could not generate story.")


# --------------------------------------------------
# Function 3: Convert story to speech
# --------------------------------------------------
def text2audio(story_text):
    """
    Convert story text to audio using gTTS and save to a temporary MP3 file.
    """
    tts = gTTS(text=story_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name


# --------------------------------------------------
# Upload image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Main app logic
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        try:
            with st.spinner("Generating caption..."):
                caption = img2text(image)

            with st.spinner("Generating story..."):
                story = text2story(caption)

            with st.spinner("Generating audio..."):
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
                    file_name="kids_story.mp3",
                    mime="audio/mpeg"
                )

            if os.path.exists(audio_path):
                os.remove(audio_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an image to begin.")
