import streamlit as st
from gtts import gTTS
import tempfile
import os

st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)

def generate_story(image_description, child_name):
    """
    Generate a simple kid-friendly story based on the image description.
    """
    story = (
        f"One sunny day, {child_name} found something wonderful: {image_description}. "
        f"It looked magical and full of adventure. "
        f"With a big smile, {child_name} took a closer look and discovered a happy surprise. "
        f"Soon, new friends arrived and everyone played together kindly. "
        f"They laughed, learned, and shared a beautiful moment. "
        f"At the end of the day, {child_name} knew that curiosity and friendship can make every day special."
    )
    return story

def text_to_speech(text):
    """
    Convert text to speech and save as mp3.
    """
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

st.title("📚 Kids Storytelling App")
st.subheader("Upload a picture and create a fun story!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

child_name = st.text_input("Enter the child's name", value="Lily")
image_description = st.text_area(
    "Describe the image in a few words",
    placeholder="For example: a little dog playing in a garden"
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

if st.button("Generate Story"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    elif not image_description.strip():
        st.warning("Please describe the image.")
    else:
        story = generate_story(image_description, child_name)

        st.success("Story generated successfully!")

        st.markdown("### ✨ Story")
        st.write(story)

        st.write(f"Word count: {len(story.split())}")

        audio_path = text_to_speech(story)

        st.markdown("### 🔊 Audio")
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
