import os
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
# Helper: check repetitive output
# --------------------------------------------------
def clean_caption(caption):
    caption = caption.strip()
    remove_words = ["illustration", "drawing", "cartoon", "painting"]
    words = caption.split()
    words = [w for w in words if w.lower() not in remove_words]
    cleaned = " ".join(words).strip()
    return cleaned if cleaned else caption


def is_too_repetitive(text):
    words = text.lower().split()
    if len(words) < 15:
        return True

    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.30


def text2story(caption):
    tokenizer, model = load_story_model()
    caption = clean_caption(caption)

    prompt = (
        f"Image description: {caption}\n"
        "Task: Write a children's story.\n"
        "Rules:\n"
        "- 50 to 100 words\n"
        "- 5 to 6 sentences\n"
        "- simple English\n"
        "- cheerful tone\n"
        "- happy ending\n"
        "- no repetition\n"
        "Story:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    for _ in range(3):
        outputs = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            no_repeat_ngram_size=3
        )

        story = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        word_count = len(story.split())

        if 45 <= word_count <= 105 and not is_too_repetitive(story):
            return story

    return (
        "One sunny day, a group of children played happily in the park. "
        "They ran across the grass, laughed together, and enjoyed the warm sunshine. "
        "Soon, they discovered a small lost puppy near the flowers and decided to help it. "
        "They looked around carefully and finally found the puppy's owner nearby. "
        "The owner thanked the children for their kindness and gave them a big smile. "
        "At the end of the day, everyone went home feeling proud, cheerful, and happy."
    )
# --------------------------------------------------
# Function 3: Text to audio
# --------------------------------------------------
def text2audio(story_text):
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
# Main app
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        try:
            with st.spinner("Generating image caption..."):
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
