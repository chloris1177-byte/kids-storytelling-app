import os
import tempfile
import streamlit as st
from PIL import Image
from gtts import gTTS
from transformers import (
    pipeline,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)


st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Kids Storytelling App")
st.write("Upload an image and generate a fun children's story with audio!")


@st.cache_resource
def load_caption_model():
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    return model, feature_extractor, tokenizer


@st.cache_resource
def load_story_pipeline():
    return pipeline(
        task="text-generation",
        model="distilgpt2"
    )


def img2text(uploaded_image):
    model, feature_extractor, tokenizer = load_caption_model()

    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    pixel_values = feature_extractor(
        images=[uploaded_image],
        return_tensors="pt"
    ).pixel_values

    output_ids = model.generate(
        pixel_values,
        max_length=30,
        num_beams=5,
        num_return_sequences=3
    )

    captions = [
        tokenizer.decode(ids, skip_special_tokens=True).strip()
        for ids in output_ids
    ]

    caption = max(captions, key=len)
    return caption


def text2story(caption):
    story_generator = load_story_pipeline()

    prompt = (
        "Write a short children's story in 50 to 100 words. "
        "Use simple English for children aged 3 to 10. "
        "Make the story cheerful, clear, and easy to understand. "
        "Include a beginning, a fun middle, and a happy ending. "
        f"The story is about: {caption}. "
        "Story:"
    )

    result = story_generator(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        full_text = result[0]["generated_text"]

        if full_text.startswith(prompt):
            story = full_text[len(prompt):].strip()
        else:
            story = full_text.strip()

        if len(story.split()) >= 30:
            return story

    return (
        f"There was once {caption}. "
        f"It was a lovely moment, full of curiosity and joy. "
        f"Soon, a little adventure began, bringing smiles and surprises. "
        f"Everyone worked together, helped one another, and had lots of fun. "
        f"In the end, it became a beautiful memory with a happy ending."
    )


def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

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
