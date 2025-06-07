import streamlit as st
from PyPDF2 import PdfReader
from langdetect import detect
from gtts import gTTS
import tempfile
import os

from transformers import pipeline
from deep_translator import GoogleTranslator

# -------------------- Streamlit Page Configuration -------------------- #
# This must be the first Streamlit command in the script
# Removed 'icon' argument as it might not be supported in older Streamlit versions.
st.set_page_config(page_title="PDF & Text to Audio", layout="centered")

# -------------------- Caching Models -------------------- #
@st.cache_resource
def load_summarizer():
    """Caches the summarization pipeline to avoid reloading the model on every rerun."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -------------------- Language Support -------------------- #
@st.cache_resource
def get_supported_translation_languages():
    """
    Dynamically fetches supported languages from GoogleTranslator and caches them.
    Provides a fallback list if fetching fails.
    """
    try:
        # get_supported_languages returns a dictionary {language_name: language_code}
        # Note: deep-translator's GoogleTranslator supports a wide range of languages.
        # This will keep the the app's language list up-to-date with the library.
        return GoogleTranslator().get_supported_languages(as_dict=True)
    except Exception as e:
        st.error(f"Failed to fetch supported languages for translation: {e}. Using a default list.")
        # Fallback to a predefined list if dynamic fetching fails (e.g., no internet)
        return {
            "English": "en",
            "Tamil": "ta",
            "Hindi": "hi",
            "Telugu": "te",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Bengali": "bn",
            "Marathi": "mr",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Chinese (Simplified)": "zh-CN",
            "Japanese": "ja",
            "Arabic": "ar"
        }

supported_languages = get_supported_translation_languages()

# -------------------- Helper Functions -------------------- #
def extract_text_from_pdf(uploaded_file, start_page=None, end_page=None):
    """
    Extracts text from specified pages of a PDF file.

    Args:
        uploaded_file: The Streamlit UploadedFile object.
        start_page (int, optional): The 0-indexed start page. Defaults to 0.
        end_page (int, optional): The 0-indexed end page. Defaults to last page.

    Returns:
        tuple: A tuple containing the extracted text (str) and the total number of pages (int).
    """
    reader = PdfReader(uploaded_file)
    num_pages = len(reader.pages)

    # Ensure page ranges are valid
    start = max(0, start_page if start_page is not None else 0)
    end = min(end_page if end_page is not None else num_pages - 1, num_pages - 1)

    text = ""
    for i in range(start, end + 1):
        page = reader.pages[i].extract_text()
        if page:
            text += page + "\n"
    return text.strip(), num_pages

def detect_language(text):
    """
    Detects the language of the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The detected language code (e.g., 'en', 'fr') or 'unknown' if detection fails.
    """
    try:
        return detect(text)
    except:
        return "unknown" # Fallback if language detection fails

def convert_text_to_audio(text, lang, slow=False):
    """
    Converts text to an MP3 audio file using gTTS.

    Args:
        text (str): The text to convert.
        lang (str): The language code for the audio.
        slow (bool, optional): If True, reads text more slowly. Defaults to False.
                               Note: gTTS only supports normal or ~40% slower speed.

    Returns:
        str: The path to the temporary MP3 file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        try:
            tts = gTTS(text=text, lang=lang, slow=slow)
            tts.save(tmp_file.name)
            return tmp_file.name
        except Exception as e:
            st.error(f"Failed to generate audio for language '{lang}': {e}. Please try another language or shorter text.")
            return None # Indicate failure

def apply_summarization(text):
    """
    Applies summarization to the text using the pre-loaded summarizer pipeline.
    Handles long texts by chunking them.

    Args:
        text (str): The input text to summarize.

    Returns:
        str: The summarized text.
    """
    # For very long texts, warn the user about potential summarization quality impact
    # due to fixed-character chunking (models prefer token-aware chunking).
    if len(text) > 5000: # Arbitrary threshold to warn about very long texts
        st.warning("The input text is very long. Summarization is applied in chunks, "
                   "which might affect the coherence of the overall summary.")

    # Chunk the text into 1000-character segments for summarization
    # Note: Transformer models like BART-large-cnn have an input token limit (e.g., 1024 tokens).
    # 1000 characters is a heuristic; actual token count can vary.
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            # max_length and min_length control the output summary length for each chunk
            summary_output = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary_output[0]['summary_text'])
        except Exception as e:
            st.error(f"Error summarizing chunk {i+1}: {e}. Skipping this chunk.")
            # Optionally append original chunk or a placeholder if summarization fails
            summaries.append("")
    return " ".join(summaries).strip()

def apply_translation(text, target_lang_code):
    """
    Translates the given text to the target language code.

    Args:
        text (str): The text to translate.
        target_lang_code (str): The target language code (e.g., 'es', 'fr').

    Returns:
        str: The translated text.
    """
    return GoogleTranslator(source='auto', target=target_lang_code).translate(text)

# -------------------- Streamlit UI -------------------- #
st.title("🔊 PDF & Text to Audio Converter")

tab1, tab2 = st.tabs(["📄 PDF to Speech", "✍️ Text to Speech"])

# -------------------- Tab 1: PDF to Speech -------------------- #
with tab1:
    st.header("Upload a PDF file to convert to audio")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        raw_text, total_pages = extract_text_from_pdf(uploaded_pdf)
        if raw_text:
            st.success(f"✅ PDF loaded successfully! It contains {total_pages} pages.")

            st.markdown("### Select Pages (Optional)")
            col1, col2 = st.columns(2)
            # Ensure start and end page inputs are within valid bounds
            start_page = col1.number_input("Start Page (0-indexed)", 0, total_pages - 1, 0, key="pdf_start_page")
            end_page = col2.number_input("End Page (0-indexed)", 0, total_pages - 1, total_pages - 1, key="pdf_end_page")

            # Re-extract text based on selected range
            text, _ = extract_text_from_pdf(uploaded_pdf, int(start_page), int(end_page))

            st.markdown("### 🧠 Pre-processing Options")
            summarize = st.checkbox("Summarize text before audio conversion", key="pdf_summarize")
            translate = st.checkbox("Translate text before audio conversion", key="pdf_translate")

            processed_text = text # Initialize with original extracted text

            if summarize:
                try:
                    with st.spinner("Summarizing text..."):
                        processed_text = apply_summarization(processed_text)
                    st.success("Text summarized!")
                except Exception as e:
                    st.error(f"Summarization failed: {e}. Please try again or disable summarization.")

            selected_translation_lang = None
            if translate:
                # Use a sorted list of language names for the selectbox for better UX
                sorted_lang_names = sorted(supported_languages.keys())
                selected_lang_name = st.selectbox(
                    "Translate to",
                    sorted_lang_names,
                    index=sorted_lang_names.index("English") if "English" in sorted_lang_names else 0, # Default to English
                    key="pdf_translate_select"
                )
                selected_translation_lang = supported_languages[selected_lang_name]

                try:
                    with st.spinner(f"Translating to {selected_lang_name}..."):
                        processed_text = apply_translation(processed_text, selected_translation_lang)
                    st.success(f"Translated to {selected_lang_name}!")
                except Exception as e:
                    st.error(f"Translation failed: {e}. Please check your text or try another language.")
                    # Fallback to original text if translation fails
                    processed_text = text

            # Determine the language for gTTS
            lang_for_audio = "en" # Default to English
            if translate and selected_translation_lang:
                lang_for_audio = selected_translation_lang
            else:
                detected_lang = detect_language(processed_text)
                # Use detected language if it's in gTTS supported languages, otherwise default to English
                if detected_lang in supported_languages.values(): # Check against translation languages as proxy for gTTS
                    lang_for_audio = detected_lang
                else:
                    st.warning(f"Detected language '{detected_lang}' may not be fully supported for audio generation. Defaulting to English (en).")
                    lang_for_audio = "en"


            st.subheader("📋 Final Text Preview")
            # Truncate for display in text_area to avoid UI lag for very long texts
            st.text_area("Preview", processed_text[:5000] + ("..." if len(processed_text) > 5000 else ""), height=250)
            st.download_button("⬇️ Download Final Text", processed_text.encode('utf-8'), file_name="extracted_processed_text.txt", mime="text/plain")

            slow = st.radio("Speech Speed", ["Normal", "Slow"], horizontal=True, key="pdf_speed_radio") == "Slow"

            if st.button("🎧 Convert to Audio", key="pdf_convert_button"):
                if processed_text.strip():
                    audio_file_path = None
                    try:
                        with st.spinner("Generating MP3 audio... This might take a moment for long texts."):
                            audio_file_path = convert_text_to_audio(processed_text, lang_for_audio, slow)

                        if audio_file_path:
                            with open(audio_file_path, "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/mp3")
                                st.download_button("⬇️ Download MP3", audio_file, file_name="pdf_speech.mp3", mime="audio/mp3")
                        else:
                            st.error("Could not generate audio. Please check the text and selected language.")
                    except Exception as e:
                        st.error(f"An error occurred during audio conversion: {e}")
                    finally:
                        if audio_file_path and os.path.exists(audio_file_path):
                            os.remove(audio_file_path) # Ensure temporary file is deleted
                else:
                    st.warning("No text to convert to audio. Please extract text from PDF or adjust page range.")
        else:
            st.warning("❌ No text could be extracted from the selected pages of the PDF. "
                       "This might happen if the PDF is image-based or corrupted.")
    else:
        st.info("⬆️ Please upload a PDF to get started with PDF to Speech conversion.")

# -------------------- Tab 2: Text to Speech -------------------- #
with tab2:
    st.header("Enter your text to convert to audio")
    user_text = st.text_area("Type or paste your text here", height=250, key="user_text_input")

    if user_text.strip():
        st.markdown("### 🧠 Pre-processing Options")
        summarize_txt = st.checkbox("Summarize text before audio (for typed text)", key="txt_summarize")
        translate_txt = st.checkbox("Translate text before audio (for typed text)", key="txt_translate")

        processed_user_text = user_text # Initialize with original user text

        if summarize_txt:
            try:
                with st.spinner("Summarizing text..."):
                    processed_user_text = apply_summarization(processed_user_text)
                st.success("Text summarized!")
            except Exception as e:
                st.error(f"Summarization failed: {e}. Please try again or disable summarization.")

        selected_translation_lang_txt = None
        if translate_txt:
            sorted_lang_names_txt = sorted(supported_languages.keys())
            selected_lang_name_txt = st.selectbox(
                "Translate to",
                sorted_lang_names_txt,
                index=sorted_lang_names_txt.index("English") if "English" in sorted_lang_names_txt else 0, # Default to English
                key="txt_translate_select"
            )
            selected_translation_lang_txt = supported_languages[selected_lang_name_txt]

            try:
                with st.spinner(f"Translating to {selected_lang_name_txt}..."):
                    processed_user_text = apply_translation(processed_user_text, selected_translation_lang_txt)
                st.success(f"Translated to {selected_lang_name_txt}!")
            except Exception as e:
                st.error(f"Translation failed: {e}. Please check your text or try another language.")
                # Fallback to original text if translation fails
                processed_user_text = user_text


        # Determine the language for gTTS
        lang_for_audio_txt = "en" # Default to English
        if translate_txt and selected_translation_lang_txt:
            lang_for_audio_txt = selected_translation_lang_txt
        else:
            detected_lang_txt = detect_language(processed_user_text)
            # Use detected language if it's in gTTS supported languages, otherwise default to English
            if detected_lang_txt in supported_languages.values(): # Check against translation languages as proxy for gTTS
                lang_for_audio_txt = detected_lang_txt
            else:
                st.warning(f"Detected language '{detected_lang_txt}' may not be fully supported for audio generation. Defaulting to English (en).")
                lang_for_audio_txt = "en"


        st.subheader("📋 Final Text Preview")
        # Truncate for display in text_area
        st.text_area("Preview", processed_user_text[:5000] + ("..." if len(processed_user_text) > 5000 else ""), height=250, key="final_text_preview")
        st.download_button("⬇️ Download Final Text", processed_user_text.encode('utf-8'), file_name="typed_processed_text.txt", mime="text/plain")

        slow_txt = st.radio("Speech Speed", ["Normal", "Slow"], horizontal=True, key="txt_speed_radio") == "Slow"

        if st.button("🎧 Convert to Audio", key="txt_convert_button"):
            if processed_user_text.strip():
                audio_file_path_txt = None
                try:
                    with st.spinner("Generating MP3 audio..."):
                        audio_file_path_txt = convert_text_to_audio(processed_user_text, lang_for_audio_txt, slow_txt)

                    if audio_file_path_txt:
                        with open(audio_file_path_txt, "rb") as audio_file_txt:
                            st.audio(audio_file_txt.read(), format="audio/mp3")
                            st.download_button("⬇️ Download MP3", audio_file_txt, file_name="text_speech.mp3", mime="audio/mp3")
                    else:
                        st.error("Could not generate audio. Please check the text and selected language.")
                except Exception as e:
                    st.error(f"An error occurred during audio conversion: {e}")
                finally:
                    if audio_file_path_txt and os.path.exists(audio_file_path_txt):
                        os.remove(audio_file_path_txt) # Ensure temporary file is deleted
            else:
                st.warning("No text to convert to audio. Please enter some text.")
    else:
        st.info("✍️ Enter some text above to convert to speech.")
