import streamlit as st
import os
import tempfile
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
import re
import io

# ----------------------------- TOKEN SETUP -----------------------------
def validate_hf_token(token):
    if not token:
        return False
    return token.startswith('hf_') and len(token) >= 30

def prompt_for_token():
    st.sidebar.markdown("### 🔐 HuggingFace Token Required")
    st.sidebar.markdown("""
    To use this app, enter your HuggingFace token below.
    - Go to [huggingface.co](https://huggingface.co/)
    - Create an account → Settings → Access Tokens → Create new token with 'read' access
    """)
    token = st.sidebar.text_input("Enter your HuggingFace token", type="password", key="hf_token_input")
    if st.sidebar.button("Submit Token", key="submit_token_btn"):
        if validate_hf_token(token):
            os.environ["HF_TOKEN"] = token
            st.session_state["hf_token"] = token
            st.session_state["hf_token_validated"] = True
            st.sidebar.success("Token saved successfully!")
            st.rerun()
        else:
            st.sidebar.error("Invalid token format. Must start with 'hf_' and be at least 30 characters long.")

# Check token status at startup
if "hf_token_validated" not in st.session_state:
    st.session_state["hf_token_validated"] = False

if not st.session_state.get("hf_token_validated", False):
    prompt_for_token()
    st.stop()
else:
    # Ensure token is in environment
    if "hf_token" in st.session_state:
        os.environ["HF_TOKEN"] = st.session_state["hf_token"]

# Set page config
st.set_page_config(
    page_title="SINE English Speaking Tutor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        margin-bottom: 30px;
    }
    .message-bot {
        background-color: #e9e9eb;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2E86C1;
    }
    .message-user {
        background-color: #dcf8c6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #28B463;
        text-align: right;
    }
    .feedback-box {
        background-color: #FEF9E7;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #F39C12;
        margin: 10px 0;
    }
    .evaluation-box {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #6C757D;
        margin: 10px 0;
    }
    .report-box {
        background-color: #EBF5FB;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3498DB;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'grade' not in st.session_state:
        st.session_state.grade = None
    if 'topic' not in st.session_state:
        st.session_state.topic = None
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'vocab_index' not in st.session_state:
        st.session_state.vocab_index = 0
    if 'phase' not in st.session_state:
        st.session_state.phase = "intro"
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = []
    if 'pronunciation_scores' not in st.session_state:
        st.session_state.pronunciation_scores = []
    if 'lesson_started' not in st.session_state:
        st.session_state.lesson_started = False
    if 'lesson_completed' not in st.session_state:
        st.session_state.lesson_completed = False
    if 'last_audio_message' not in st.session_state:
        st.session_state.last_audio_message = ""
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = []
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
        
    # Load models only once
    if not st.session_state.models_loaded:
        load_models()

def load_models():
    """Load all required models with flexible fallback and dynamic local path resolution"""
    import pathlib

    try:
        # Load STT model
        if 'stt_model' not in st.session_state:
            with st.spinner("Loading speech recognition model..."):
                st.session_state.stt_model = WhisperModel("base", device="cpu", compute_type="int8")

        # Load LLM model with dynamic fallback
        if 'llm_model' not in st.session_state:
            st.session_state.llm_model = None
            st.session_state.llm_tokenizer = None

            try:
                with st.spinner("Loading language model..."):
                    model_name = "google/flan-t5-small"
                    base_dir = pathlib.Path(__file__).parent if "__file__" in globals() else pathlib.Path.cwd()
                    local_model_path = base_dir / "models" / "google" / "flan-t5-small"

                    if local_model_path.exists():
                        st.session_state.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                            str(local_model_path), local_files_only=True
                        )
                        st.session_state.llm_tokenizer = AutoTokenizer.from_pretrained(
                            str(local_model_path), local_files_only=True
                        )
                        st.success("✅ Language model loaded from local path.")
                    else:
                        raise FileNotFoundError("Local model folder not found at: " + str(local_model_path))

            except Exception as e:
                st.warning(f"Local load failed: {e}. Trying Hugging Face Hub if token is available...")
                try:
                    token = st.secrets.get("hf_token")  # Option A: Load from secrets.toml
                    if token:
                        st.session_state.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            token=token,
                            local_files_only=False
                        )
                        st.session_state.llm_tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            token=token,
                            local_files_only=False
                        )
                        st.success("✅ Language model loaded using Hugging Face Hub.")
                    else:
                        raise ValueError("No HF token found in secrets and local model unavailable.")
                except Exception as e:
                    st.warning(f"Could not load language model from Hugging Face: {e}. Using fallback responses.")

        st.session_state.models_loaded = True

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.session_state.models_loaded = False


# Topic data
topics = {
    "Grade 7": {
        "Travel": {
            "vocab": [
                {
                    "word": "take off",
                    "meaning": "when a plane leaves the ground and begins to fly",
                    "example": "The plane will take off at 9 AM.",
                    "question": "What time does your flight usually take off?"
                },
                {
                    "word": "flight",
                    "meaning": "a trip on an airplane",
                    "example": "I have a flight to Bangkok tomorrow.",
                    "question": "Tell me about your last flight. Where did you go?"
                },
                {
                    "word": "delay",
                    "meaning": "when something happens later than expected",
                    "example": "My flight was delayed for two hours.",
                    "question": "Have you ever had a flight delay? What happened?"
                }
            ],
            "final_challenge": "Can you tell a short story using all 3 words: take off, flight, and delay? Try to describe a real or made-up travel experience."
        },
        "Free Time": {
            "vocab": [
                {
                    "word": "hobby",
                    "meaning": "something you enjoy doing in your free time",
                    "example": "Drawing is my favorite hobby.",
                    "question": "What is one of your hobbies?"
                },
                {
                    "word": "relax",
                    "meaning": "to rest or take it easy",
                    "example": "I relax by listening to soft music.",
                    "question": "How do you usually relax on weekends?"
                },
                {
                    "word": "outdoors",
                    "meaning": "outside, not indoors",
                    "example": "I love spending time outdoors in the park.",
                    "question": "What do you like to do outdoors?"
                }
            ],
            "final_challenge": "Can you tell a short story using all 3 words: hobby, relax, and outdoors? Try to describe your ideal weekend."
        },
        "Shopping": {
            "vocab": [
                {
                    "word": "buy",
                    "meaning": "to get something by paying money for it",
                    "example": "I want to buy a new shirt.",
                    "question": "What did you buy last week?"
                },
                {
                    "word": "expensive",
                    "meaning": "costing a lot of money",
                    "example": "This phone is too expensive for me.",
                    "question": "What's the most expensive thing you've ever bought?"
                },
                {
                    "word": "discount",
                    "meaning": "a reduction in price",
                    "example": "I got a 20% discount on these shoes.",
                    "question": "Do you like shopping when there are discounts?"
                }
            ],
            "final_challenge": "Can you tell a story using all 3 words: buy, expensive, and discount? Describe a shopping experience you had."
        }
    },
    "Grade 8": {
        "Travel": {
            "vocab": [
                {
                    "word": "itinerary",
                    "meaning": "a detailed plan of a journey or trip",
                    "example": "I carefully planned my itinerary for the European tour.",
                    "question": "Do you usually plan a detailed itinerary before traveling, or do you prefer spontaneous trips?"
                },
                {
                    "word": "turbulence",
                    "meaning": "violent or irregular movement of air during flight",
                    "example": "We experienced severe turbulence during our flight to Tokyo.",
                    "question": "How do you feel when there's turbulence during a flight? Are you nervous or calm?"
                },
                {
                    "word": "accommodation",
                    "meaning": "a place to stay, such as a hotel or hostel",
                    "example": "The accommodation we booked had an amazing view of the ocean.",
                    "question": "What type of accommodation do you prefer when traveling - hotels, hostels, or staying with locals?"
                }
            ],
            "final_challenge": "Create a detailed narrative using all 3 words: itinerary, turbulence, and accommodation. Describe a memorable travel experience, including planning, the journey, and where you stayed."
        },
        "Free Time": {
            "vocab": [
                {
                    "word": "recreational",
                    "meaning": "relating to activities done for enjoyment and relaxation",
                    "example": "Swimming is my favorite recreational activity during summer.",
                    "question": "What recreational activities do you find most fulfilling and why?"
                },
                {
                    "word": "rejuvenate",
                    "meaning": "to make someone feel refreshed and energized again",
                    "example": "A weekend in nature always helps me rejuvenate after a stressful week.",
                    "question": "What activities help you rejuvenate when you feel tired or stressed?"
                },
                {
                    "word": "contemplative",
                    "meaning": "thoughtful and reflective",
                    "example": "I enjoy contemplative walks along the beach at sunset.",
                    "question": "Do you prefer active or contemplative activities during your free time?"
                }
            ],
            "final_challenge": "Compose a reflective narrative using all 3 words: recreational, rejuvenate, and contemplative. Describe how you spend your ideal free time and its impact on your well-being."
        },
        "Technology": {
            "vocab": [
                {
                    "word": "innovative",
                    "meaning": "featuring new methods or ideas; advanced and original",
                    "example": "The new smartphone has innovative features that make daily tasks easier.",
                    "question": "What innovative technology do you think has had the greatest impact on society?"
                },
                {
                    "word": "compatibility",
                    "meaning": "the ability of different systems or devices to work together",
                    "example": "I need to check the compatibility of this software with my computer.",
                    "question": "Have you ever had compatibility issues with your devices or software?"
                },
                {
                    "word": "obsolete",
                    "meaning": "no longer in use or no longer useful",
                    "example": "Many people think physical books will become obsolete with digital reading.",
                    "question": "What technology from the past do you think is now obsolete, and do you miss it?"
                }
            ],
            "final_challenge": "Develop a comprehensive discussion using all 3 words: innovative, compatibility, and obsolete. Share your thoughts on how technology evolves and its impact on our daily lives."
        }
    }
}

# Helper functions
def transcribe_audio(audio_file):
    """Transcribe audio from UploadedFile object"""
    if audio_file is None:
        return ""
    
    # Create temporary file and write audio data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        # Read bytes from UploadedFile and write to temp file
        audio_bytes = audio_file.read()
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        segments, _ = st.session_state.stt_model.transcribe(tmp_file_path)
        result = " ".join([s.text for s in segments])
        os.unlink(tmp_file_path)  # Clean up temp file
        return result
    except Exception as e:
        st.error(f"Transcription error: {e}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)  # Clean up temp file
        return ""

def pronunciation_score(expected: str, actual: str) -> int:
    """Calculate pronunciation similarity score"""
    return int(SequenceMatcher(None, expected.lower(), actual.lower()).ratio() * 100)

def clean_text_for_tts(text):
    """Clean text for text-to-speech - more aggressive cleaning"""
    # Remove markdown formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # Remove *italic*
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)  # Remove _underline_
    text = re.sub(r"#{1,6}\s*", "", text)         # Remove headers
    
    # Remove emojis and special characters that might cause TTS issues
    text = re.sub(r"[🤖👤📊📈📝💪🎯✅❌🎉🔊✨🌟👍⚡📚💪🔄📊🎯]", "", text)
    text = re.sub(r"[*_:#`~•]", "", text)          # Remove remaining special chars
    
    # Remove bullet points and formatting
    text = re.sub(r"^[\s]*[•\-\*]\s*", "", text, flags=re.MULTILINE)
    
    # Clean up spacing and newlines
    text = re.sub(r"\s{2,}", " ", text)           # Remove extra spaces
    text = re.sub(r"\n{2,}", ". ", text)          # Replace multiple newlines with periods
    text = re.sub(r"\n", ". ", text)              # Replace single newlines with periods
    
    # Remove evaluation markers and technical terms
    text = re.sub(r"Question \d+ Evaluation:", "Here's your evaluation:", text)
    text = re.sub(r"Final Challenge Results", "Here are your final results", text)
    text = re.sub(r"Progress Update - Question \d+:", "Here's your progress:", text)
    
    # Limit length for TTS (gTTS has character limits)
    if len(text) > 500:
        sentences = text.split('. ')
        truncated = []
        char_count = 0
        for sentence in sentences:
            if char_count + len(sentence) + 2 < 500:  # +2 for '. '
                truncated.append(sentence)
                char_count += len(sentence) + 2
            else:
                break
        text = '. '.join(truncated)
        if not text.endswith('.'):
            text += '.'
    
    return text.strip()

def clean_text_for_display(text):
    """Clean text for chat display - removes markdown and special characters"""
    # Remove markdown formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # Remove *italic*
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)  # Remove _underline_
    text = re.sub(r"#{1,6}\s*", "", text)         # Remove headers
    text = re.sub(r"[*_:#`~]", "", text)          # Remove remaining special chars
    text = re.sub(r"\s{2,}", " ", text)           # Remove extra spaces
    text = re.sub(r"\n{2,}", "\n", text)          # Remove extra newlines
    return text.strip()

def text_to_speech(text):
    """Convert text to speech and return audio bytes"""
    try:
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text.strip():
            return None
            
        tts = gTTS(text=cleaned_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def ask_llm(prompt):
    """Use Flan-T5 for text generation with fallback"""
    if st.session_state.llm_model is None or st.session_state.llm_tokenizer is None:
        return "Great work! Keep practicing your vocabulary and speaking skills."

    try:
        inputs = st.session_state.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = st.session_state.llm_model.generate(
            inputs.input_ids,
            max_length=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=st.session_state.llm_tokenizer.eos_token_id
        )
        response = st.session_state.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return "Great work! Keep practicing your vocabulary and speaking skills."

def evaluate_response(user_text: str, expected_question: str, topic: str, vocab_word: str = "") -> dict:
    """Comprehensive evaluation of user response"""
    evaluation = {
        "pronunciation_score": 0,
        "relevance_score": 0,
        "vocabulary_usage": False,
        "grammar_feedback": "",
        "overall_rating": "",
        "strengths": [],
        "improvements": []
    }
    
    # Pronunciation score
    evaluation["pronunciation_score"] = pronunciation_score(expected_question, user_text)
    
    # Vocabulary usage check
    if vocab_word and vocab_word.lower() in user_text.lower():
        evaluation["vocabulary_usage"] = True
        evaluation["strengths"].append(f"Successfully used the target word '{vocab_word}'")
    elif vocab_word:
        evaluation["improvements"].append(f"Try to include the word '{vocab_word}' in your response")
    
    # Relevance scoring based on question type
    relevance = 0
    if "what time" in expected_question.lower():
        time_indicators = ["am", "pm", "morning", "afternoon", "evening", "o'clock", ":", "time"]
        if any(indicator in user_text.lower() for indicator in time_indicators):
            relevance = 85
            evaluation["strengths"].append("Appropriately answered the time-related question")
        else:
            relevance = 30
            evaluation["improvements"].append("Include specific time information in your answer")
    
    elif "where" in expected_question.lower():
        location_indicators = ["to", "from", "airport", "city", "country", "place", "destination"]
        if any(indicator in user_text.lower() for indicator in location_indicators):
            relevance = 85
            evaluation["strengths"].append("Provided location information as requested")
        else:
            relevance = 35
            evaluation["improvements"].append("Include location details in your response")
    
    elif "how" in expected_question.lower():
        if len(user_text.split()) >= 5:  # Reasonable explanation length
            relevance = 80
            evaluation["strengths"].append("Provided a detailed explanation")
        else:
            relevance = 50
            evaluation["improvements"].append("Try to give more detailed explanations")
    
    else:
        # General response evaluation
        if len(user_text.split()) >= 3:
            relevance = 75
        else:
            relevance = 40
            evaluation["improvements"].append("Try to provide more complete sentences")
    
    evaluation["relevance_score"] = relevance
    
    # Grammar feedback
    evaluation["grammar_feedback"] = grammar_tutor_chat(user_text, topic, expected_question)
    
    # Overall rating
    avg_score = (evaluation["pronunciation_score"] + evaluation["relevance_score"]) / 2
    if avg_score >= 80:
        evaluation["overall_rating"] = "Excellent"
        evaluation["strengths"].append("Great overall performance!")
    elif avg_score >= 65:
        evaluation["overall_rating"] = "Good"
        evaluation["strengths"].append("Solid performance with room for improvement")
    elif avg_score >= 50:
        evaluation["overall_rating"] = "Fair"
        evaluation["improvements"].append("Keep practicing to improve fluency")
    else:
        evaluation["overall_rating"] = "Needs Practice"
        evaluation["improvements"].append("Focus on speaking more clearly and completely")
    
    # Length and fluency check
    word_count = len(user_text.split())
    if word_count >= 8:
        evaluation["strengths"].append("Good sentence length and detail")
    elif word_count < 4:
        evaluation["improvements"].append("Try to speak in more complete sentences")
    
    return evaluation

def format_evaluation_for_chat(evaluation: dict, question_number: int) -> str:
    """Format evaluation as a chat message"""
    eval_text = f"📊 **Question {question_number} Evaluation:**\n\n"
    eval_text += f"**Scores:**\n"
    eval_text += f"• Pronunciation: {evaluation['pronunciation_score']}%\n"
    eval_text += f"• Relevance: {evaluation['relevance_score']}%\n"
    eval_text += f"• Overall Rating: {evaluation['overall_rating']}\n"
    eval_text += f"• Vocabulary Used: {'✅ Yes' if evaluation['vocabulary_usage'] else '❌ No'}\n\n"
    
    if evaluation['strengths']:
        eval_text += "**💪 Strengths:**\n"
        for strength in evaluation['strengths']:
            eval_text += f"• {strength}\n"
        eval_text += "\n"
    
    if evaluation['improvements']:
        eval_text += "**🎯 Areas for Improvement:**\n"
        for improvement in evaluation['improvements']:
            eval_text += f"• {improvement}\n"
        eval_text += "\n"
    
    eval_text += f"**📝 Grammar Feedback:**\n{evaluation['grammar_feedback']}"
    
    return eval_text

def grammar_tutor_chat(user_sentence: str, topic: str, question: str = "") -> str:
    """Provide grammar feedback"""
    if not user_sentence.strip():
        return "Correction: (No input detected)\nExplanation: Please try speaking again.\nEncouragement: You can do it!"

    # Simple grammar corrections based on patterns
    corrected_sentence = user_sentence
    explanation = ""

    # Common grammar fixes
    if "take off" in user_sentence.lower():
        if "usually take off" in user_sentence.lower():
            corrected_sentence = user_sentence.replace("take off", "takes off")
            explanation = "Use 'takes off' (third person singular) instead of 'take off' when talking about 'flight' or 'plane'."
        elif "will take off" in user_sentence.lower():
            explanation = "Good use of future tense with 'will take off'!"

    # Check if answer matches the question context
    question_match = True
    example_answer = ""

    if question and "what time" in question.lower() and "take off" in question.lower():
        if not any(time_word in user_sentence.lower() for time_word in ["am", "pm", "morning", "afternoon", "evening", "o'clock", ":"]):
            question_match = False
            example_answer = "My flight usually takes off at 9 AM."
    elif question and "where" in question.lower() and "flight" in question.lower():
        if not any(place_word in user_sentence.lower() for place_word in ["to", "from", "airport", "city", "country"]):
            question_match = False
            example_answer = "My last flight was to Bangkok."
    elif question and "delay" in question.lower():
        if "delay" not in user_sentence.lower():
            question_match = False
            example_answer = "Yes, my flight was delayed for two hours because of bad weather."

    # Build response
    if not question_match:
        return (f"Correction: {corrected_sentence}\n"
                f"Explanation: Your answer doesn't fully address the question. The question asks {question}\n"
                f"Example Answer: {example_answer}\n"
                f"Encouragement: Try to answer the question directly. You're doing great with the {topic} vocabulary!")

    if corrected_sentence != user_sentence:
        return (f"Correction: {corrected_sentence}\n"
                f"Explanation: {explanation}\n"
                f"Alternative: You could also say {example_answer if example_answer else corrected_sentence}\n"
                f"Encouragement: Great job using {topic} vocabulary! Your pronunciation is improving!")
    else:
        return (f"Correction: Perfect! No changes needed.\n"
                f"Explanation: Your sentence correctly answers the question about {topic}.\n"
                f"Encouragement: Excellent work! You're using the {topic} vocabulary naturally!")

def generate_progress_summary(topic, user_responses, evaluations):
    """Generate progress summary after each vocabulary word"""
    if not evaluations:
        return "Keep practicing!"
    
    current_eval = evaluations[-1]
    question_num = len(evaluations)
    
    summary = f"📈 **Progress Update - Question {question_num}:**\n\n"
    
    # Current performance
    summary += f"**This Question:**\n"
    summary += f"• Score: {current_eval['pronunciation_score']}% pronunciation, {current_eval['relevance_score']}% relevance\n"
    summary += f"• Rating: {current_eval['overall_rating']}\n"
    summary += f"• Vocabulary: {'Used correctly ✅' if current_eval['vocabulary_usage'] else 'Not detected ❌'}\n\n"
    
    # Overall progress if more than one question
    if len(evaluations) > 1:
        avg_pronunciation = sum(e["pronunciation_score"] for e in evaluations) / len(evaluations)
        avg_relevance = sum(e["relevance_score"] for e in evaluations) / len(evaluations)
        
        summary += f"**Overall Progress ({len(evaluations)} questions):**\n"
        summary += f"• Average Pronunciation: {avg_pronunciation:.1f}%\n"
        summary += f"• Average Relevance: {avg_relevance:.1f}%\n"
        
        # Progress trend
        if len(evaluations) >= 2:
            prev_score = (evaluations[-2]["pronunciation_score"] + evaluations[-2]["relevance_score"]) / 2
            curr_score = (current_eval["pronunciation_score"] + current_eval["relevance_score"]) / 2
            
            if curr_score > prev_score:
                summary += f"• Trend: Improving! 📈 (+{curr_score - prev_score:.1f} points)\n"
            elif curr_score < prev_score:
                summary += f"• Trend: Keep practicing 📊 ({curr_score - prev_score:.1f} points)\n"
            else:
                summary += f"• Trend: Steady performance 📊\n"
        
        summary += "\n"
    
    # Encouragement
    if current_eval['overall_rating'] == "Excellent":
        summary += "🌟 Amazing work! You're mastering this vocabulary!"
    elif current_eval['overall_rating'] == "Good":
        summary += "👍 Great job! You're making solid progress!"
    elif current_eval['overall_rating'] == "Fair":
        summary += "💪 Good effort! Keep practicing to improve!"
    else:
        summary += "📚 Don't worry! Practice makes perfect!"
    
    return summary

def generate_report(topic, user_responses, pronunciation_scores):
    """Generate final progress report"""
    avg_pronunciation = sum(pronunciation_scores) / len(pronunciation_scores) if pronunciation_scores else 0

    # Extract topic info for better fallback
    topic_words = []
    grade_topic = topic.split(" - ")
    topic_name = "English"
    if len(grade_topic) == 2:
        grade, topic_name = grade_topic
        if grade in topics and topic_name in topics[grade]:
            topic_words = [vocab["word"] for vocab in topics[grade][topic_name]["vocab"]]

    # Try using the model first
    if st.session_state.llm_model is not None and st.session_state.llm_tokenizer is not None:
        prompt = (
            f"Generate a progress report for a student practicing '{topic}' vocabulary. "
            f"Key words practiced: {', '.join(topic_words)}. "
            f"Completed {len(user_responses)} exercises with {avg_pronunciation:.1f}% pronunciation score. "
            f"Focus on their {topic} vocabulary usage and speaking confidence. "
            f"Provide specific recommendations for improving {topic} conversations. "
            f"Keep under 100 words, be encouraging and topic-specific."
        )

        try:
            response = ask_llm(prompt)
            if response and len(response) > 20:  # Check if we got a meaningful response
                return response
        except Exception as e:
            pass

    # Fallback report generation
    performance_level = "excellent" if avg_pronunciation >= 80 else "good" if avg_pronunciation >= 60 else "developing"

    display_words = []
    if topic_words:
        display_words = topic_words[:3]
        if len(topic_words) > 3:
            display_words.append("and more")

    fallback_report_parts = [
        f"You've completed {len(user_responses)} speaking exercises, achieving a **{performance_level}** "
        f"pronunciation score of **{avg_pronunciation:.1f}%**.",
        "Your speaking confidence is clearly on the rise!",
    ]

    if display_words:
        fallback_report_parts.insert(1, f"You've actively practiced key **{topic_name}** terms like **{', '.join(display_words)}**.")
    else:
        fallback_report_parts.insert(1, f"You've actively practiced key **{topic_name}** vocabulary.")

    fallback_report_parts.append(
        f"To keep this momentum, try integrating these **{topic_name}** words into your daily chats. "
        f"Focus on incorporating them naturally, especially when discussing subjects related to **{topic_name.lower()}**."
    )
    fallback_report_parts.append("Fantastic progress!")

    return " ".join(fallback_report_parts)

def display_history_with_audio():
    """Display conversation history with automatic audio for bot messages"""
    for i, (role, message) in enumerate(st.session_state.history):
        # Clean the message for display
        cleaned_message = clean_text_for_display(message)
        
        if role == "Bot":
            st.markdown(f'<div class="message-bot"><strong>🤖 Bot:</strong> {cleaned_message}</div>', unsafe_allow_html=True)
            
            # Only generate and play audio for the LAST bot message that hasn't been played yet
            is_last_bot_message = (i == len(st.session_state.history) - 1)
            audio_key = f"audio_played_{i}"
            
            # Check if this is a new bot message that needs audio
            if is_last_bot_message and audio_key not in st.session_state:
                try:
                    with st.spinner("🔊 Generating audio..."):
                        audio_bytes = text_to_speech(message)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        # Mark this message as having had its audio played
                        st.session_state[audio_key] = True
                except Exception as e:
                    st.warning(f"Could not generate audio: {e}")
            
            # For older messages, show audio player without autoplay
            elif audio_key in st.session_state:
                try:
                    audio_bytes = text_to_speech(message)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                except Exception as e:
                    pass  # Silently fail for older messages
                    
        elif role == "User":
            st.markdown(f'<div class="message-user"><strong>👤 You:</strong> {cleaned_message}</div>', unsafe_allow_html=True)
        elif role == "Evaluation":
            st.markdown(f'<div class="evaluation-box">{cleaned_message}</div>', unsafe_allow_html=True)

def display_history():
    """Display conversation history with cleaned text (kept for backward compatibility)"""
    display_history_with_audio()

def reset_lesson():
    """Reset all lesson state including audio cache"""
    st.session_state.history = []
    st.session_state.grade = None
    st.session_state.topic = None
    st.session_state.step = 0
    st.session_state.vocab_index = 0
    st.session_state.phase = "intro"
    st.session_state.user_responses = []
    st.session_state.pronunciation_scores = []
    st.session_state.lesson_started = False
    st.session_state.lesson_completed = False
    st.session_state.last_audio_message = ""
    st.session_state.evaluations = []
    
    # Clear audio playing cache (updated key pattern)
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith("audio_played_")]
    for key in keys_to_remove:
        del st.session_state[key]

# Main app
def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">🤖 SINE English Speaking Tutor</h1>', unsafe_allow_html=True)
    
    # Show token status in sidebar
    with st.sidebar:
        if st.session_state.get("hf_token_validated", False):
            st.success("✅ HuggingFace Token Valid")
            if st.button("Reset Token", key="reset_token_btn"):
                st.session_state.hf_token_validated = False
                if "hf_token" in st.session_state:
                    del st.session_state["hf_token"]
                st.rerun()
        
        st.header("Lesson Setup")
        
        # Grade selection
        grade_options = ["Grade 7", "Grade 8"]
        selected_grade = st.selectbox("Select Grade", grade_options, index=None)
        
        # Topic selection (depends on grade)
        topic_options = []
        if selected_grade and selected_grade in topics:
            topic_options = list(topics[selected_grade].keys())
        
        selected_topic = st.selectbox("Choose Topic", topic_options, index=None, disabled=not bool(topic_options))
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_lesson = st.button("Start Lesson", disabled=not (selected_grade and selected_topic))
        with col2:
            reset_lesson_btn = st.button("Reset")
        
        if reset_lesson_btn:
            reset_lesson()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Conversation")
        
        # Start lesson logic
        if start_lesson and selected_grade and selected_topic and not st.session_state.lesson_started:
            st.session_state.grade = selected_grade
            st.session_state.topic = selected_topic
            st.session_state.lesson_started = True
            st.session_state.phase = "intro"
            st.session_state.vocab_index = 0
            
            # Introduce topic and first vocab word
            topic_data = topics[st.session_state.grade][st.session_state.topic]
            first_vocab = topic_data["vocab"][0]
            
            bot_intro = f"Hi there! I'm your SINE English speaking assistant. Today we're learning about {st.session_state.topic} for {st.session_state.grade}. You'll practice 3 useful words. Let's start with the first word: **{first_vocab['word']}**."
            bot_explanation = f"\n\n**Meaning:** {first_vocab['meaning']}\n**Example:** {first_vocab['example']}\n\nReady to practice using this word?"
            full_bot_response = bot_intro + bot_explanation
            
            st.session_state.history.append(("Bot", full_bot_response))
            st.rerun()
        
        # Display conversation history with audio
        if st.session_state.history:
            display_history_with_audio()
        else:
            st.info("Welcome! Please select your grade and topic from the sidebar to begin.")
    
    with col2:
        st.header("Practice Area")
        
        if st.session_state.lesson_started and not st.session_state.lesson_completed:
            # Continue button logic
            if st.session_state.phase in ["intro", "vocab_intro", "challenge_intro"]:
                if st.button("Continue", key="continue_btn"):
                    topic_data = topics[st.session_state.grade][st.session_state.topic]
                    
                    if st.session_state.phase == "intro":
                        # First continue - ask the first question
                        current_word = topic_data["vocab"][st.session_state.vocab_index]
                        bot_response = f"Now let's practice! **Question:** {current_word['question']}\n\nPlease record your answer using the microphone below."
                        st.session_state.history.append(("Bot", bot_response))
                        st.session_state.phase = "vocab_practice"
                        
                    elif st.session_state.phase == "vocab_intro":
                        # Continue after introducing a new vocab word
                        current_word = topic_data["vocab"][st.session_state.vocab_index]
                        bot_response = f"Now let's practice this word! **Question:** {current_word['question']}\n\nPlease record your answer."
                        st.session_state.history.append(("Bot", bot_response))
                        st.session_state.phase = "vocab_practice"
                        
                    elif st.session_state.phase == "challenge_intro":
                        # Continue to final challenge
                        bot_response = f"Here's your final challenge:\n\n**{topic_data['final_challenge']}**\n\nPlease record your story."
                        st.session_state.history.append(("Bot", bot_response))
                        st.session_state.phase = "challenge"
                    
                    st.rerun()
            
            # Audio recording and submission
            if st.session_state.phase in ["vocab_practice", "challenge"]:
                st.subheader("🎤 Record Your Answer")
                audio_file = st.audio_input("Speak here")
                
                if st.button("Submit Voice", key="submit_btn") and audio_file:
                    with st.spinner("Processing your speech..."):
                        user_text = transcribe_audio(audio_file)
                        
                        if user_text:
                            st.session_state.history.append(("User", user_text))
                            
                            topic_data = topics[st.session_state.grade][st.session_state.topic]
                            
                            if st.session_state.phase == "vocab_practice":
                                # Process vocab practice response
                                current_vocab = topic_data["vocab"][st.session_state.vocab_index]
                                expected_text = current_vocab["question"]
                                vocab_word = current_vocab["word"]
                                
                                # Comprehensive evaluation
                                evaluation = evaluate_response(
                                    user_text, 
                                    expected_text, 
                                    st.session_state.topic,
                                    vocab_word
                                )
                                
                                st.session_state.pronunciation_scores.append(evaluation["pronunciation_score"])
                                st.session_state.user_responses.append(user_text)
                                st.session_state.evaluations.append(evaluation)
                                
                                # Add evaluation to chat history
                                question_num = st.session_state.vocab_index + 1
                                eval_message = format_evaluation_for_chat(evaluation, question_num)
                                st.session_state.history.append(("Evaluation", eval_message))
                                
                                # Add progress summary to chat history
                                progress_summary = generate_progress_summary(
                                    st.session_state.topic, 
                                    st.session_state.user_responses, 
                                    st.session_state.evaluations
                                )
                                st.session_state.history.append(("Bot", progress_summary))
                                
                                # Move to next vocab word or final challenge
                                st.session_state.vocab_index += 1
                                
                                if st.session_state.vocab_index < len(topic_data["vocab"]):
                                    # Introduce next vocab word
                                    next_word_data = topic_data["vocab"][st.session_state.vocab_index]
                                    bot_response = f"Great job! Now let's learn the next word: **{next_word_data['word']}**.\n\n**Meaning:** {next_word_data['meaning']}\n**Example:** {next_word_data['example']}\n\nReady to practice?"
                                    st.session_state.history.append(("Bot", bot_response))
                                    st.session_state.phase = "vocab_intro"
                                else:
                                    # Move to final challenge
                                    bot_response = "🎉 Excellent! You've practiced all the vocabulary words. Ready for your final challenge? This will test how well you can use all the words together!"
                                    st.session_state.history.append(("Bot", bot_response))
                                    st.session_state.phase = "challenge_intro"
                                
                            elif st.session_state.phase == "challenge":
                                # Process final challenge response
                                expected_text = topic_data["final_challenge"]
                                
                                # Check which vocabulary words were used
                                vocab_words_used = []
                                all_vocab_words = [v["word"] for v in topic_data["vocab"]]
                                for word in all_vocab_words:
                                    if word.lower() in user_text.lower():
                                        vocab_words_used.append(word)
                                
                                # Comprehensive evaluation for final challenge
                                evaluation = evaluate_response(
                                    user_text, 
                                    expected_text, 
                                    st.session_state.topic,
                                    ", ".join(all_vocab_words)
                                )
                                
                                # Adjust evaluation for final challenge
                                if len(vocab_words_used) >= 2:
                                    evaluation["strengths"].append(f"Used {len(vocab_words_used)} vocabulary words: {', '.join(vocab_words_used)}")
                                    evaluation["vocabulary_usage"] = True
                                elif len(vocab_words_used) == 1:
                                    evaluation["strengths"].append(f"Used vocabulary word: {vocab_words_used[0]}")
                                    evaluation["improvements"].append("Try to include more vocabulary words from the lesson")
                                else:
                                    evaluation["improvements"].append("Try to include the vocabulary words from this lesson")
                                
                                # Story length evaluation
                                word_count = len(user_text.split())
                                if word_count >= 15:
                                    evaluation["strengths"].append("Great story length and detail!")
                                elif word_count < 8:
                                    evaluation["improvements"].append("Try to tell a longer, more detailed story")
                                
                                st.session_state.pronunciation_scores.append(evaluation["pronunciation_score"])
                                st.session_state.user_responses.append(user_text)
                                st.session_state.evaluations.append(evaluation)
                                
                                # Add final challenge evaluation to chat history
                                eval_message = format_evaluation_for_chat(evaluation, "Final Challenge")
                                st.session_state.history.append(("Evaluation", eval_message))
                                
                                bot_response = "🎉 Fantastic work! You've completed the entire lesson. Click 'Start New Lesson' to continue practice!"
                                st.session_state.history.append(("Bot", bot_response))
                                st.session_state.phase = "completed"
                                st.session_state.lesson_completed = True
                        else:
                            st.error("Could not transcribe audio. Please try again.")
                    
                    st.rerun()
    
            # End conversation button and progress report
            if st.session_state.phase == "completed":
                st.subheader("🎉 Lesson Complete!")
                
                # Show Get Progress Report button
                if st.button("📊 Get Detailed Progress Report", key="progress_report_btn", type="primary"):
                    # Generate and display final report in chat
                    report = generate_report(f"{st.session_state.grade} - {st.session_state.topic}", 
                                           st.session_state.user_responses, 
                                           st.session_state.pronunciation_scores)
                    
                    # Create comprehensive session summary
                    summary_stats = ""
                    if st.session_state.evaluations:
                        avg_pronunciation = sum(e["pronunciation_score"] for e in st.session_state.evaluations) / len(st.session_state.evaluations)
                        avg_relevance = sum(e["relevance_score"] for e in st.session_state.evaluations) / len(st.session_state.evaluations)
                        excellent_count = sum(1 for e in st.session_state.evaluations if e["overall_rating"] == "Excellent")
                        good_count = sum(1 for e in st.session_state.evaluations if e["overall_rating"] == "Good")
                        
                        summary_stats = f"\n\n📊 **Session Statistics:**\n"
                        summary_stats += f"• Questions Completed: {len(st.session_state.evaluations)}\n"
                        summary_stats += f"• Average Pronunciation: {avg_pronunciation:.1f}%\n"
                        summary_stats += f"• Average Relevance: {avg_relevance:.1f}%\n"
                        summary_stats += f"• Excellent Responses: {excellent_count}/{len(st.session_state.evaluations)}\n"
                        summary_stats += f"• Good Responses: {good_count}/{len(st.session_state.evaluations)}\n"
                        
                        # Calculate improvement trend
                        if len(st.session_state.evaluations) >= 2:
                            first_half = st.session_state.evaluations[:len(st.session_state.evaluations)//2]
                            second_half = st.session_state.evaluations[len(st.session_state.evaluations)//2:]
                            
                            first_avg = sum((e["pronunciation_score"] + e["relevance_score"])/2 for e in first_half) / len(first_half)
                            second_avg = sum((e["pronunciation_score"] + e["relevance_score"])/2 for e in second_half) / len(second_half)
                            
                            if second_avg > first_avg:
                                summary_stats += f"• Learning Trend: 📈 Improving! (+{second_avg - first_avg:.1f} points)\n"
                            elif second_avg < first_avg:
                                summary_stats += f"• Learning Trend: 📊 Steady practice needed ({second_avg - first_avg:.1f} points)\n"
                            else:
                                summary_stats += f"• Learning Trend: 📊 Consistent performance\n"
                    
                    final_report = f"🎯 **Final Progress Report**\n\n{report}{summary_stats}\n\n✨ Thank you for practicing with me today! Keep up the great work!"
                    
                    st.session_state.history.append(("Bot", final_report))
                    st.session_state.phase = "report_shown"
                    st.rerun()
                
                # Show summary statistics in sidebar as preview
                if st.session_state.evaluations:
                    st.markdown("### 📈 Quick Stats Preview")
                    avg_pronunciation = sum(e["pronunciation_score"] for e in st.session_state.evaluations) / len(st.session_state.evaluations)
                    excellent_count = sum(1 for e in st.session_state.evaluations if e["overall_rating"] == "Excellent")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg. Score", f"{avg_pronunciation:.1f}%", 
                                 delta=f"{avg_pronunciation - 70:.1f}" if avg_pronunciation > 70 else None)
                    with col2:
                        st.metric("Excellent", f"{excellent_count}/{len(st.session_state.evaluations)}")
                    
                    # Show vocabulary mastery
                    if st.session_state.topic and st.session_state.grade:
                        topic_data = topics[st.session_state.grade][st.session_state.topic]
                        vocab_used = []
                        for response in st.session_state.user_responses:
                            for vocab_item in topic_data["vocab"]:
                                if vocab_item["word"].lower() in response.lower() and vocab_item["word"] not in vocab_used:
                                    vocab_used.append(vocab_item["word"])
                        
                        st.markdown(f"**Vocabulary Used:** {len(vocab_used)}/{len(topic_data['vocab'])}")
                        if vocab_used:
                            st.markdown(f"✅ {', '.join(vocab_used)}")
            
            # After report is shown, show restart option
            elif st.session_state.phase == "report_shown":
                st.success("✅ Complete progress report generated!")
                if st.button("🔄 Start New Lesson", key="restart_btn", type="secondary"):
                    reset_lesson()
                    st.rerun()
        
        elif st.session_state.lesson_completed or st.session_state.phase == "report_shown":
            st.success("Lesson completed! 🎉")
            st.info("You can view your detailed progress report above or start a new lesson.")
            if st.button("Start New Lesson", key="new_lesson_btn"):
                reset_lesson()
                st.rerun()

if __name__ == "__main__":
    main()