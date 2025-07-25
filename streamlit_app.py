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
    token = st.sidebar.text_input("Enter your HuggingFace token", type="password")
    if st.sidebar.button("Submit Token"):
        if validate_hf_token(token):
            os.environ["HF_TOKEN"] = token
            st.session_state["hf_token"] = token
            st.success("Token saved. Loading models...")
            st.rerun()
        else:
            st.sidebar.error("Invalid token format. Must start with 'hf_' and be at least 30 characters long.")

if "hf_token" not in st.session_state:
    prompt_for_token()
    st.stop()
else:
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
    if 'hf_token_validated' not in st.session_state:
        st.session_state.hf_token_validated = False
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
    if 'stt_model' not in st.session_state:
        with st.spinner("Loading speech recognition model..."):
            st.session_state.stt_model = WhisperModel("base", device="cpu", compute_type="int8")
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
        st.session_state.llm_tokenizer = None
        try:
            with st.spinner("Loading language model..."):
                model_name = "google/flan-t5-base"
                st.session_state.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                st.session_state.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            st.warning(f"Could not load language model: {e}. Using fallback responses.")

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
    """Clean text for text-to-speech"""
    text = re.sub(r"\*\*|[*_:]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
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
    cleaned_text = clean_text_for_tts(text)
    tts = gTTS(cleaned_text)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()

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

def display_evaluation(evaluation: dict, question_number: int):
    """Display comprehensive evaluation results"""
    st.markdown("### 📊 Question Evaluation")
    
    # Create columns for scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pronunciation", f"{evaluation['pronunciation_score']}%")
    with col2:
        st.metric("Relevance", f"{evaluation['relevance_score']}%")
    with col3:
        # Overall rating with color
        rating = evaluation['overall_rating']
        if rating == "Excellent":
            st.markdown(f"**Overall:** 🌟 {rating}")
        elif rating == "Good":
            st.markdown(f"**Overall:** 👍 {rating}")
        elif rating == "Fair":
            st.markdown(f"**Overall:** ⚡ {rating}")
        else:
            st.markdown(f"**Overall:** 📚 {rating}")
    
    # Vocabulary usage indicator
    if evaluation['vocabulary_usage']:
        st.success("✅ Target vocabulary used correctly!")
    else:
        st.warning("⚠️ Target vocabulary not detected in response")
    
    # Strengths and improvements in expandable sections
    if evaluation['strengths']:
        with st.expander("💪 Strengths", expanded=True):
            for strength in evaluation['strengths']:
                st.write(f"• {strength}")
    
    if evaluation['improvements']:
        with st.expander("🎯 Areas for Improvement", expanded=True):
            for improvement in evaluation['improvements']:  
                st.write(f"• {improvement}")
    
    # Grammar feedback in expandable section
    with st.expander("📝 Grammar Feedback"):
        cleaned_feedback = clean_text_for_display(evaluation['grammar_feedback'])
        st.write(cleaned_feedback)

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

def generate_report(topic, user_responses, pronunciation_scores):
    """Generate progress report"""
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

def display_history():
    """Display conversation history with cleaned text"""
    for role, message in st.session_state.history:
        # Clean the message for display
        cleaned_message = clean_text_for_display(message)
        
        if role == "Bot":
            st.markdown(f'<div class="message-bot"><strong>🤖 Bot:</strong> {cleaned_message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-user"><strong>👤 You:</strong> {cleaned_message}</div>', unsafe_allow_html=True)

def reset_lesson():
    """Reset all lesson state"""
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

# Main app
def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header"> SINE English Speaking Tutor</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
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
            
            bot_intro = f"Hi there! I'm your SINE English speaking assistant. Today we're learning about {st.session_state.topic} for {st.session_state.grade}. You'll practice 3 useful words. Let's start with the first word: {first_vocab['word']}."
            bot_explanation = f"\nMeaning: {first_vocab['meaning']}\nExample: {first_vocab['example']}\n\nReady to practice using this word?"
            full_bot_response = bot_intro + bot_explanation
            
            st.session_state.history.append(("Bot", full_bot_response))
            st.rerun()
        
        # Display conversation history
        if st.session_state.history:
            display_history()
            
            # Add audio for the latest bot message (only if it's new)
            if (st.session_state.history and 
                st.session_state.history[-1][0] == "Bot" and 
                st.session_state.history[-1][1] != st.session_state.last_audio_message):
                
                latest_bot_message = st.session_state.history[-1][1]
                try:
                    with st.spinner("🔊 Generating audio..."):
                        audio_bytes = text_to_speech(latest_bot_message)
                        st.audio(audio_bytes, format="audio/mp3")
                        st.session_state.last_audio_message = latest_bot_message
                except Exception as e:
                    st.warning(f"Could not generate audio: {e}")
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
                        bot_response = f"Now let's practice! Question: {current_word['question']}\n\nPlease record your answer using the microphone below."
                        st.session_state.history.append(("Bot", bot_response))
                        st.session_state.phase = "vocab_practice"
                        
                    elif st.session_state.phase == "vocab_intro":
                        # Continue after introducing a new vocab word
                        current_word = topic_data["vocab"][st.session_state.vocab_index]
                        bot_response = f"Now let's practice this word! Question: {current_word['question']}\n\nPlease record your answer."
                        st.session_state.history.append(("Bot", bot_response))
                        st.session_state.phase = "vocab_practice"
                        
                    elif st.session_state.phase == "challenge_intro":
                        # Continue to final challenge
                        bot_response = f"Here's your final challenge:\n\n{topic_data['final_challenge']}\n\nPlease record your story."
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
                                st.session_state.evaluations.append(evaluation)
                                eval_summary_msg = f"**Evaluation Summary:**\n- Pronunciation: {evaluation['pronunciation_score']}%\n- Relevance: {evaluation['relevance_score']}%\n- Vocabulary Used: {'✅' if evaluation['vocabulary_usage'] else '❌'}\n- Overall: {evaluation['overall_rating']}"
                                st.session_state.history.append(("Bot", eval_summary_msg))
                                
                                # Display comprehensive evaluation
                                question_num = st.session_state.vocab_index + 1
                                st.markdown(f"#### Question {question_num} Results")
                                display_evaluation(evaluation, question_num)
                                
                                # Move to next vocab word or final challenge
                                st.session_state.vocab_index += 1
                                
                                if st.session_state.vocab_index < len(topic_data["vocab"]):
                                    # Introduce next vocab word
                                    next_word_data = topic_data["vocab"][st.session_state.vocab_index]
                                    bot_response = f"Great job! Now let's learn the next word: {next_word_data['word']}.\nMeaning: {next_word_data['meaning']}\nExample: {next_word_data['example']}\n\nReady to practice?"
                                    st.session_state.history.append(("Bot", bot_response))
                                    st.session_state.phase = "vocab_intro"
                                else:
                                    # Move to final challenge
                                    bot_response = "Excellent! You've practiced all the vocabulary words. Ready for your final challenge?"
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
                                
                                # Display final challenge evaluation
                                st.markdown("#### 🏆 Final Challenge Results")
                                display_evaluation(evaluation, "Final")
                                
                                bot_response = "Fantastic work! You've completed the entire lesson."
                                st.session_state.history.append(("Bot", bot_response))
                                st.session_state.phase = "completed"
                                st.session_state.lesson_completed = True
                        else:
                            st.error("Could not transcribe audio. Please try again.")
                    
                    st.rerun()
    
            # End conversation button
            if st.session_state.phase == "completed":
                if st.button("Get Progress Report", key="end_btn"):
                    # Display comprehensive session summary
                    st.markdown("## 📊 Complete Session Summary")
                    
                    # Overall statistics
                    if st.session_state.evaluations:
                        avg_pronunciation = sum(e["pronunciation_score"] for e in st.session_state.evaluations) / len(st.session_state.evaluations)
                        avg_relevance = sum(e["relevance_score"] for e in st.session_state.evaluations) / len(st.session_state.evaluations)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Questions Completed", len(st.session_state.evaluations))
                        with col2:
                            st.metric("Avg. Pronunciation", f"{avg_pronunciation:.1f}%")
                        with col3:
                            st.metric("Avg. Relevance", f"{avg_relevance:.1f}%")
                        with col4:
                            excellent_count = sum(1 for e in st.session_state.evaluations if e["overall_rating"] == "Excellent")
                            st.metric("Excellent Responses", f"{excellent_count}/{len(st.session_state.evaluations)}")
                    
                    # Individual question breakdown
                    st.markdown("### 📝 Question-by-Question Breakdown")
                    for i, evaluation in enumerate(st.session_state.evaluations):
                        question_type = "Final Challenge" if i == len(st.session_state.evaluations) - 1 and st.session_state.phase == "completed" and len(st.session_state.evaluations) > 3 else f"Question {i+1}"
                        
                        with st.expander(f"{question_type} - {evaluation['overall_rating']} ({evaluation['pronunciation_score']}% pronunciation)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Your Response:**")
                                st.write(f'"{st.session_state.user_responses[i]}"')
                            with col2:
                                st.write("**Evaluation:**")
                                st.write(f"• Pronunciation: {evaluation['pronunciation_score']}%")
                                st.write(f"• Relevance: {evaluation['relevance_score']}%")
                                st.write(f"• Vocabulary Used: {'✅' if evaluation['vocabulary_usage'] else '❌'}")
                            
                            if evaluation['strengths']:
                                st.write("**Strengths:**")
                                for strength in evaluation['strengths']:
                                    st.write(f"• {strength}")
                    
                    # Generate and display final report
                    report = generate_report(f"{st.session_state.grade} - {st.session_state.topic}", 
                                           st.session_state.user_responses, 
                                           st.session_state.pronunciation_scores)
                    
                    report_msg = f"📊 Your Progress Report\n\n{report}\n\nThank you for practicing with me today! Keep up the great work! 🌟"
                    
                    # Clean the report for display
                    cleaned_report = clean_text_for_display(report_msg)
                    st.markdown(f'<div class="report-box">{cleaned_report}</div>', unsafe_allow_html=True)
        
        elif st.session_state.lesson_completed:
            st.success("Lesson completed! 🎉")
            if st.button("Start New Lesson"):
                reset_lesson()
                st.rerun()

if __name__ == "__main__":
    main()