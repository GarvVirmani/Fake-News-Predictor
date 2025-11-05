import streamlit as st
import joblib
import re
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load your trained model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize stemmer and stopwords
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Your Google Gemini API key here
GEMINI_API_KEY = "AIzaSyBuAD0Mz3nYm9aBi1HHh_SBGiYPE6NmXwA"

# Text preprocessing function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [port_stem.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Function to query Google Gemini API with Google Search tool enabled
def query_gemini_api(prompt_text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "tools": [
            {
                "google_search": {}
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data['candidates'][0]['content']['parts'][0]['text']
        return content
    except Exception as e:
        # Return error message as explanation
        return f"Error querying Gemini: {e}"

# Streamlit UI
st.set_page_config(page_title="Real-Time Fake News Detector using LLM and ML", layout="wide")

# === Your original CSS (NO changes here!) ===
st.markdown("""
    <style>
    /* Overall background & text */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #121212 !important;
        color: #e0e0e0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: #121212 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1f1f1f !important;
        border-right: 1px solid #333 !important;
    }
    
    /* Headings style */
    h1, h2, h3 {
        color: #bb86fc !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Input labels - FIXED */
    label {
        color: #e0e0e0 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        display: block !important;
    }

    /* Specific Streamlit label styling */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label {
        color: #e0e0e0 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }

    /* Input boxes */
    .stTextArea textarea, .stTextInput input {
        background-color: #1f1f1f !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
        transition: border-color 0.3s ease;
        width: 100% !important;
        box-sizing: border-box !important;
        caret-color: #bb86fc !important; /* Cursor color */
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #bb86fc !important;
        outline: none !important;
        box-shadow: 0 0 8px #bb86fcaa !important;
    }

    /* Text selection color */
    .stTextArea textarea::selection, .stTextInput input::selection {
        background-color: #bb86fc !important;
        color: #121212 !important;
    }

    /* Placeholder text color */
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #888 !important;
        opacity: 1 !important;
    }

    /* Button style */
    .stButton button {
        background-color: #bb86fc !important;
        color: #121212 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 12px 24px !important;
        width: 100% !important;
        border-radius: 10px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(187, 134, 252, 0.5) !important;
    }
    
    .stButton button:hover {
        background-color: #9a54e0 !important;
        color: #fff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(187, 134, 252, 0.7) !important;
    }

    /* Markdown text */
    .stMarkdown {
        color: #e0e0e0 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Success/Warning/Info boxes */
    .stAlert {
        background-color: #1f1f1f !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #bb86fc transparent transparent transparent !important;
    }

    /* Results boxes */
    [data-testid="stExpander"] {
        background-color: #1f1f1f !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }

    /* Text elements */
    .stText, .stMarkdown p {
        color: #e0e0e0 !important;
    }

    /* Custom result boxes */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) {
        background-color: #1f1f1f !important;
        padding: 20px !important;
        border-radius: 10px !important;
        border: 1px solid #333 !important;
        margin: 10px 0 !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #121212;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #bb86fc;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #9a54e0;
    }

    /* Specific styling for text areas */
    textarea::-webkit-scrollbar {
        width: 8px;
    }
    
    textarea::-webkit-scrollbar-track {
        background: #1f1f1f;
    }
    
    textarea::-webkit-scrollbar-thumb {
        background-color: #bb86fc;
        border-radius: 4px;
    }

    /* Fix for any remaining white elements */
    .css-1d391kg, .css-1kyxreq, .css-2trqyj, .css-1v3fvcr {
        background-color: #121212 !important;
        color: #e0e0e0 !important;
    }

    /* Card-like containers */
    .element-container {
        background-color: #1f1f1f !important;
        padding: 15px !important;
        border-radius: 10px !important;
        margin: 10px 0 !important;
        border: 1px solid #333 !important;
    }

    /* Custom badge for results */
    .final-verdict {
        background: linear-gradient(135deg, #bb86fc, #9a54e0) !important;
        color: #121212 !important;
        padding: 15px 25px !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        font-size: 24px !important;
        text-align: center !important;
        margin: 20px 0 !important;
        border: 2px solid #9a54e0 !important;
    }

    /* Specific fix for Streamlit widget labels */
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stTextInput"] label div,
    div[data-testid="stTextArea"] label div {
        color: #e0e0e0 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* Ensure all p tags in labels are visible */
    p {
        color: #e0e0e0 !important;
    }

    /* Cursor styling for better visibility */
    input, textarea {
        caret-color: #bb86fc !important;
    }

    /* Focus states for better accessibility */
    .stTextInput input:focus, .stTextArea textarea:focus {
        caret-color: #bb86fc !important;
    }

    </style>
""", unsafe_allow_html=True)


st.title("📰 Real-Time Fake News Detector using LLM and ML")
st.write("Enter a news **Title** and **Text** below to predict whether it is REAL or FAKE using both ML model and Gemini LLM")

title = st.text_input("News Title")
text = st.text_area("News Text")

if st.button("Detect News"):
    if title.strip() and text.strip():
        with st.spinner("Analyzing news with ML model and Gemini LLM..."):
            content = title + ' ' + text
            cleaned_content = clean_text(content)

            # ML Model prediction
            vect = vectorizer.transform([cleaned_content])
            ml_pred = model.predict(vect)[0]
            ml_result = "REAL 🟢" if ml_pred == 0 else "FAKE 🔴"

            # Query Gemini LLM with Google Search tool enabled
            gemini_prompt = f"Is the following news real or fake? Please answer always first word with REAL. or FAKE. and provide a short explanation: Also give sources always clickable links \n\n{content}"
            gemini_response = query_gemini_api(gemini_prompt).strip()

        # Determine final verdict & source
        gemini_upper = gemini_response.upper()
        print(gemini_upper)
        if (gemini_upper.startswith("REAL.") and ml_result.startswith("REAL")) or (gemini_upper.startswith("FAKE.") and ml_result.startswith("FAKE")):
            final_verdict = ml_result
            source = "ML + LLM"
        elif gemini_upper.startswith("REAL.") or gemini_upper.startswith("FAKE."):
            # LLM disagrees with ML → trust LLM only
            if gemini_upper.startswith("REAL."):
                final_verdict = "REAL 🟢"
            else:
                final_verdict = "FAKE 🔴"
            source = "LLM only"
        else:
            # LLM unclear → fallback to ML only
            final_verdict = "NOT CLEAR"
            source = "ML only"

        st.markdown(f"🧠 **Final Verdict:** {final_verdict}")

        st.markdown("🤖 **Explanation:**")
        st.write(gemini_response)
    else:
        st.warning("⚠️ Please enter both news title and text!")
