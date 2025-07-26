# clauseiq_app.py
# Streamlit App for ClauseIQ with CounterClause and LawyerConnect – Dashboard-Style UI (iPadOS Inspired)
# Now using Google Gemini API for LLM decisions, with Firebase for Authentication.

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
import requests # For making HTTP requests to the Gemini API

# Firebase Imports for Authentication and Firestore
import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- Firebase Initialization ---
# This block initializes Firebase only once per Streamlit app run.
# It uses credentials provided by the Canvas environment.
if not firebase_admin._apps:
    try:
        # __firebase_config is a global variable provided by the Canvas environment
        firebase_config = json.loads(__firebase_config)
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
    except NameError:
        st.error("Firebase configuration not found. Please ensure __firebase_config is set in the environment.")
        st.session_state.firebase_initialized = False
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")
        st.session_state.firebase_initialized = False

db = firestore.client() # Initialize Firestore client

# --- Streamlit Session State for Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'auth_error' not in st.session_state:
    st.session_state.auth_error = ""

# --- Load LLM and Embedding Model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Gemini API Configuration ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Assuming the secret is named GEMINI_API_KEY
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Sample Clause Database ---
clauses = [
    "Clause 5.1: Surgery covered only after 4 months of continuous policy.",
    "Clause 3.2: Surgery due to accident may be exempt from waiting period.",
    "Clause 6.4: Portability clause allows prior policy duration to be counted."
]

embeddings = model.encode(clauses)
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# --- Helper Functions ---
def get_top_clause(user_query):
    """
    Finds the top 2 most similar clauses from the database based on the user's query.
    """
    query_vec = model.encode([user_query])
    D, I = index.search(query_vec, k=2)
    return [clauses[i] for i in I[0]]

def get_llm_decision(user_query, matched_clauses):
    """
    Uses the Google Gemini API (gemini-2.0-flash) to make a decision
    based on the user query and matched clauses.
    """
    context = "\n".join(matched_clauses)
    prompt = f"""
You are an insurance claim reasoning assistant.
User Query: "{user_query}"
Relevant Clauses:
{context}

Based on the clauses and query, respond with a JSON containing:
- decision: approved or rejected
- reason: short explanation
- counterclause: any alternate way the user might still be approved
- clause_reference: the clause(s) used

Respond only with JSON.
"""

    headers = {
        'Content-Type': 'application/json'
    }
    params = {
        'key': GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            return json_string
        else:
            st.error(f"Gemini API response format unexpected: {result}")
            return json.dumps({"decision": "error", "reason": "Unexpected API response format", "counterclause": "", "clause_reference": ""})

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return json.dumps({"decision": "error", "reason": f"API call failed: {e}", "counterclause": "", "clause_reference": ""})
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Gemini API: {e}. Raw response: {response.text}")
        return json.dumps({"decision": "error", "reason": f"Invalid JSON from API: {e}", "counterclause": "", "clause_reference": ""})

# --- Authentication Functions (for UI simulation) ---
def login_user(email, password):
    """Simulates login. In a real app, this would use Firebase Auth."""
    st.session_state.auth_error = ""
    try:
        # In the Canvas environment, we rely on __initial_auth_token
        # For a real web app, you'd use auth.sign_in_with_email_and_password(email, password)
        # For demonstration, we'll just check if the token exists.
        if st.session_state.firebase_initialized and '__initial_auth_token' in globals() and __initial_auth_token:
            # Simulate successful login if token is present
            st.session_state.logged_in = True
            st.session_state.user_id = auth.verify_id_token(__initial_auth_token)['uid']
            st.success(f"Logged in as {st.session_state.user_id}")
            st.rerun()
        else:
            st.session_state.auth_error = "Login failed: No authentication token available. For real login, connect to Firebase Auth."
    except Exception as e:
        st.session_state.auth_error = f"Login failed: {e}"

def create_account(email, password):
    """Simulates account creation. In a real app, this would use Firebase Auth."""
    st.session_state.auth_error = ""
    try:
        # In a real web app, you'd use auth.create_user(email=email, password=password)
        # For demonstration, we'll just show a success message.
        if st.session_state.firebase_initialized:
            st.success(f"Account creation simulated for {email}. In a real app, this would create a user.")
            st.session_state.auth_error = "Please note: Actual user creation requires Firebase Admin SDK in a backend, not directly in Streamlit frontend."
        else:
            st.session_state.auth_error = "Account creation failed: Firebase not initialized."
    except Exception as e:
        st.session_state.auth_error = f"Account creation failed: {e}"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.auth_error = ""
    st.success("Logged out successfully!")
    st.rerun()

# --- Streamlit UI ---
st.set_page_config(page_title="ClauseIQ Dashboard", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    /* Overall background with a subtle gradient */
    body {
        background: linear-gradient(135deg, #F0F8FF 0%, #E6E6FA 100%); /* Light blue to lavender gradient */
        background-attachment: fixed; /* Keep gradient fixed on scroll */
    }
    /* Main content blocks (cards) */
    .main, .block-container {
        background-color: #FFFFFF !important; /* Pure white for cards */
        border-radius: 25px; /* More rounded corners */
        padding: 2rem; /* Increased padding */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08); /* Stronger, softer shadow */
        margin-bottom: 20px; /* Space between main blocks */
    }
    /* Buttons - primary action */
    .stButton > button {
        background-color: #6A5ACD; /* A soft, rich purple */
        color: #fff;
        border: none;
        border-radius: 20px; /* Very rounded buttons */
        padding: 0.8rem 1.6rem;
        font-size: 17px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 6px 12px rgba(106, 90, 205, 0.4); /* Deeper purple shadow */
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #5B42D1; /* Slightly darker purple on hover */
        transform: translateY(-3px); /* More pronounced lift effect */
        box-shadow: 0 8px 16px rgba(106, 90, 205, 0.5);
    }
    /* Download button - secondary action */
    .stDownloadButton > button {
        background-color: #87CEEB; /* A soft sky blue */
        color: white;
        border-radius: 20px;
        padding: 0.8rem 1.6rem;
        font-size: 17px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 6px 12px rgba(135, 206, 235, 0.4); /* Deeper blue shadow */
        cursor: pointer;
    }
    .stDownloadButton > button:hover {
        background-color: #7AC5E2; /* Slightly darker blue on hover */
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(135, 206, 235, 0.5);
    }
    /* Text input fields */
    .stTextInput > div > div > input {
        border-radius: 18px; /* Rounded input fields */
        padding: 14px 18px;
        border: 1px solid #D8BFD8; /* Medium lavender border */
        background-color: #FDFDFE; /* Very light background */
        color: #333333; /* Darker text for readability */
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05); /* Subtle inner shadow */
    }
    .stTextInput > label { /* Style for text input labels */
        color: #5B42D1; /* Purple label */
        font-weight: bold;
    }
    /* Expander headers */
    .streamlit-expanderHeader {
        background-color: #F5EEF8; /* Light pastel purple */
        border-radius: 18px;
        padding: 14px 18px;
        font-weight: bold;
        color: #5B42D1 !important; /* Purple text for expander header */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    /* General Markdown text */
    .stMarkdown, .markdown-text-container {
        font-family: 'Inter', sans-serif; /* Using Inter font for modern look */
        font-size: 17px;
        line-height: 1.6;
        color: #333333 !important; /* Force dark gray for body text */
    }
    /* Subheaders and Title */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #483D8B !important; /* Darker royal blue/purple for headings */
        font-weight: 700; /* Bolder headings */
    }
    /* Specific styling for st.caption to ensure readability */
    div[data-testid="stCaptionContainer"] p {
        color: #666666 !important; /* Slightly lighter gray for caption text */
        font-size: 15px;
    }
    /* Specific styling for st.metric labels and values */
    div[data-testid="stMetricLabel"] div {
        color: #483D8B !important; /* Darker purple for metric labels */
        font-weight: 600;
        font-size: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #333333 !important; /* Dark gray for metric values */
        font-size: 28px; /* Larger metric values */
        font-weight: 700;
    }

    /* Streamlit's default success/info/warning messages */
    .stAlert {
        border-radius: 15px; /* Rounded alerts */
        padding: 15px;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    /* Customizing Streamlit's default success message for softer green */
    .stAlert.success {
        background-color: #E6F7E6; /* Light pastel green */
        color: #2F855A; /* Darker green text */
        border-left: 6px solid #48BB78; /* Green border */
    }
    /* Customizing Streamlit's default info message for softer blue */
    .stAlert.info {
        background-color: #E0F2F7; /* Light pastel blue */
        color: #2B6CB0; /* Darker blue text */
        border-left: 6px solid #4299E1; /* Blue border */
    }
    /* Customizing Streamlit's default error message for softer red */
    .stAlert.error {
        background-color: #FEE8E8; /* Light pastel red */
        color: #C53030; /* Darker red text */
        border-left: 6px solid #E53E3E; /* Red border */
    }

    /* Custom style for the login/signup container */
    .auth-container {
        background-color: #FFFFFF;
        border-radius: 25px;
        padding: 30px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        max-width: 500px;
        margin: 50px auto; /* Center the container */
        text-align: center;
    }
    .auth-container h2 {
        color: #483D8B !important;
        margin-bottom: 20px;
    }
    .auth-container .stTextInput {
        margin-bottom: 15px;
    }
    .auth-container .stButton {
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main App Logic ---
def main_app_page():
    """Displays the main ClauseIQ dashboard content."""
    # Display a logo image and the main title of the application.
    # Using a more thematic icon image for the logo
    st.image("https://www.flaticon.com/svg/static/icons/svg/2924/2924976.svg", width=70) # Placeholder for a legal/justice icon
    st.title("🧠 ClauseIQ – Smart Insurance Decision Dashboard")
    st.caption("Claim eligibility, clause discovery, rebuttals & lawyer escalation – now seamless.")

    # Create two columns for the main layout: Claim Analyzer on the left, System Status on the right.
    with st.container():
        left, right = st.columns([3, 2]) # 3:2 ratio for columns
        with left:
            st.subheader("💬 Claim Analyzer")
            # Expander to show example queries, keeping the UI clean.
            with st.expander("📌 Example Queries"):
                st.markdown("- My mom had breast surgery after 3 months of insurance, will it be covered?")
                st.markdown("- Dad's accident-based knee operation, policy started in January. Can we claim?")
            # Text input field for the user to describe their situation.
            user_query = st.text_input("Describe your situation:",
                                         placeholder="e.g. My dad had knee surgery, policy is 3 months old")

        with right:
            st.subheader("📊 System Status")
            # Display metrics for system information.
            st.metric("Policy Index", "3 files") # Placeholder for number of policies indexed
            st.metric("Clause Match Confidence", "97.6%") # Placeholder for confidence level
            st.metric("Uptime", "100%") # Placeholder for system uptime

    # Conditional display of results once a user query is entered.
    if user_query:
        # Show a spinner while the AI is processing the query.
        with st.spinner("🔍 Analyzing your policy..."):
            top_clauses = get_top_clause(user_query)
            decision = get_llm_decision(user_query, top_clauses)
            try:
                parsed = json.loads(decision)
            except json.JSONDecodeError:
                st.error("Failed to parse JSON response from Gemini. Please try again.")
                parsed = {"decision": "error", "reason": "Invalid JSON response", "counterclause": "", "clause_reference": ""}

        st.success("✅ AI Decision Complete")

        a, b = st.columns([2, 1])
        with a:
            st.markdown("### 🤖 Decision Summary")
            st.json(parsed)
        with b:
            st.markdown("### 📄 Clauses Matched")
            for clause in top_clauses:
                st.markdown(f"- {clause}")

        st.markdown("---")
        st.markdown("## 👨‍⚖️ LawyerConnect™")
        st.info("Still unclear? Request legal support for escalation.")

        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("🔗 Book Legal Review"):
                st.success("A legal advisor will reach out soon. Your case file is summarized below.")

        with col4:
            if st.button("📤 Export Summary"):
                summary = f"""
                CLAIM CASE SUMMARY

                User Query: {user_query}
                Decision: {parsed['decision']}
                Reason: {parsed['reason']}
                Suggested Rebuttal: {parsed['counterclause']}
                Referenced Clause: {parsed['clause_reference']}
                """
                st.download_button("Download Summary", summary, file_name="claim_summary.txt")

    # Logout button
    st.sidebar.button("Logout", on_click=logout_user)
    if st.session_state.user_id:
        st.sidebar.markdown(f"**Logged in as:** `{st.session_state.user_id}`")
        st.sidebar.markdown(f"**App ID:** `{__app_id}`") # Display app ID for Firestore reference

def login_page():
    """Displays the login/account creation interface."""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.subheader("Welcome to ClauseIQ")
    st.markdown("### Login or Create Account")

    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", key="login_btn"):
            login_user(email, password)
    with col2:
        if st.button("Create Account", key="create_account_btn"):
            create_account(email, password)

    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 20px; color: #666;">
        <p>Note: For a real deployed app, Firebase Email/Password authentication would be enabled and handled via a backend.
        In this Canvas environment, login is simulated if an initial token is available.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main App Entry Point ---
# Check if Firebase is initialized and if an initial auth token is available (from Canvas environment)
if st.session_state.firebase_initialized and '__initial_auth_token' in globals() and __initial_auth_token and not st.session_state.logged_in:
    # Attempt to sign in with the provided custom token
    try:
        user = auth.sign_in_with_custom_token(__initial_auth_token)
        st.session_state.logged_in = True
        st.session_state.user_id = user.uid
        st.success(f"Automatically logged in as {st.session_state.user_id}")
        st.rerun() # Rerun to switch to main app page
    except Exception as e:
        st.session_state.auth_error = f"Automatic login failed: {e}"
        login_page() # Show login page if auto-login fails
elif st.session_state.logged_in:
    main_app_page()
else:
    login_page()
