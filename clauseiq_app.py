# clauseiq_app.py
# Streamlit App for ClauseIQ with CounterClause and LawyerConnect – Dashboard-Style UI (iPadOS Inspired)

import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
import faiss
import json

# --- Load LLM and Embedding Model ---
# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get your OpenAI API key from Streamlit secrets.
# This is the recommended secure way to handle API keys in Streamlit apps.
# Ensure you have a .streamlit/secrets.toml file with OPENAI_API_KEY="your_key_here"
# and that secrets.toml is NOT committed to your public GitHub repository.
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Sample Clause Database ---
# A list of sample insurance clauses for demonstration purposes.
# In a real application, this would be loaded from a more robust database.
clauses = [
    "Clause 5.1: Surgery covered only after 4 months of continuous policy.",
    "Clause 3.2: Surgery due to accident may be exempt from waiting period.",
    "Clause 6.4: Portability clause allows prior policy duration to be counted."
]

# Generate embeddings for the sample clauses using the SentenceTransformer model.
# These embeddings are numerical representations of the clauses, allowing for semantic search.
embeddings = model.encode(clauses)

# Initialize a FAISS index for efficient similarity search.
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
# IndexFlatL2 uses L2 (Euclidean) distance for similarity measurement.
index = faiss.IndexFlatL2(len(embeddings[0])) # Create an index with the dimension of the embeddings
index.add(embeddings) # Add the clause embeddings to the FAISS index

# --- Helper Functions ---
def get_top_clause(user_query):
    """
    Finds the top 2 most similar clauses from the database based on the user's query.

    Args:
        user_query (str): The text query provided by the user.

    Returns:
        list: A list containing the text of the top 2 matching clauses.
    """
    # Encode the user query into a vector.
    query_vec = model.encode([user_query])
    # Search the FAISS index for the top 2 nearest neighbors (clauses).
    # D contains distances, I contains indices of the nearest neighbors.
    D, I = index.search(query_vec, k=2)
    # Return the actual clause texts corresponding to the found indices.
    return [clauses[i] for i in I[0]]

def get_llm_decision(user_query, matched_clauses):
    """
    Uses the OpenAI GPT-4 model to make a decision based on the user query and matched clauses.

    Args:
        user_query (str): The original query from the user.
        matched_clauses (list): A list of clauses identified as relevant.

    Returns:
        str: A JSON string containing the decision, reason, counterclause, and clause reference.
    """
    # Format the matched clauses into a context string for the LLM.
    context = "\n".join(matched_clauses)
    # Construct the prompt for the OpenAI model.
    # The prompt instructs the LLM to act as an insurance claim reasoning assistant
    # and to respond strictly in JSON format.
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
    # Make a call to the OpenAI Chat Completion API using the gpt-4 model.
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}] # Pass the prompt as a user message.
    )
    # Extract and return the content of the model's response.
    return res.choices[0].message['content']

# --- Streamlit UI ---
# Configure the Streamlit page settings for a wide layout and a custom title/icon.
st.set_page_config(page_title="ClauseIQ Dashboard", page_icon="🧠", layout="wide")

# Apply custom CSS for styling the Streamlit app.
# This makes the UI look more like an iPadOS dashboard with rounded corners,
# custom button styles, and a background image.
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
        background-size: cover;
    }
    .main, .block-container {
        background-color: rgba(255, 255, 255, 0.85) !important; /* Semi-transparent white background */
        border-radius: 15px; /* Rounded corners for main content blocks */
        padding: 1.5rem; /* Padding inside the content blocks */
    }
    .stButton > button {
        background-color: #111827; /* Dark background for buttons */
        color: #fff; /* White text for buttons */
        border: none;
        border-radius: 12px; /* Rounded corners for buttons */
        padding: 0.6rem 1.2rem;
        font-size: 15px;
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }
    .stButton > button:hover {
        background-color: #374151; /* Slightly lighter on hover */
    }
    .stDownloadButton > button {
        background-color: #2563eb; /* Blue background for download button */
        color: white;
        border-radius: 10px;
        padding: 0.4rem 1rem;
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }
    .stDownloadButton > button:hover {
        background-color: #3b82f6; /* Slightly lighter blue on hover */
    }
    .stMarkdown, .markdown-text-container {
        font-family: 'Helvetica Neue', sans-serif; /* Modern, clean font */
        font-size: 16px;
    }
    /* Style for text input */
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    /* Style for expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Display a logo image and the main title of the application.
st.image("https://cdn-icons-png.flaticon.com/512/1041/1041916.png", width=70)
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
        # Get the top matching clauses from the FAISS index.
        top_clauses = get_top_clause(user_query)
        # Get the AI's decision from the OpenAI model.
        decision = get_llm_decision(user_query, top_clauses)
        # Parse the JSON response from the LLM.
        parsed = json.loads(decision)

    # Indicate that the AI decision is complete.
    st.success("✅ AI Decision Complete")

    # Create two columns for displaying the decision summary and matched clauses.
    a, b = st.columns([2, 1])
    with a:
        st.markdown("### 🤖 Decision Summary")
        # Display the parsed JSON decision.
        st.json(parsed)
    with b:
        st.markdown("### 📄 Clauses Matched")
        # Display each of the matched clauses.
        for clause in top_clauses:
            st.markdown(f"- {clause}")

    # Separator line.
    st.markdown("---")
    st.markdown("## 👨‍⚖️ LawyerConnect™")
    st.info("Still unclear? Request legal support for escalation.")

    # Create two columns for action buttons.
    col3, col4 = st.columns([1, 1])
    with col3:
        # Button to simulate booking a legal review.
        if st.button("🔗 Book Legal Review"):
            st.success("A legal advisor will reach out soon. Your case file is summarized below.")

    with col4:
        # Button to export the summary of the claim.
        if st.button("📤 Export Summary"):
            # Format the summary text.
            summary = f"""
            CLAIM CASE SUMMARY

            User Query: {user_query}
            Decision: {parsed['decision']}
            Reason: {parsed['reason']}
            Suggested Rebuttal: {parsed['counterclause']}
            Referenced Clause: {parsed['clause_reference']}
            """
            # Provide a download button for the summary.
            st.download_button("Download Summary", summary, file_name="claim_summary.txt")