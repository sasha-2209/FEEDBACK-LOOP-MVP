import os
import json
import pandas as pd
import streamlit as st
from classifier import analyze_texts_batch

# App title & layout
st.set_page_config(page_title="💬 Feedback Loop 2.0", layout="wide")
st.title("💬 Feedback Loop 2.0 – Smart NPS / Feedback Classifier")

st.markdown("""
This tool classifies user feedback from **Jira**, **Slack**, or **manual uploads**  
into actionable categories using the **Groq Llama 3.1** model.
""")

# --- Sidebar setup ---
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose Input Source:", [
    "📄 Upload CSV",
    "🧠 Enter Feedback Manually",
    "💬 Connect Jira (coming soon)",
    "⚡ Connect Slack (coming soon)"
])

texts = []

# --- File upload mode ---
if input_mode == "📄 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV with feedback", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())
        text_col = st.selectbox("Select column containing feedback:", df.columns)
        texts = df[text_col].dropna().tolist()

# --- Manual input mode ---
elif input_mode == "🧠 Enter Feedback Manually":
    user_input = st.text_area(
        "Paste feedback (one per line):",
        placeholder="e.g.\nBilling invoices are incorrect.\nDashboard is slow.\nI love the new reports feature!"
    )
    if user_input:
        texts = [t.strip() for t in user_input.split("\n") if t.strip()]

# --- Coming soon modes ---
elif input_mode in ["💬 Connect Jira (coming soon)", "⚡ Connect Slack (coming soon)"]:
    st.info("This integration is under development. Stay tuned for OAuth login and API sync!")

# --- Run classification ---
if texts:
    if st.button("🚀 Run Classification"):
        with st.spinner("Analyzing feedback..."):
            results = analyze_texts_batch(texts)

        st.success(f"✅ Classified {len(results)} feedback items.")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # Download button
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, "classified_feedback.csv", "text/csv")

        # Summary section
        st.markdown("### 📊 Summary Insights")
        summary = df_results["category"].value_counts().reset_index()
        summary.columns = ["Category", "Count"]
        st.bar_chart(summary.set_index("Category"))

else:
    st.info("👆 Choose an input mode and provide feedback to begin analysis.")

# --- Footer ---
st.markdown("""
---
Powered by **Groq Llama 3.1** & **Streamlit**
""")
