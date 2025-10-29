import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from jira_connector import fetch_jira_issues
from mapper import map_feedback_to_dealblockers
from classifier import analyze_texts_batch

load_dotenv()

st.set_page_config(page_title="Feedback ↔️ Dealblocker Intelligence", layout="wide")

st.title("🧠 Feedback & Dealblocker Correlation Dashboard")

# --- SECTION 1: FEEDBACK UPLOAD ---
st.header("📤 Upload Product Feedback")
feedback_file = st.file_uploader("Upload feedback CSV", type=["csv"])

feedback_df = None
if feedback_file is not None:
    feedback_df = pd.read_csv(feedback_file)
    st.write("✅ Feedback Data Preview:")
    st.dataframe(feedback_df.head())

# --- SECTION 2: JIRA FILTER INPUT ---
st.header("🎯 Jira Dealblocker Fetch")

project = st.text_input("Project Key (e.g., SDK)", value="SDK")
reporter = st.text_input("Reporter ID", value="5efb524c3404690bae83acd1")
status = st.text_input("Exclude Status (e.g., CLOSED)", value="CLOSED")

if st.button("Fetch Dealblockers"):
    jql_query = f"project = {project} AND reporter = {reporter} AND status != {status}"
    st.info(f"Using JQL: `{jql_query}`")

    try:
        jira_df = fetch_jira_issues(jql_query)
        st.success(f"✅ Fetched {len(jira_df)} Jira issues")
        st.dataframe(jira_df.head())

        csv = jira_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Jira Results (CSV)", csv, "jira_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error fetching Jira issues: {e}")
        jira_df = None

# --- SECTION 3: CORRELATION ---
if feedback_df is not None and "jira_df" in locals() and jira_df is not None:
    st.header("🔗 Feedback ↔️ Dealblocker Mapping")

    try:
        mapped_df = map_feedback_to_dealblockers(feedback_df, jira_df)
        st.dataframe(mapped_df.head())

        st.download_button("⬇️ Download Correlated Report (CSV)",
                           mapped_df.to_csv(index=False).encode('utf-8'),
                           "feedback_dealblocker_map.csv", "text/csv")

    except Exception as e:
        st.error(f"Error mapping feedback and Jira issues: {e}")

# --- SECTION 4: FEEDBACK CONSOLIDATION ---
if feedback_df is not None:
    st.header("📊 Feedback Consolidation Report (via Gemini AI)")

    if st.button("Generate Consolidation Report"):
        with st.spinner("Analyzing feedback..."):
            try:
                feedback_texts = feedback_df.iloc[:, 0].astype(str).tolist()
                results = analyze_texts_batch(feedback_texts)

                combined_output = "\n".join(results)
                st.text_area("AI Consolidated Output", combined_output, height=400)

                # Convert to downloadable file
                st.download_button(
                    label="⬇️ Download Feedback Consolidation Report",
                    data=combined_output.encode("utf-8"),
                    file_name="feedback_consolidation_report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error generating consolidation report: {e}")
