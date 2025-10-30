import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

# --- CORRECTED IMPORTS ---
# We import the cluster-finder from mapper and the group-summarizer from classifier
from classifier import summarize_clusters
# ✅ Import both functions from mapper
from mapper import get_semantic_clusters, map_feedback_to_dealblockers 
# ---
from jira_connector import fetch_jira_issues

# Load env (so jira_connector can read credentials from .env)
load_dotenv()

st.set_page_config(page_title="Feedback Consolidation & Dealblocker Mapper", layout="wide")
st.title("🧠 Feedback Consolidation & Dealblocker Mapping Tool")

# --- Utility: Download button for DataFrames ---
def download_button(df, label, filename):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label=label,
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
    )

# --- Step 1: Upload Feedback CSV ---
st.subheader("📂 Step 1: Upload Feedback CSV")
uploaded_file = st.file_uploader("Upload your feedback CSV file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            feedback_df = pd.read_csv(uploaded_file)
        else:
            feedback_df = pd.read_excel(uploaded_file)
        st.success("✅ Feedback file uploaded")
        st.dataframe(feedback_df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("Please upload a feedback CSV/XLSX to begin.")
    st.stop()

# --- Step 2: Fetch Jira Issues via JQL (credentials read from .env inside jira_connector) ---
st.subheader("📡 Step 2: Fetch Jira Issues")

col1, col2, col3 = st.columns(3)
project = col1.text_input("Project Key (optional)", value="SDK")
reporter = col2.text_input("Reporter ID (optional)", value="")
status_exclude = col3.text_input("Exclude Status (optional)", value="CLOSED")

if "jql_input" not in st.session_state:
    st.session_state["jql_input"] = ""

if st.button("Build JQL from inputs"):
    jql_parts = []
    if project:
        jql_parts.append(f"project = {project}")
    if reporter:
        jql_parts.append(f"reporter = {reporter}")
    if status_exclude:
        jql_parts.append(f"status != {status_exclude}")
    built_jql = " AND ".join(jql_parts) if jql_parts else ""
    st.session_state["jql_input"] = built_jql
    st.success("✅ Built JQL")
    st.code(built_jql, language="jql")

jql_input = st.text_area(
    "Or enter full JQL to fetch issues (overrides built JQL if present):",
    value=st.session_state["jql_input"],
    key="jql_input_box",
)

if st.button("Fetch Jira Issues"):
    jql_to_run = jql_input.strip() or st.session_state.get("jql_input", "")
    if not jql_to_run:
        st.error("Please enter a JQL query to fetch issues.")
    else:
        with st.spinner("Fetching Jira issues (credentials from .env)..."):
            try:
                jira_df = fetch_jira_issues(jql_to_run)
                if jira_df is None or jira_df.empty:
                    st.warning("No Jira issues returned for this JQL. Try adjusting the JQL or check Jira permissions.")
                else:
                    st.success(f"✅ Fetched {len(jira_df)} Jira issues")
                    st.dataframe(jira_df.head())
                    # Save the file for Step 4
                    jira_df.to_csv("jira_dealblockers.csv", index=False)
                    download_button(jira_df, "⬇️ Download Jira Results (CSV)", "jira_dealblockers.csv")
            except Exception as e:
                st.error(f"Error fetching Jira issues: {e}")


# 🧩 Step 3: Classify and Cluster Feedback
st.header("🧩 Step 3: Classify and Cluster Feedback")

selected_columns = st.multiselect(
    "Select one or more columns containing feedback text for classification:",
    options=feedback_df.columns.tolist(),
    default=[feedback_df.columns[0]] if len(feedback_df.columns) > 0 else [],
)

if st.button("Generate Feedback Consolidation Report", type="primary"):
    if selected_columns:
        try:
            # --- THIS IS THE NEW HYBRID FLOW ---

            # Combine selected columns into a single text field
            feedback_df["combined_text"] = feedback_df[selected_columns] \
                .astype(str).apply(lambda row: " ".join([v for v in row if v and v.lower() != "nan"]), axis=1)

            # 1. Use mapper.py to get semantic groups
            with st.spinner("Step 1/2: Finding semantic clusters using ML..."):
                # This returns a dictionary of groups: {1: [text_a, text_b], 2: [text_c]}
                feedback_groups = get_semantic_clusters(feedback_df, "combined_text")
                if not feedback_groups:
                    st.error("Clustering failed to produce any groups.")
                    st.stop()

            # 2. Use classifier.py to summarize those groups with Gemini
            with st.spinner(f"Step 2/2: Using Gemini to summarize {len(feedback_groups)} clusters..."):
                clustered_df = summarize_clusters(feedback_groups)
            
            # --- END OF NEW FLOW ---

            st.success("✅ Feedback Consolidation Complete")

            if not clustered_df.empty:
                st.subheader("🧠 Feedback Clusters Summary")

                # Save the file for Step 4
                clustered_df.to_csv("feedback_consolidation.csv", index=False)

                # Make table look readable in Streamlit
                clustered_df_display = clustered_df.copy()
                clustered_df_display["feedback_text"] = clustered_df_display["feedback_text"].apply(
                    lambda x: x[:250] + "..." if len(x) > 250 else x
                )

                st.dataframe(clustered_df_display, use_container_width=True)

                # Download as CSV
                download_button(clustered_df, "⬇️ Download Consolidated Feedback (CSV)", "feedback_consolidation.csv")

            else:
                st.warning("No clusters generated. Check if selected columns contain meaningful text.")

        except Exception as e:
            st.error(f"Error during classification: {e}")
    else:
        st.warning("Please select at least one column to classify.")


# --- Step 4: Map Feedback → Jira Dealblockers (clustered) ---
st.subheader("🔗 Step 4: Map Consolidated Feedback to Dealblockers")

st.info("This step reads the saved files from Step 2 and 3 and maps them using both explicit keys and semantic search.")

# ---
# ✅ NOTE: This block is now UNCOMMENTED and will run.
# ---
if st.button("Run Mapping with Dealblockers"):
    # ensure files exist
    try:
        feedback_consolidation = pd.read_csv("feedback_consolidation.csv")
    except FileNotFoundError:
        st.error("Feedback consolidation report not found. Run Step 3 first.")
        st.stop()

    try:
        jira_dealblockers = pd.read_csv("jira_dealblockers.csv")
    except FileNotFoundError:
        st.error("Jira dealblockers CSV not found. Run Step 2 (Fetch Jira Issues) first.")
        st.stop()

    with st.spinner("Mapping consolidated feedback clusters to Jira dealblockers..."):
        try:
            # map_feedback_to_dealblockers in mapper.py expects the consolidated clusters and jira df
            # ✅ This function now exists!
            mapped_df = map_feedback_to_dealblockers(feedback_consolidation, jira_dealblockers) 
            if mapped_df is None or mapped_df.empty:
                st.warning("No mappings were found.")
            else:
                st.success("✅ Mapping complete")
                st.dataframe(mapped_df.head(100), use_container_width=True)
                mapped_df.to_csv("mapped_feedback_dealblockers.csv", index=False)
                download_button(mapped_df, "⬇️ Download Mapped Feedback → Dealblockers CSV", "mapped_feedback_dealblockers.csv")
        except Exception as e:
            st.error(f"Error mapping feedback and Jira issues: {e}")