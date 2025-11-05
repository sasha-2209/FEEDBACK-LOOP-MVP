import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from datetime import datetime
import plotly.express as px
from sqlalchemy import text # <-- 1. ADD THIS IMPORT

# --- Imports for app logic ---
from classifier import summarize_clusters
from mapper import get_semantic_clusters, map_feedback_to_dealblockers
from jira_connector import fetch_jira_issues

# Load env (so jira_connector can read credentials from .env)
load_dotenv()

st.set_page_config(page_title="Feedback Consolidation & Dealblocker Mapper", layout="wide")
st.title("🧠 Feedback Consolidation & Dealblocker Mapping Tool")

# -----------------------------------------------------------------
# --- DATABASE CONNECTION ---
# -----------------------------------------------------------------
conn = st.connection("my_db", type="sql", url="sqlite:///history.db")

# --- Database Utility Functions (NOW FIXED) ---

@st.cache_resource
def init_db():
    """Create the history tables in the database if they don't exist."""
    with conn.session as s:
        # --- 2. WRAP SQL IN text() ---
        s.execute(text("CREATE TABLE IF NOT EXISTS step_3_history (run_id TEXT, run_timestamp TEXT, data_json TEXT);"))
        s.execute(text("CREATE TABLE IF NOT EXISTS step_4_history (run_id TEXT, run_timestamp TEXT, data_json TEXT);"))
        # -----------------------------
        s.commit()

def save_run_data_db(df, table_name, run_id):
    """Saves a dataframe to the database for a specific run_id."""
    if df is None or df.empty:
        return
    
    data_json = df.to_json(orient='records')
    timestamp = datetime.now().isoformat()
    
    with conn.session as s:
        # --- 3. WRAP SQL IN text() ---
        s.execute(text(f"DELETE FROM {table_name} WHERE run_id = :id"), params={"id": run_id})
        s.execute(
            text(f"INSERT INTO {table_name} (run_id, run_timestamp, data_json) VALUES (:id, :ts, :data)"),
            params={"id": run_id, "ts": timestamp, "data": data_json}
        )
        # -----------------------------
        s.commit()

def load_all_history_db():
    """Loads all run data from the database."""
    with conn.session as s:
        try:
            # --- 4. WRAP SQL IN text() ---
            hist_3 = pd.read_sql(text("SELECT * FROM step_3_history"), s.connection)
        except Exception as e:
            print(f"No Step 3 history or error: {e}")
            hist_3 = pd.DataFrame(columns=['run_id', 'run_timestamp', 'data_json'])
            
        try:
            # --- 5. WRAP SQL IN text() ---
            hist_4 = pd.read_sql(text("SELECT * FROM step_4_history"), s.connection)
        except Exception as e:
            print(f"No Step 4 history or error: {e}")
            hist_4 = pd.DataFrame(columns=['run_id', 'run_timestamp', 'data_json'])
            
    return hist_3, hist_4

def clear_all_history_db():
    """Deletes all data from the history tables."""
    with conn.session as s:
        # --- 6. WRAP SQL IN text() ---
        s.execute(text("DELETE FROM step_3_history;"))
        s.execute(text("DELETE FROM step_4_history;"))
        # -----------------------------
        s.commit()

# --- Initialize the database (creates tables if needed) ---
init_db()

# --- Generate a unique ID for this session's run ---
if "run_id" not in st.session_state:
    st.session_state.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%MS')}"

# -----------------------------------------------------------------
# --- SIDEBAR - RUN HISTORY (Now reads from DB) ---
# -----------------------------------------------------------------
st.sidebar.title("🕰️ Run History")
st.sidebar.info(f"Current Run ID: `{st.session_state.run_id}`")

# ... (This section is unchanged and will work now) ...
hist_df_3, hist_df_4 = load_all_history_db()

all_run_ids = set()
if not hist_df_3.empty:
    all_run_ids.update(hist_df_3['run_id'].unique())
if not hist_df_4.empty:
    all_run_ids.update(hist_df_4['run_id'].unique())

if not all_run_ids:
    st.sidebar.write("No history yet. Run Step 3 or 4 to save results.")
else:
    sorted_run_ids = sorted(list(all_run_ids), reverse=True)
    st.sidebar.write(f"Found {len(sorted_run_ids)} previous run(s):")
    
    for run_id in sorted_run_ids:
        with st.sidebar.expander(f"**{run_id}**"):
            
            st.markdown("--- \n #### Step 3: Consolidation")
            data_3 = hist_df_3[hist_df_3['run_id'] == run_id]
            if not data_3.empty:
                run_3_df = pd.read_json(data_3['data_json'].iloc[0], orient='records')
                st.dataframe(run_3_df)
            else:
                st.write("No Step 3 data for this run.")
                
            st.markdown("--- \n #### Step 4: Mapping")
            data_4 = hist_df_4[hist_df_4['run_id'] == run_id]
            if not data_4.empty:
                run_4_df = pd.read_json(data_4['data_json'].iloc[0], orient='records')
                st.dataframe(run_4_df)
            else:
                st.write("No Step 4 data for this run.")

if st.sidebar.button("Clear All History", type="secondary"):
    clear_all_history_db()
    st.rerun()

# -----------------------------------------------------------------
# --- Step 1: Upload Feedback CSV ---
# -----------------------------------------------------------------
st.header("📂 Step 1: Upload Feedback CSV")
# ... (This section is unchanged) ...
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

# -----------------------------------------------------------------
# --- Step 2: Fetch Jira Issues ---
# -----------------------------------------------------------------
st.header("📡 Step 2: Fetch Jira Issues")
# ... (This section is unchanged) ...
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
        with st.spinner("Fetching Jira issues (will use cache if JQL is unchanged)..."):
            try:
                jira_df = fetch_jira_issues(jql_to_run)
                if jira_df is None or jira_df.empty:
                    st.warning("No Jira issues returned for this JQL. Try adjusting the JQL or check Jira permissions.")
                else:
                    st.success(f"Fetched {len(jira_df)} Jira issues")
                    st.dataframe(jira_df.head())
                    # --- FIX ---
                    csv_data = jira_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Jira Results (CSV)",
                        data=csv_data,
                        file_name="jira_dealblockers.csv",
                        mime='text/csv')
            except Exception as e:
                st.error(f"Error fetching Jira issues: {e}")

# -----------------------------------------------------------------
# 🧩 Step 3: Classify and Cluster Feedback
# -----------------------------------------------------------------
st.header("🧩 Step 3: Classify and Cluster Feedback")
# ... (This section is unchanged) ...
selected_columns = st.multiselect(
    "Select one or more columns containing feedback text for classification:",
    options=feedback_df.columns.tolist(),
    default=[feedback_df.columns[0]] if len(feedback_df.columns) > 0 else [],
)

user_context = st.text_area(
    "Add Context (Optional):",
    placeholder="e.g., 'Focus on mobile performance' or 'We are a gaming company'. This will influence both grouping and labeling."
)

if st.button("Generate Feedback Consolidation Report", type="primary"):
    if selected_columns:
        try:
            feedback_df["combined_text"] = feedback_df[selected_columns] \
                .astype(str).apply(lambda row: " ".join([v for v in row if v and v.lower() != "nan"]), axis=1)

            with st.spinner("Step 1/2: Finding semantic clusters (using cache)..."):
                feedback_groups = get_semantic_clusters(
                    feedback_df, "combined_text", grouping_context=user_context
                )
                if not feedback_groups:
                    st.error("Clustering failed to produce any groups.")
                    st.stop()

            with st.spinner(f"Step 2/2: Using Gemini to summarize {len(feedback_groups)} clusters (using cache)..."):
                clustered_df = summarize_clusters(
                    feedback_groups, labeling_context=user_context
                )
            
            st.success("✅ Feedback Consolidation Complete")

            if not clustered_df.empty:
                st.subheader("🧠 Feedback Clusters Summary (Current Run)")
                clustered_df.to_csv("feedback_consolidation.csv", index=False)
                
                clustered_df_display = clustered_df.copy()
                clustered_df_display["feedback_text"] = clustered_df_display["feedback_text"].apply(
                    lambda x: str(x)[:250] + "..." if len(str(x)) > 250 else str(x)
                )
                st.dataframe(clustered_df_display, use_container_width=True)
                csv_data_3 = clustered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Consolidated Feedback (CSV)",
                    data=csv_data_3,
                    file_name="feedback_consolidation.csv",
                    mime='text/csv'
                )
                
                # --- SAVE TO DB ---
                save_run_data_db(clustered_df, "step_3_history", st.session_state.run_id)
                st.toast(f"Saved results to history! Sidebar will update on next refresh.")

            else:
                st.warning("No clusters generated. Check if selected columns contain meaningful text.")
        except Exception as e:
            st.error(f"Error during classification: {e}")
    else:
        st.warning("Please select at least one column to classify.")

# -----------------------------------------------------------------
# --- 🗺️ Visualize Clusters ---
# -----------------------------------------------------------------
st.subheader("🗺️ Visualize Clusters")
# ... (This section is unchanged) ...
if st.button("Generate Mindmap / Treemap"):
    try:
        clustered_df = pd.read_csv("feedback_consolidation.csv")
        
        if 'category' not in clustered_df.columns or 'cluster_label' not in clustered_df.columns or 'feedback_text' not in clustered_df.columns:
            st.error("Could not find 'category', 'cluster_label', or 'feedback_text' in the saved data.")
        else:
            treemap_data = []
            for _, row in clustered_df.iterrows():
                try:
                    feedback_texts = str(row['feedback_text']).split(' | ')
                except Exception:
                    feedback_texts = ["Error parsing feedback"]
                
                for text in feedback_texts:
                    if text.strip():
                        treemap_data.append({
                            'category': row['category'],
                            'cluster_label': row['cluster_label'],
                            'reasoning': row['reasoning'],
                            'individual_feedback': text.strip(),
                            'size': 1
                        })

            if not treemap_data:
                st.warning("No individual feedback items to display.")
                st.stop()
                
            treemap_df = pd.DataFrame(treemap_data)
            
            st.info("Generating interactive treemap... You can click to zoom. Boxes are sized equally.")
            fig = px.treemap(
                treemap_df,
                path=[px.Constant("All Feedback"), 'category', 'cluster_label', 'individual_feedback'],
                values='size',
                color='category',
                hover_data={
                    'reasoning': True,
                    'cluster_label': True,
                    'individual_feedback': True,
                    'size': False
                }
            )
            fig.update_traces(hovertemplate='<b>Cluster:</b> %{customdata[1]}<br><b>Feedback:</b> %{customdata[2]}<br><b>Reasoning:</b> %{customdata[0]}<extra></extra>')
            fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Please run Step 3 'Generate Feedback Consolidation Report' first to create the data.")
    except Exception as e:
        st.error(f"An error occurred while generating the treemap: {e}")

# -----------------------------------------------------------------
# --- Step 4: Map Feedback → Jira Dealblockers (clustered) ---
# -----------------------------------------------------------------
st.header("🔗 Step 4: Map Consolidated Feedback to Dealblockers")

st.markdown("Set the minimum similarity score for a 'Semantic Match'.")
st.info("[Recommended match_score = 0.7]")

match_threshold = st.slider(
    "Match Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,  # <-- Default value
    step=0.05
)
st.info("This step reads the saved files from Step 2 and 3 and maps them using both explicit keys and semantic search.")

if st.button("Run Mapping with Dealblockers"):
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

    with st.spinner("Mapping consolidated feedback clusters to Jira dealblockers (using cache)..."):
        try:
            mapped_df = map_feedback_to_dealblockers(
                feedback_consolidation, 
                jira_dealblockers,
                similarity_threshold=match_threshold  # <-- Pass the slider value
            )
            if mapped_df is None or mapped_df.empty:
                st.warning("No mappings were found.")
            else:
                st.success("✅ Mapping complete")
                st.subheader("🗺️ Mapped Results (Current Run)")

                mapped_df_display = mapped_df.copy()
                
               # if "original_feedback_texts" in mapped_df_display.columns:
                 #   mapped_df_display["original_feedback_texts"] = mapped_df_display["original_feedback_texts"].apply(
                 #       lambda x: str(x)[:250] + "..." if len(str(x)) > 250 else str(x)
                 #   )
                
                st.dataframe(mapped_df_display.head(100), use_container_width=True)

                csv_data_4 = mapped_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Mapped Feedback → Dealblockers CSV",
                    data=csv_data_4,
                    file_name="mapped_feedback_dealblockers.csv",
                    mime='text/csv')
                
                # --- SAVE TO DB ---
                save_run_data_db(mapped_df, "step_4_history", st.session_state.run_id)
                st.toast(f"Saved mapping results to history! Sidebar will update on next refresh.")

        except Exception as e:
            st.error(f"Error mapping feedback and Jira issues: {e}")