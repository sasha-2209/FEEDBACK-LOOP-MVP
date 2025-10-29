import streamlit as st
import pandas as pd
import json
import re
from classifier import analyze_texts_batch

st.set_page_config(page_title="Feedback Intelligence Analyzer", layout="wide")
st.title("💡 Product Feedback Intelligence System")

uploaded_file = st.file_uploader("📤 Upload CSV or Excel with feedback data", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Loaded {len(df)} rows.")
        st.write("Columns detected:", df.columns.tolist())

        feedback_col = st.selectbox("📝 Select feedback column:", df.columns)
        issue_col = st.selectbox("🧾 Optional: Select issue key / ticket column (if available):",
                                 ["None"] + list(df.columns))

        feedback_list = df[feedback_col].dropna().astype(str).tolist()
        issue_keys = None if issue_col == "None" else df[issue_col].astype(str).tolist()

        if st.button("🔍 Analyze Feedback"):
            with st.spinner("Analyzing feedback..."):
                results = analyze_texts_batch(feedback_list, issue_keys)

            parsed_rows = []

            def parse_json(text):
                try:
                    clean = re.sub(r"^```json\s*|\s*```$", "", text.strip())
                    return json.loads(clean)
                except Exception:
                    return []

            for r in results:
                if isinstance(r, str):
                    parsed_rows.extend(parse_json(r))
                elif isinstance(r, list):
                    parsed_rows.extend(r)

            if parsed_rows:
                parsed_df = pd.DataFrame(parsed_rows)
                st.dataframe(parsed_df, use_container_width=True)

                csv = parsed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Structured Feedback CSV",
                    data=csv,
                    file_name="structured_feedback.csv",
                    mime="text/csv"
                )
            else:
                st.error("No valid feedback parsed. Check API key or file content.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV or Excel file to start.")
