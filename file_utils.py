import pandas as pd

def extract_feedback_from_file(uploaded_file):
    """
    Reads feedback data from CSV or Excel file and extracts text content.
    """
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    # Look for columns containing text or feedback keywords
    for col in df.columns:
        if "feedback" in col.lower() or "comment" in col.lower() or "text" in col.lower():
            return df[col].dropna().tolist()
    return df.iloc[:, 0].dropna().tolist()
