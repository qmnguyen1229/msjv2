import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import tempfile

# Load model and encoders
model = joblib.load('msj_model_nb.pkl')
le_case = joblib.load('le_case.pkl')
le_plaintiff = joblib.load('le_plaintiff.pkl')
le_defendant = joblib.load('le_defendant.pkl')
le_filing = joblib.load('le_filing.pkl')
le_outcome = joblib.load('le_outcome.pkl')

color_map = {
    'Denied': '#FF4B4B',
    'Granted': '#4CAF50',
    'Partially Granted / Denied': '#FFDD57',
    'Partially Granted, Partially Denied': '#FFDD57',
}
def get_color(label): return color_map.get(label, '#CCCCCC')

tab1, tab2, tab3 = st.tabs(["📊 Predictor", "🔁 What-If Comparator", "ℹ️ About"])

with tab1:
    st.title("Motion for Summary Judgment Outcome Predictor")
    st.markdown("Use this tool to predict the likely outcome of a Motion for Summary Judgment based on your case type, the party types and filing party.")

    case_type = st.selectbox("Case Type", le_case.classes_)
    plaintiff_type = st.selectbox("Plaintiff Type", le_plaintiff.classes_)
    defendant_type = st.selectbox("Defendant Type", le_defendant.classes_)
    filing_party = st.selectbox("Filing Party", le_filing.classes_)

    if st.button("Predict Outcome"):
        input_data = np.array([[
            le_case.transform([case_type])[0],
            le_plaintiff.transform([plaintiff_type])[0],
            le_defendant.transform([defendant_type])[0],
            le_filing.transform([filing_party])[0]
        ]])
        probabilities = np.clip(model.predict_proba(input_data)[0], 0.01, 0.99)

        chart_data = pd.DataFrame({
            'Outcome': le_outcome.classes_,
            'Probability': probabilities,
            'Color': [get_color(o) for o in le_outcome.classes_]
        })
        st.bar_chart(chart_data.set_index("Outcome")["Probability"])

        sorted_probs = sorted(zip(le_outcome.classes_, probabilities), key=lambda x: x[1], reverse=True)
        for i, (label, prob) in enumerate(sorted_probs):
            pct = f"{prob:.2%}"
            if i == 0:
                st.success(f"Most Likely: {label} ({pct})")
            elif i == len(sorted_probs) - 1:
                st.error(f"Least Likely: {label} ({pct})")
            else:
                st.warning(f"Other Possibility: {label} ({pct})")

with tab2:
    st.title("🔁 What-If Comparator")
    st.markdown("Use this tool to see how changing one or more variables—like the party types or the filing party—might affect the predicted outcome of a Motion for Summary Judgment.")

    case_name = st.text_input("Case Name (used to name PDF)", value="My Case")
    case_type = st.selectbox("Select Case Type", le_case.classes_, key="case_type_comparator")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Inputs")
        plaintiff_type = st.selectbox("Plaintiff Type", le_plaintiff.classes_, key="plaintiff_orig")
        defendant_type = st.selectbox("Defendant Type", le_defendant.classes_, key="defendant_orig")
        filing_party = st.selectbox("Filing Party", le_filing.classes_, key="filer_orig")
    with col2:
        st.markdown("#### What-If Inputs")
        whatif_plaintiff_type = st.selectbox("Plaintiff Type", le_plaintiff.classes_, key="plaintiff_whatif")
        whatif_defendant_type = st.selectbox("Defendant Type", le_defendant.classes_, key="defendant_whatif")
        whatif_filing_party = st.selectbox("Filing Party", le_filing.classes_, key="filer_whatif")

    if st.button("Run What-If Comparison"):
        enc = lambda l, v: l.transform([v])[0]
        case_enc = enc(le_case, case_type)

        input_original = np.array([[case_enc, enc(le_plaintiff, plaintiff_type),
                                    enc(le_defendant, defendant_type), enc(le_filing, filing_party)]])
        input_whatif = np.array([[case_enc, enc(le_plaintiff, whatif_plaintiff_type),
                                  enc(le_defendant, whatif_defendant_type), enc(le_filing, whatif_filing_party)]])

        prob_orig = np.clip(model.predict_proba(input_original)[0], 0.01, 0.99)
        prob_whatif = np.clip(model.predict_proba(input_whatif)[0], 0.01, 0.99)

        outcome_labels = le_outcome.classes_
        df_comparison = pd.DataFrame({
            "Outcome": outcome_labels,
            "Original %": [f"{p:.2%}" for p in prob_orig],
            "What-If %": [f"{p:.2%}" for p in prob_whatif],
            "Change": [f"{w - o:+.2%}" for o, w in zip(prob_orig, prob_whatif)]
        })
        st.subheader("📋 Outcome Comparison Table")
        st.dataframe(df_comparison, use_container_width=True)

        # Construct what-if summary sentence
        diffs = []
        if plaintiff_type != whatif_plaintiff_type:
            diffs.append(f"plaintiff from {plaintiff_type} to {whatif_plaintiff_type}")
        if defendant_type != whatif_defendant_type:
            diffs.append(f"defendant from {defendant_type} to {whatif_defendant_type}")
        if filing_party != whatif_filing_party:
            diffs.append(f"filing party from {filing_party} to {whatif_filing_party}")
        if diffs:
            change_sentence = "This prediction compares a change in: " + ", ".join(diffs) + "."
        else:
            change_sentence = "No changes were made between the original and what-if inputs."

        # Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(outcome_labels))
        width = 0.35
        ax.bar(x - width/2, prob_orig, width, label='Original', color='#1f77b4')
        ax.bar(x + width/2, prob_whatif, width, label='What-If', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(outcome_labels, rotation=20)
        ax.set_ylabel("Probability")
        ax.set_title("MSJ Prediction Comparison")
        ax.legend()
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Motion for Summary Judgment Prediction - {case_name}", align='L')
            pdf.ln(2)
            pdf.set_font("Arial", style="", size=11)
            pdf.multi_cell(0, 8, change_sentence, align='L')

            # Save and insert chart
            chart_path = tmpfile.name.replace(".pdf", ".png")
            plt.savefig(chart_path)
            pdf.image(chart_path, x=10, y=None, w=pdf.w - 20)

            pdf.ln(8)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(80, 10, "Outcome", 1)
            pdf.cell(40, 10, "Original %", 1)
            pdf.cell(40, 10, "What-If %", 1)
            pdf.cell(30, 10, "Change", 1)
            pdf.ln()

            pdf.set_font("Arial", "", 11)
            for row in df_comparison.itertuples(index=False):
                pdf.cell(80, 8, row[0], 1)
                pdf.cell(40, 8, row[1], 1)
                pdf.cell(40, 8, row[2], 1)
                pdf.cell(30, 8, row[3], 1)
                pdf.ln()

            pdf.output(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                st.download_button("📥 Download What-If PDF", data=f.read(), file_name=f"{case_name} - MSJ Prediction.pdf")

with tab3:
    st.title("ℹ️ About")
    st.markdown("""
This program was created by Liam Bigbee, Lauren Bretz, and Quan Nguyen as a final project for Professor Andrew Torrance's Legal Analytics class at the University of Kansas School of Law.  
The program is just a school project and **should not** be used to predict the outcome of real-world motions.
""")
