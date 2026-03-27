"""
Step 5 — Streamlit Inference UI + PDF Report Generation
Provides a web UI for uploading fundus images and generating screening reports.

Usage:
  streamlit run app.py
"""

import base64
import io
import os
import tempfile
import streamlit as st
from datetime import datetime
from fpdf import FPDF, XPos, YPos
from PIL import Image
from predict import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Screening Assistant",
    page_icon="👁",
    layout="wide",
)

CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
GRADE_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

# ── Medical inferences per grade ──────────────────────────────────────────────
MEDICAL_INFERENCES = {
    0: [
        "No signs of diabetic retinopathy detected on this examination.",
        "Annual retinal screening is recommended to monitor for future progression.",
        "Continue current glycaemic management; target HbA1c < 7% (53 mmol/mol).",
        "Maintain blood pressure control (target < 130/80 mmHg).",
    ],
    1: [
        "Microaneurysms present, indicating early retinal vascular damage.",
        "Optimise glycaemic control - reassess HbA1c within 3 months.",
        "Blood pressure and lipid management should be formally reviewed.",
        "Repeat retinal screening in 6-12 months.",
        "No referral required at this stage; monitor closely.",
    ],
    2: [
        "Moderate non-proliferative DR - haemorrhages and/or hard exudates likely.",
        "Referral to ophthalmologist within 3-6 months is recommended.",
        "Risk of progression to vision-threatening DR is elevated.",
        "Intensified glycaemic, blood pressure, and lipid management required.",
        "Patient should be advised to report any sudden visual changes immediately.",
    ],
    3: [
        "Severe non-proliferative DR - high risk of progression to proliferative stage.",
        "Urgent ophthalmology referral within 4 weeks is strongly recommended.",
        "Pan-retinal photocoagulation (PRP) evaluation may be required.",
        "Systemic risk factors (HbA1c, BP, lipids) require urgent optimisation.",
        "Bilateral assessment advised - fellow eye may also be significantly affected.",
    ],
    4: [
        "Proliferative DR detected - this is a vision-threatening stage.",
        "Immediate ophthalmology referral is required.",
        "Vitreoretinal assessment and potential surgical intervention may be needed.",
        "Anti-VEGF therapy (e.g. ranibizumab, bevacizumab) may be indicated.",
        "Risk of vitreous haemorrhage and tractional retinal detachment is significant.",
        "Urgent bilateral comprehensive ophthalmic examination required.",
    ],
}


def _pdf_safe(text: str) -> str:
    """Replace characters outside Latin-1 with ASCII equivalents."""
    return (text
            .replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2022", "-")   # bullet
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .encode("latin-1", errors="replace").decode("latin-1"))


# ── PDF Report ────────────────────────────────────────────────────────────────
def generate_pdf(result: dict, patient_info: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_fill_color(30, 30, 100)
    pdf.rect(0, 0, 210, 20, "F")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 5)
    pdf.cell(0, 10, "DIABETIC RETINOPATHY SCREENING REPORT", new_x="LMARGIN", new_y="NEXT")

    # Patient details
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 25)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")

    fields = [
        ("Patient ID",          patient_info.get("patient_id") or "N/A"),
        ("Patient Name",        patient_info.get("patient_name") or "N/A"),
        ("Age",                 patient_info.get("age") or "N/A"),
        ("Sex",                 patient_info.get("sex") or "N/A"),
        ("Eye Examined",        patient_info.get("eye") or "N/A"),
        ("Referring Physician", patient_info.get("referring_physician") or "N/A"),
    ]
    for label, value in fields:
        pdf.cell(0, 6, _pdf_safe(f"{label}: {value}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Diagnosis summary
    grade    = result["grade"]
    referral = result["referral"]

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 240, 255)
    pdf.cell(0, 8, "DIAGNOSIS SUMMARY", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _pdf_safe(f"DR Grade     : {grade} - {result['grade_name']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, _pdf_safe(f"Confidence   : {result['confidence']:.1%}"), new_x="LMARGIN", new_y="NEXT")

    ref_text = "YES - Refer to ophthalmologist" if referral else "NO - Follow up as per schedule"
    pdf.set_font("Helvetica", "B", 11)
    ref_color = (200, 0, 0) if referral else (0, 150, 0)
    pdf.set_text_color(*ref_color)
    pdf.cell(0, 7, _pdf_safe(f"Referral     : {ref_text}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Medical inferences
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 240, 255)
    pdf.cell(0, 8, "CLINICAL INFERENCES", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 10)
    for inference in MEDICAL_INFERENCES.get(grade, []):
        pdf.cell(6, 6, "-")
        pdf.cell(0, 6, _pdf_safe(inference), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Images — loaded from bytes in memory (no disk files)
    raw_bytes = result.get("raw_bytes")
    hm_bytes  = result.get("heatmap_bytes")

    if raw_bytes:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 255)
        pdf.cell(0, 8, "FUNDUS IMAGE", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.image(Image.open(io.BytesIO(raw_bytes)), x=10, w=90)
        pdf.ln(2)

    if hm_bytes:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 255)
        pdf.cell(0, 8, "GRAD-CAM LESION HEATMAP", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.image(Image.open(io.BytesIO(hm_bytes)), x=10, w=90)
        pdf.ln(2)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6,
             "This report is generated by an AI screening tool and does not replace "
             "clinical diagnosis by a qualified ophthalmologist.",
             new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Diabetic Retinopathy Screening Assistant")
st.caption("EfficientNet-B0/B3 severity grading with Grad-CAM lesion heatmap")

# ── Sidebar — Patient Details ─────────────────────────────────────────────────
# st.form prevents per-keystroke reruns; values are committed to session_state
# only on submit, so they are always available when the PDF is generated.
_SEX_OPTS = ["", "Male", "Female", "Other"]
_EYE_OPTS = ["", "Right Eye (OD)", "Left Eye (OS)", "Both"]

with st.sidebar:
    st.header("Patient Details")
    with st.form("patient_form"):
        _pid  = st.text_input("Patient ID",         value=st.session_state.get("pt_id",   ""))
        _pname= st.text_input("Patient Name",        value=st.session_state.get("pt_name", ""))
        _age  = st.text_input("Age",                 value=st.session_state.get("pt_age",  ""))
        _sex_idx = _SEX_OPTS.index(st.session_state.get("pt_sex", "")) if st.session_state.get("pt_sex","") in _SEX_OPTS else 0
        _eye_idx = _EYE_OPTS.index(st.session_state.get("pt_eye", "")) if st.session_state.get("pt_eye","") in _EYE_OPTS else 0
        _sex  = st.selectbox("Sex",          _SEX_OPTS, index=_sex_idx)
        _eye  = st.selectbox("Eye Examined", _EYE_OPTS, index=_eye_idx)
        _ref  = st.text_input("Referring Physician", value=st.session_state.get("pt_ref",  ""))
        if st.form_submit_button("Save Patient Details", use_container_width=True):
            st.session_state["pt_id"]   = _pid
            st.session_state["pt_name"] = _pname
            st.session_state["pt_age"]  = _age
            st.session_state["pt_sex"]  = _sex
            st.session_state["pt_eye"]  = _eye
            st.session_state["pt_ref"]  = _ref
            st.success("Saved.")

    st.markdown("---")
    st.markdown("**Grading Scale**")
    for i, (name, color) in enumerate(zip(CLASS_NAMES, GRADE_COLORS)):
        st.markdown(f"<span style='color:{color}'>●</span> Grade {i}: {name}", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Referral required for Grade 2 (Moderate) and above.")

# ── Main upload & inference ───────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded:
    raw_bytes = uploaded.getvalue()          # always readable, stream-position-safe
    st.image(raw_bytes, caption="Uploaded image", width=400)

    # Run inference only when a NEW file is uploaded (cache result by file_id)
    if st.session_state.get("file_id") != uploaded.file_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        with st.spinner("Running DR screening pipeline..."):
            try:
                result = predict(tmp_path)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                result = None

        os.unlink(tmp_path)

        if result:
            result["raw_bytes"] = raw_bytes
            st.session_state["file_id"]  = uploaded.file_id
            st.session_state["result"]   = result

    # Use cached result for all subsequent reruns (sidebar edits, button clicks)
    result = st.session_state.get("result")

    if result:
        grade    = result["grade"]
        referral = result["referral"]

        # ── Summary metrics ───────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("DR Grade", f"{grade} — {result['grade_name']}")
        col2.metric("Confidence", f"{result['confidence']:.1%}")
        col3.metric("Referral", "YES" if referral else "NO",
                    delta="Needs ophthalmologist" if referral else "Routine follow-up",
                    delta_color="inverse" if referral else "normal")

        if referral:
            st.error(f"Grade {grade} ({result['grade_name']}) — Referral to ophthalmologist recommended.")
        else:
            st.success(f"Grade {grade} ({result['grade_name']}) — No immediate referral required.")

        # ── Clinical inferences ───────────────────────────────────────────────
        st.subheader("Clinical Inferences")
        for inference in MEDICAL_INFERENCES.get(grade, []):
            st.markdown(f"- {inference}")

        # ── Visualisations (from in-memory bytes) ─────────────────────────────
        img_col1, img_col2 = st.columns(2)
        ann_bytes = result.get("annotated_bytes")
        hm_bytes  = result.get("heatmap_bytes")

        if ann_bytes:
            img_col1.subheader("Annotated Image")
            img_col1.image(ann_bytes)

        if hm_bytes:
            img_col2.subheader("Lesion Heatmap")
            img_col2.image(hm_bytes)

        # ── PDF download ──────────────────────────────────────────────────────
        # Patient variables are plain locals — always current on every rerun.
        # Base64 link embeds the bytes directly so there is no Streamlit
        # download-button buffering between the generated PDF and the browser.
        st.subheader("Download Report")
        patient_info = {
            "patient_id":          st.session_state.get("pt_id")   or "N/A",
            "patient_name":        st.session_state.get("pt_name") or "N/A",
            "age":                 st.session_state.get("pt_age")  or "N/A",
            "sex":                 st.session_state.get("pt_sex")  or "N/A",
            "eye":                 st.session_state.get("pt_eye")  or "N/A",
            "referring_physician": st.session_state.get("pt_ref")  or "N/A",
        }
        pdf_bytes = generate_pdf(result, patient_info)
        b64 = base64.b64encode(pdf_bytes).decode()
        fname = f"dr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64}" download="{fname}">'
            f'<button style="background:#1e1e64;color:white;padding:8px 20px;'
            f'border:none;border-radius:4px;cursor:pointer;font-size:14px;">'
            f'Download PDF Report</button></a>',
            unsafe_allow_html=True,
        )
