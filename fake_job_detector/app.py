"""
app.py - Streamlit interface for the NLP-Based Fake Job Post Detector.
Run with:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src.predict import MODELS_DIR, JobPostDetector          # noqa: E402
from src.preprocessing import explain_steps                   # noqa: E402
from src.sample_jobs import SAMPLE_JOBS                       # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

st.set_page_config(page_title="Fake Job Post Detector", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; color: white;
       background: linear-gradient(120deg, #1f3c88, #2e86de);}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: .3rem 0 0 0; opacity: .9;}
.small-note {font-size: .85rem; color: #666;}
</style>
""", unsafe_allow_html=True)

EMPLOYMENT = ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
EDUCATION = ["", "High School or equivalent", "Vocational", "Associate Degree", "Bachelor's Degree",
             "Master's Degree", "Doctorate", "Certification", "Professional", "Unspecified"]
EXPERIENCE = ["", "Internship", "Entry level", "Associate", "Mid-Senior level", "Director",
              "Executive", "Not Applicable"]

DEFAULTS = {
    "title": "", "company_profile": "", "description": "", "requirements": "", "benefits": "",
    "location": "", "salary_range": "", "employment_type": "", "education": "", "experience": "",
    "industry": "", "function": "", "full_text": "",
    "telecommuting": False, "has_logo": True, "has_questions": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_sample(index):
    """Button callback: copy a demo ad into the form."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = SAMPLE_JOBS[index].get(key, value)


def clear_form():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def fmt_pct(p):
    """Never show a flat 100% or 0% - a model is never that certain."""
    if p >= 0.99:
        return ">99%"
    if p <= 0.01:
        return "<1%"
    return f"{p * 100:.1f}%"


@st.cache_resource
def get_detector():
    return JobPostDetector()


# ------------------------------------------------------------------ header
st.markdown('<div class="hero"><h1>🕵️ NLP-Based Fake Job Post Detector</h1>'
            '<p>Paste a job advertisement and let Natural Language Processing + Machine Learning '
            'screen it for warning signs.</p></div>', unsafe_allow_html=True)
st.write("")

try:
    detector = get_detector()
except FileNotFoundError as err:
    st.error("The model has not been trained yet.")
    st.code("python -m src.train_model", language="bash")
    st.caption(str(err))
    st.stop()

tab_detect, tab_perf, tab_about = st.tabs(["🔍 Detect", "📊 Model performance", "ℹ️ About & limits"])

# ------------------------------------------------------------------ detect tab
with tab_detect:
    with st.expander("Try a demo advertisement (hand-written test examples, not real job posts)"):
        cols = st.columns(len(SAMPLE_JOBS))
        for i, col in enumerate(cols):
            col.button(f"Demo {i + 1}", key=f"demo{i}", on_click=load_sample, args=(i,),
                       width="stretch", help=SAMPLE_JOBS[i]["name"])
        st.button("Clear form", on_click=clear_form)

    left, right = st.columns(2)
    with left:
        st.text_input("Job Title", key="title")
        st.text_area("Company Profile", key="company_profile", height=110)
        st.text_area("Job Description", key="description", height=170)
        st.text_area("Requirements", key="requirements", height=110)
        st.text_area("Benefits", key="benefits", height=80)
    with right:
        st.text_input("Location", key="location", placeholder="e.g. IN, TN, Chennai")
        st.text_input("Salary Range", key="salary_range", placeholder="e.g. 400000-600000")
        st.selectbox("Employment Type", EMPLOYMENT, key="employment_type",
                     format_func=lambda x: x or "(not specified)")
        st.selectbox("Education", EDUCATION, key="education",
                     format_func=lambda x: x or "(not specified)")
        st.selectbox("Experience", EXPERIENCE, key="experience",
                     format_func=lambda x: x or "(not specified)")
        st.text_input("Industry", key="industry")
        st.text_input("Function", key="function")
        st.markdown("**Details of the original posting**")
        st.checkbox("Remote / work-from-home", key="telecommuting")
        st.checkbox("Company logo was shown", key="has_logo")
        st.checkbox("Screening questions were asked", key="has_questions")
        st.markdown('<p class="small-note">The model learned from the fields above, so leaving '
                    'fields empty is treated as "information missing".</p>', unsafe_allow_html=True)

    st.text_area("…or paste the complete job advertisement here", key="full_text", height=160)

    if st.button("🔎 Detect Fake Job", type="primary", width="stretch"):
        fields = {k: st.session_state[k] for k in DEFAULTS}
        try:
            result = detector.predict(fields)
        except ValueError as err:                       # incorrect / too little input
            st.warning(str(err))
        except Exception as err:                        # anything unexpected
            st.error(f"Something went wrong while analysing the text: {err}")
        else:
            st.subheader("Prediction")
            if result["is_fake"]:
                st.error("## 🔴 POTENTIALLY FAKE JOB POST")
            else:
                st.success("## 🟢 REAL JOB POST")
            m1, m2 = st.columns(2)
            m1.metric("Prediction Confidence", fmt_pct(result["confidence"]))
            m2.metric("Fake-likelihood score", fmt_pct(result["fake_probability"]))
            st.progress(min(max(result["fake_probability"], 0.0), 1.0))
            st.caption("The percentage is the model's estimated probability based on patterns in "
                       "historical data. It is not a guarantee.")

            st.subheader("Why was this prediction made?")
            st.caption("These are indicators, not proof that a job is fraudulent (or genuine).")
            if result["indicators"]:
                for category, detail in result["indicators"]:
                    st.warning(f"**{category}** — {detail}")
            else:
                st.info("No rule-based warning signs were found.")
            if result["positive_signals"]:
                st.markdown("**Reassuring signs**")
                for category, detail in result["positive_signals"]:
                    st.markdown(f"- ✅ **{category}** — {detail}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Words linked to FAKE posts in the training data**")
                if result["fake_terms"]:
                    st.dataframe(pd.DataFrame(result["fake_terms"], columns=["word (stemmed)", "weight"]),
                                 hide_index=True, width="stretch")
                else:
                    st.write("None of the strongest fake-linked words appeared.")
            with c2:
                st.markdown("**Words linked to REAL posts in the training data**")
                if result["real_terms"]:
                    st.dataframe(pd.DataFrame(result["real_terms"], columns=["word (stemmed)", "weight"]),
                                 hide_index=True, width="stretch")
                else:
                    st.write("None of the strongest real-linked words appeared.")

            with st.expander("See how your text was preprocessed (NLP steps)"):
                text = " ".join([st.session_state["title"], st.session_state["description"],
                                 st.session_state["full_text"]])[:1500]
                for stage, value in explain_steps(text).items():
                    st.markdown(f"**{stage}**")
                    st.write(value if isinstance(value, str) else ", ".join(value[:60]) + (" …" if len(value) > 60 else ""))

            st.info("⚠️ This tool is a screening aid, not a verdict. Always verify a company through its "
                    "official website, and never pay money or share bank details to get a job.")

# ------------------------------------------------------------------ performance tab
with tab_perf:
    info_path = MODELS_DIR / "model_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        st.markdown(f"**Selected model:** {info['selected_model']}  ·  "
                    f"train posts: {info['train_size']}  ·  test posts: {info['test_size']}")
    table = REPORTS_DIR / "model_comparison.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table), hide_index=True, width="stretch")
    for name, caption in [("model_comparison.png", "Model comparison"),
                          ("confusion_matrices.png", "Confusion matrices"),
                          ("class_distribution.png", "Class distribution")]:
        image = REPORTS_DIR / name
        if image.exists():
            st.image(str(image), caption=caption)
    if not table.exists():
        st.info("Train the model to generate the evaluation tables and charts.")

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown("""
**How it works:** the text is cleaned (lowercase, punctuation removed, stop words removed, stemming),
converted to numbers with TF-IDF, combined with a few structured features (missing company profile,
short description, number of links, …) and passed to a machine-learning classifier.

**Important limits**
- Predictions come from patterns in *historical* data (English-language postings from an older period).
  New scam styles may not be recognised.
- A genuine post can look suspicious and a well-written scam can look genuine.
- Use the result to *decide what to check*, not as proof. Verify employers independently.
""")
