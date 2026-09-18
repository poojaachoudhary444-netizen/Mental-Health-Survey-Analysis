"""
dashboard.py — Mental Health in Tech Survey: Streamlit Dashboard
====================================================================
Interactive dashboard exploring the OSMI 2014 "Mental Health in Tech"
survey: demographics, geography, workplace policies, stigma/attitudes,
and predictors of treatment-seeking behavior.

Run with:
    streamlit run dashboard.py

Reads from db/mental_health.db (built by app.py). Self-heals: if the DB
is missing (e.g. a fresh deploy where only .py files were committed but
data/survey.csv is present), it builds the DB automatically on first load.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Mental Health in Tech Survey",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "mental_health.db"
TABLE_NAME = "survey"

PRIMARY = "#5B4B8A"
ACCENT = "#2CA58D"
YES_COLOR = "#2CA58D"
NO_COLOR = "#D64550"
TREATMENT_COLORS = {"Yes": YES_COLOR, "No": NO_COLOR}


# --------------------------------------------------------------------------
# Data loading (self-healing)
# --------------------------------------------------------------------------
def _ensure_data_exists():
    import app as etl

    if DB_PATH.exists():
        return
    with st.spinner("First run: preparing data (this happens once)..."):
        etl.run_pipeline()


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    _ensure_data_exists()
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    for c in ["Is_US", "Treatment_Flag", "Family_History_Flag", "Has_Comment"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    return df


df_raw = load_data()

if df_raw.empty:
    st.title("🧠 Mental Health in Tech Survey")
    st.error(
        "Could not load or build the dataset. Make sure `data/survey.csv` "
        "(the OSMI 2014 survey export) is present, then rerun."
    )
    st.stop()

AGE_ORDER = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
EMP_ORDER = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("🧠 Filters")

country_opts = df_raw["Country"].value_counts().index.tolist()
default_countries = df_raw["Country"].value_counts().head(6).index.tolist()
sel_countries = st.sidebar.multiselect("Country", country_opts, default=default_countries)

gender_opts = sorted(df_raw["Gender_Clean"].unique().tolist())
sel_genders = st.sidebar.multiselect("Gender", gender_opts, default=gender_opts)

age_opts = [a for a in AGE_ORDER if a in df_raw["Age_Group"].unique()]
sel_ages = st.sidebar.multiselect("Age Group", age_opts, default=age_opts)

emp_opts = [e for e in EMP_ORDER if e in df_raw["no_employees"].unique()]
sel_emp = st.sidebar.multiselect("Company Size", emp_opts, default=emp_opts)

tech_only = st.sidebar.checkbox("Tech companies only", value=False)
treated_only = st.sidebar.checkbox("Sought treatment only", value=False)
family_hist_only = st.sidebar.checkbox("Family history of mental illness only", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: OSMI 2014 Mental Health in Tech Survey (1,251 cleaned responses). "
    "Filters apply across every tab."
)

df = df_raw[
    df_raw["Country"].isin(sel_countries)
    & df_raw["Gender_Clean"].isin(sel_genders)
    & df_raw["Age_Group"].isin(sel_ages)
    & df_raw["no_employees"].isin(sel_emp)
]
if tech_only:
    df = df[df["tech_company"] == "Yes"]
if treated_only:
    df = df[df["Treatment_Flag"]]
if family_hist_only:
    df = df[df["Family_History_Flag"]]

if df.empty:
    st.warning("No responses match the current filters. Try widening your selection.")
    st.stop()


# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("🧠 Mental Health in Tech Survey")
st.caption("OSMI 2014 · Attitudes, workplace policy, and treatment-seeking behavior in the tech industry")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Respondents", f"{len(df):,}")
treat_rate = df["Treatment_Flag"].mean() * 100
k2.metric("Sought Treatment", f"{treat_rate:.1f}%")
fh_rate = df["Family_History_Flag"].mean() * 100
k3.metric("Family History", f"{fh_rate:.1f}%")
k4.metric("Countries Represented", f"{df['Country'].nunique()}")
avg_support = df["Workplace_Support_Score"].mean()
k5.metric("Avg. Workplace Support Score", f"{avg_support:.1f} / 5")

st.markdown("---")

tabs = st.tabs([
    "📊 Overview",
    "🌍 Geography",
    "👤 Demographics",
    "🏢 Workplace Factors",
    "💬 Attitudes & Stigma",
    "👪 Family History & Interference",
    "🔎 Predictors of Treatment",
    "📋 Explore Raw Data",
])

# ==========================================================================
# TAB 1 — OVERVIEW
# ==========================================================================
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        t_counts = df["treatment"].value_counts().reset_index()
        t_counts.columns = ["treatment", "Respondents"]
        fig = px.pie(t_counts, names="treatment", values="Respondents", hole=0.5,
                     color="treatment", color_discrete_map=TREATMENT_COLORS,
                     title="Have You Sought Treatment for a Mental Health Condition?")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gender_counts = df["Gender_Clean"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Respondents"]
        fig = px.bar(gender_counts, x="Gender", y="Respondents", color="Gender",
                     title="Respondents by Gender")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        age_counts = df["Age_Group"].value_counts().reindex(AGE_ORDER).dropna().reset_index()
        age_counts.columns = ["Age_Group", "Respondents"]
        fig = px.bar(age_counts, x="Age_Group", y="Respondents", color="Respondents",
                     color_continuous_scale="Purples", title="Respondents by Age Group")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        emp_counts = df["no_employees"].value_counts().reindex(EMP_ORDER).dropna().reset_index()
        emp_counts.columns = ["Company_Size", "Respondents"]
        fig = px.bar(emp_counts, x="Company_Size", y="Respondents", color="Respondents",
                     color_continuous_scale="Teal", title="Respondents by Company Size")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 2 — GEOGRAPHY
# ==========================================================================
with tabs[1]:
    st.subheader("Treatment-Seeking Rate by Country (min. 15 respondents)")
    country_stats = (
        df.groupby("Country")
        .agg(Respondents=("treatment", "size"), Treatment_Rate=("Treatment_Flag", "mean"))
        .query("Respondents >= 15")
        .sort_values("Treatment_Rate", ascending=False)
    )
    country_stats["Treatment_Rate"] = (country_stats["Treatment_Rate"] * 100).round(1)
    fig = px.bar(
        country_stats.reset_index().sort_values("Treatment_Rate"),
        x="Treatment_Rate", y="Country", orientation="h", color="Treatment_Rate",
        color_continuous_scale="Purples", title="% Who Sought Treatment, by Country",
        labels={"Treatment_Rate": "Treatment Rate (%)"},
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        top_countries = df["Country"].value_counts().head(10).reset_index()
        top_countries.columns = ["Country", "Respondents"]
        fig = px.bar(top_countries.sort_values("Respondents"), x="Respondents", y="Country",
                     orientation="h", title="Top 10 Countries by Respondent Count",
                     color="Respondents", color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        us_df = df[df["Is_US"] & (df["state"] != "N/A (Outside US)")]
        if len(us_df):
            state_counts = us_df["state"].value_counts().head(12).reset_index()
            state_counts.columns = ["State", "Respondents"]
            fig = px.bar(state_counts.sort_values("Respondents"), x="Respondents", y="State",
                         orientation="h", title="Top 12 US States by Respondent Count",
                         color="Respondents", color_continuous_scale="Oranges")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No US respondents in the current filter selection.")

# ==========================================================================
# TAB 3 — DEMOGRAPHICS
# ==========================================================================
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age", nbins=30, color="treatment",
                            color_discrete_map=TREATMENT_COLORS, barmode="overlay", opacity=0.65,
                            title="Age Distribution by Treatment Status")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gt = df.groupby(["Gender_Clean", "treatment"]).size().reset_index(name="Respondents")
        fig = px.bar(gt, x="Gender_Clean", y="Respondents", color="treatment", barmode="group",
                     color_discrete_map=TREATMENT_COLORS, title="Treatment-Seeking by Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Treatment Rate by Age Group")
    age_treat = df.groupby("Age_Group")["Treatment_Flag"].mean().reindex(AGE_ORDER).dropna() * 100
    fig = px.line(age_treat.reset_index(), x="Age_Group", y="Treatment_Flag", markers=True,
                  labels={"Treatment_Flag": "Treatment Rate (%)"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Treatment Rate by Gender")
    gender_treat = (df.groupby("Gender_Clean")["Treatment_Flag"].mean() * 100).round(1).reset_index()
    gender_treat.columns = ["Gender", "Treatment Rate (%)"]
    st.dataframe(gender_treat, use_container_width=True, hide_index=True)

# ==========================================================================
# TAB 4 — WORKPLACE FACTORS
# ==========================================================================
with tabs[3]:
    st.subheader("Do Workplace Policies Correlate with Treatment-Seeking?")
    policy_cols = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]
    rows = []
    for col in policy_cols:
        for val in df[col].unique():
            sub = df[df[col] == val]
            if len(sub) >= 10:
                rows.append({"Policy": col, "Response": val, "Treatment_Rate": sub["Treatment_Flag"].mean() * 100, "n": len(sub)})
    policy_df = pd.DataFrame(rows)
    fig = px.bar(
        policy_df, x="Policy", y="Treatment_Rate", color="Response", barmode="group",
        title="Treatment Rate (%) by Workplace Policy Response",
        labels={"Treatment_Rate": "Treatment Rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Workplace_Support_Score", color="treatment",
                            color_discrete_map=TREATMENT_COLORS, barmode="group",
                            title="Workplace Support Score (0-5) vs. Treatment",
                            labels={"Workplace_Support_Score": "Support Score (higher = more supportive)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        remote_t = df.groupby("remote_work")["Treatment_Flag"].mean().reset_index()
        remote_t["Treatment_Flag"] *= 100
        fig = px.bar(remote_t, x="remote_work", y="Treatment_Flag", color="remote_work",
                     title="Treatment Rate by Remote Work Status",
                     labels={"Treatment_Flag": "Treatment Rate (%)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ease of Taking Medical Leave")
    leave_order = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
    leave_counts = df["leave"].value_counts().reindex([l for l in leave_order if l in df["leave"].unique()])
    leave_df = leave_counts.reset_index()
    leave_df.columns = ["leave", "Respondents"]
    fig = px.bar(leave_df, x="leave", y="Respondents",
                 title="How Easy Is It to Take Medical Leave for a Mental Health Condition?")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 5 — ATTITUDES & STIGMA
# ==========================================================================
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        mh_counts = df["mental_health_consequence"].value_counts(normalize=True).mul(100).reset_index()
        mh_counts.columns = ["Response", "Percent"]
        mh_counts["Type"] = "Mental Health"
        ph_counts = df["phys_health_consequence"].value_counts(normalize=True).mul(100).reset_index()
        ph_counts.columns = ["Response", "Percent"]
        ph_counts["Type"] = "Physical Health"
        combined = pd.concat([mh_counts, ph_counts], ignore_index=True)
        fig = px.bar(combined, x="Response", y="Percent", color="Type", barmode="group",
                     title="Perceived Negative Consequences: Mental vs. Physical Health Disclosure",
                     labels={"Percent": "% of Respondents"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        mvp = df["mental_vs_physical"].value_counts().reset_index()
        mvp.columns = ["Response", "Respondents"]
        fig = px.pie(mvp, names="Response", values="Respondents", hole=0.4,
                     title="Does Your Employer Take Mental Health as Seriously as Physical Health?")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Willingness to Discuss Mental Health")
    willing = pd.DataFrame({
        "Coworkers": df["coworkers"].value_counts(normalize=True) * 100,
        "Supervisor": df["supervisor"].value_counts(normalize=True) * 100,
    }).fillna(0).reset_index().rename(columns={"index": "Response"})
    willing_melt = willing.melt(id_vars="Response", var_name="Who", value_name="Percent")
    fig = px.bar(willing_melt, x="Response", y="Percent", color="Who", barmode="group",
                 title="Willingness to Discuss a Mental Health Issue")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bringing It Up in a Job Interview: Mental vs. Physical Health")
    interview = pd.DataFrame({
        "Mental Health": df["mental_health_interview"].value_counts(normalize=True) * 100,
        "Physical Health": df["phys_health_interview"].value_counts(normalize=True) * 100,
    }).fillna(0).reset_index().rename(columns={"index": "Response"})
    interview_melt = interview.melt(id_vars="Response", var_name="Type", value_name="Percent")
    fig = px.bar(interview_melt, x="Response", y="Percent", color="Type", barmode="group",
                 title="Would You Bring This Up With a Potential Employer in an Interview?")
    st.plotly_chart(fig, use_container_width=True)

    obs_rate = df["obs_consequence"].value_counts(normalize=True).mul(100).round(1)
    st.info(f"**{obs_rate.get('Yes', 0):.1f}%** of respondents have observed or heard of negative "
            f"consequences for coworkers with mental health conditions at their workplace.")

# ==========================================================================
# TAB 6 — FAMILY HISTORY & WORK INTERFERENCE
# ==========================================================================
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        fh = df.groupby(["Family_History_Flag", "treatment"]).size().reset_index(name="Respondents")
        fh["Family_History_Flag"] = fh["Family_History_Flag"].map({True: "Has Family History", False: "No Family History"})
        fig = px.bar(fh, x="Family_History_Flag", y="Respondents", color="treatment", barmode="group",
                     color_discrete_map=TREATMENT_COLORS, title="Family History vs. Treatment-Seeking")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        wi_order = ["Never", "Rarely", "Sometimes", "Often", "Not applicable"]
        wi_counts = df["work_interfere"].value_counts().reindex(wi_order).dropna().reset_index()
        wi_counts.columns = ["Work_Interfere", "Respondents"]
        fig = px.bar(wi_counts, x="Work_Interfere", y="Respondents", color="Respondents",
                     color_continuous_scale="Reds", title="Does Your Condition Interfere With Work?")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Family History Prevalence by Country (min. 15 respondents)")
    fh_country = (
        df.groupby("Country")
        .agg(Respondents=("Family_History_Flag", "size"), Family_History_Rate=("Family_History_Flag", "mean"))
        .query("Respondents >= 15")
        .sort_values("Family_History_Rate", ascending=False)
    )
    fh_country["Family_History_Rate"] = (fh_country["Family_History_Rate"] * 100).round(1)
    st.dataframe(fh_country.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================================
# TAB 7 — PREDICTORS OF TREATMENT
# ==========================================================================
with tabs[6]:
    st.subheader("Treatment Rate Across Key Factors")
    factors = {
        "Family History": "Family_History_Flag",
        "Work Interferes (Sometimes/Often)": None,
        "Tech Company": "tech_company",
        "Remote Work": "remote_work",
        "Knows Care Options": "care_options",
    }

    d = df.copy()
    d["Work_Interferes_Flag"] = d["work_interfere"].isin(["Sometimes", "Often"])

    predictor_rows = []
    for label, col in [
        ("Has Family History", d["Family_History_Flag"]),
        ("No Family History", ~d["Family_History_Flag"]),
        ("Work Interferes Often/Sometimes", d["Work_Interferes_Flag"]),
        ("Work Rarely/Never Interferes", ~d["Work_Interferes_Flag"]),
        ("Tech Company = Yes", d["tech_company"] == "Yes"),
        ("Tech Company = No", d["tech_company"] == "No"),
        ("Knows Care Options = Yes", d["care_options"] == "Yes"),
        ("Knows Care Options = No", d["care_options"] == "No"),
        ("High Support (score 4-5)", d["Workplace_Support_Score"] >= 4),
        ("Low Support (score 0-1)", d["Workplace_Support_Score"] <= 1),
    ]:
        mask = col
        if mask.sum() >= 10:
            predictor_rows.append({"Factor": label, "Treatment_Rate": d.loc[mask, "Treatment_Flag"].mean() * 100, "n": int(mask.sum())})

    pred_df = pd.DataFrame(predictor_rows).sort_values("Treatment_Rate")
    fig = px.bar(pred_df, x="Treatment_Rate", y="Factor", orientation="h", color="Treatment_Rate",
                 color_continuous_scale="RdYlGn", title="Treatment Rate (%) by Key Factor — Paired Comparisons",
                 labels={"Treatment_Rate": "Treatment Rate (%)"}, hover_data=["n"])
    fig.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Reading this chart: the biggest gap between a paired 'Yes/No' factor and its "
        "counterpart is the strongest signal in this dataset. Family history and "
        "work-interference status typically show the widest gaps — i.e., they're the "
        "strongest predictors of having sought treatment."
    )

    st.subheader("Correlation Heatmap (numeric & boolean features)")
    corr_cols = ["Age", "Treatment_Flag", "Family_History_Flag", "Workplace_Support_Score", "Is_US"]
    corr = d[corr_cols].astype(float).corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 8 — EXPLORE RAW DATA
# ==========================================================================
with tabs[7]:
    st.subheader("Filtered Dataset")
    st.dataframe(df, use_container_width=True, height=500)
    st.download_button(
        "Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mental_health_survey_filtered.csv",
        mime="text/csv",
    )
    st.caption(f"Showing {len(df):,} of {len(df_raw):,} total cleaned responses.")

st.markdown("---")
st.caption(
    "Built for the Mental Health in Tech Survey project · Source: OSMI 2014 survey · "
    "Data pipeline: app.py → SQLite → this dashboard."
)
