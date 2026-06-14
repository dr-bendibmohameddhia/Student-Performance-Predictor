python"""
Student Performance Predictor — Streamlit Dashboard
Author: BENDIB Mohamed Dhia
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — glassmorphism + animations ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
:root {
  --primary: #6C63FF;
  --secondary: #48CAE4;
  --success: #06D6A0;
  --danger:  #EF476F;
  --warning: #FFD166;
  --bg-1:    #0D0D1A;
  --bg-2:    #13132A;
  --glass:   rgba(255,255,255,0.06);
  --glass-b: rgba(255,255,255,0.12);
  --text:    #E8E8F0;
  --muted:   #8888AA;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  color: var(--text);
}

/* ── Background ── */
.stApp {
  background: linear-gradient(135deg, #0D0D1A 0%, #13132A 40%, #0D1A2A 100%);
  min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(13,13,26,0.92) !important;
  border-right: 1px solid rgba(108,99,255,0.2);
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] .stSelectbox > div { background: var(--glass); }

/* ── Glass card ── */
.glass-card {
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 16px;
  padding: 24px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  margin-bottom: 16px;
}
.glass-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(108,99,255,0.15);
}

/* ── KPI Cards ── */
.kpi-card {
  background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(72,202,228,0.08));
  border: 1px solid rgba(108,99,255,0.3);
  border-radius: 16px;
  padding: 20px 24px;
  text-align: center;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  animation: fadeInUp 0.5s ease forwards;
  margin-bottom: 8px;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(circle, rgba(108,99,255,0.08) 0%, transparent 70%);
  animation: pulse 4s ease-in-out infinite;
}
.kpi-value { font-size: 2.2rem; font-weight: 800; margin: 6px 0; }
.kpi-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }
.kpi-icon  { font-size: 1.6rem; margin-bottom: 4px; }

/* ── Accent colors ── */
.accent-purple { color: var(--primary); }
.accent-cyan   { color: var(--secondary); }
.accent-green  { color: var(--success); }
.accent-red    { color: var(--danger); }
.accent-yellow { color: var(--warning); }

/* ── Section header ── */
.section-header {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(90deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 16px;
}

/* ── Hero ── */
.hero-banner {
  background: linear-gradient(135deg, rgba(108,99,255,0.2) 0%, rgba(72,202,228,0.1) 50%, rgba(6,214,160,0.08) 100%);
  border: 1px solid rgba(108,99,255,0.25);
  border-radius: 20px;
  padding: 36px 40px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero-banner::after {
  content: '🎓';
  position: absolute;
  right: 30px; top: 50%;
  transform: translateY(-50%);
  font-size: 5rem;
  opacity: 0.15;
}
.hero-title { font-size: 2.4rem; font-weight: 800; margin: 0; line-height: 1.2; }
.hero-sub   { color: var(--muted); font-size: 1.05rem; margin-top: 8px; }

/* ── Risk badge ── */
.risk-high   { background: rgba(239,71,111,0.15); border: 1px solid var(--danger); color: var(--danger); border-radius: 8px; padding: 6px 16px; font-weight: 600; display:inline-block; }
.risk-medium { background: rgba(255,209,102,0.15); border: 1px solid var(--warning); color: var(--warning); border-radius: 8px; padding: 6px 16px; font-weight: 600; display:inline-block; }
.risk-low    { background: rgba(6,214,160,0.15); border: 1px solid var(--success); color: var(--success); border-radius: 8px; padding: 6px 16px; font-weight: 600; display:inline-block; }

/* ── Prediction result ── */
.pred-pass { background: linear-gradient(135deg, rgba(6,214,160,0.2), rgba(6,214,160,0.05)); border: 2px solid var(--success); border-radius: 20px; padding: 28px; text-align: center; }
.pred-fail { background: linear-gradient(135deg, rgba(239,71,111,0.2), rgba(239,71,111,0.05)); border: 2px solid var(--danger); border-radius: 20px; padding: 28px; text-align: center; }
.pred-title { font-size: 1.8rem; font-weight: 800; }
.pred-prob  { font-size: 1rem; color: var(--muted); margin-top: 6px; }

/* ── Progress bar ── */
.prob-bar-outer { background: rgba(255,255,255,0.08); border-radius: 50px; height: 12px; margin: 12px 0; }
.prob-bar-inner { height: 12px; border-radius: 50px; transition: width 0.8s ease; }

/* ── Animations ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 0.5; }
  50%       { transform: scale(1.1); opacity: 1; }
}

/* ── Streamlit overrides ── */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), #8B7FFF);
  color: white; border: none; border-radius: 10px;
  padding: 10px 28px; font-weight: 600; font-size: 0.95rem;
  transition: all 0.3s ease; cursor: pointer; width: 100%;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(108,99,255,0.4);
}
.stSlider > div > div { color: var(--primary) !important; }
div[data-testid="metric-container"] {
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 12px;
  padding: 12px 16px;
}
h1,h2,h3 { color: var(--text); }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 10px; }
.stTabs [data-baseweb="tab"]      { color: var(--muted); font-weight: 500; }
.stTabs [aria-selected="true"]    { color: var(--primary) !important; border-bottom: 2px solid var(--primary); }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/student_processed.csv")

@st.cache_resource
def load_artifacts():
    def _l(fn):
        with open(f"data/models/{fn}", "rb") as f:
            return pickle.load(f)
    with open("data/models/results.json") as f:
        results = json.load(f)
    return {
        "lr":      _l("logistic_regression.pkl"),
        "rf":      _l("random_forest.pkl"),
        "xgb":     _l("xgboost.pkl"),
        "lgb":     _l("lightgbm.pkl"),
        "scaler":  _l("scaler.pkl"),
        "imputer": _l("imputer.pkl"),
        "features":_l("feature_cols.pkl"),
        "results": results,
    }

def kpi(icon, value, label, color="accent-purple"):
    return f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value {color}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""

def prob_bar(p, color):
    return f"""
    <div class="prob-bar-outer">
      <div class="prob-bar-inner" style="width:{p*100:.1f}%;background:{color};"></div>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:2.2rem'>🎓</div>
      <div style='font-size:1.1rem;font-weight:700;color:#6C63FF'>Student Predictor</div>
      <div style='font-size:0.72rem;color:#8888AA;margin-top:2px'>AI-Powered Analytics</div>
    </div>
    <hr style='border-color:rgba(108,99,255,0.2);margin:12px 0'>
    """, unsafe_allow_html=True)

    page = st.selectbox("📍 Navigation", [
        "🏠 Dashboard",
        "🔮 Predict Student",
        "📊 Analytics",
        "🧠 Explainability",
        "⚖️ Model Comparison",
    ])

    st.markdown("<hr style='border-color:rgba(108,99,255,0.15)'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:12px;background:rgba(108,99,255,0.08);border-radius:10px;font-size:0.8rem;color:#8888AA'>
      <b style='color:#6C63FF'>Dataset:</b> UCI Student Performance<br>
      <b style='color:#6C63FF'>Models:</b> LR · RF · XGB · LGBM<br>
      <b style='color:#6C63FF'>Explainability:</b> SHAP Values
    </div>
    """, unsafe_allow_html=True)

df   = load_data()
arts = load_artifacts()
res  = arts["results"]["results"]
best = arts["results"]["best_model"]


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    import plotly.graph_objects as go

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">Student Performance <span style='color:#6C63FF'>Predictor</span></div>
      <div class="hero-sub">AI-powered early warning system · Identify at-risk students · Drive interventions</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──
    pass_rate   = df["passed"].mean()
    avg_grade   = df["final_grade"].mean()
    at_risk     = (df["passed"] == 0).sum()
    best_acc    = res[best]["accuracy"]

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi("👩‍🎓", f"{len(df):,}", "Total Students", "accent-purple"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("✅", f"{pass_rate:.1%}", "Pass Rate", "accent-green"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("⚠️", f"{at_risk:,}", "At-Risk Students", "accent-red"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("🤖", f"{best_acc:.1%}", "Model Accuracy", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ──
    from src.visualization.charts import (
        grade_distribution, risk_scatter, study_time_pass_rate, absence_impact
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(grade_distribution(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(study_time_pass_rate(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(risk_scatter(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(absence_impact(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── At-risk table ──
    st.markdown('<div class="section-header">⚠️ At-Risk Student Profiles</div>', unsafe_allow_html=True)
    at_risk_df = df[df["passed"] == 0][
        ["age","study_time","past_failures","absences","attendance_rate","final_grade","at_risk_score"]
    ].head(10).reset_index(drop=True)
    at_risk_df.columns = ["Age","Study Time","Past Failures","Absences","Attendance %","Final Grade","Risk Score"]
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.dataframe(
        at_risk_df.style
            .background_gradient(subset=["Risk Score"], cmap="Reds")
            .background_gradient(subset=["Absences"], cmap="Oranges"),
        use_container_width=True, hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict Student
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Student":
    st.markdown('<div class="section-header">🔮 Student Pass/Fail Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8888AA">Fill in student details to predict academic outcome with AI.</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📋 Demographics**")
            age    = st.slider("Age", 15, 22, 17)
            gender = st.selectbox("Gender", ["M", "F"])
            address= st.selectbox("Address", ["Urban", "Rural"])
            internet = st.selectbox("Internet Access", [1, 0], format_func=lambda x: "Yes" if x else "No")
            romantic = st.selectbox("In Romantic Relationship", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with c2:
            st.markdown("**📚 Academic Profile**")
            study_time   = st.slider("Weekly Study Time (1=<2h, 4=>10h)", 1, 4, 2)
            past_failures= st.slider("Past Failures", 0, 3, 0)
            absences     = st.slider("Number of Absences", 0, 45, 4)
            attendance   = st.slider("Attendance Rate (%)", 40.0, 100.0, 88.0)
            g1 = st.slider("Grade Period 1 (0-20)", 0, 20, 12)
            g2 = st.slider("Grade Period 2 (0-20)", 0, 20, 13)

        with c3:
            st.markdown("**🏠 Family & Lifestyle**")
            mother_edu   = st.slider("Mother Education (0-4)", 0, 4, 2)
            father_edu   = st.slider("Father Education (0-4)", 0, 4, 2)
            family_sup   = st.selectbox("Family Support", [1, 0], format_func=lambda x: "Yes" if x else "No")
            school_sup   = st.selectbox("School Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
            go_out       = st.slider("Goes Out (1-5)", 1, 5, 3)
            workday_alc  = st.slider("Workday Alcohol (1-5)", 1, 5, 1)
            weekend_alc  = st.slider("Weekend Alcohol (1-5)", 1, 5, 2)
            health       = st.slider("Health Status (1-5)", 1, 5, 3)

        submitted = st.form_submit_button("🔮 Predict Now")

    if submitted:
        # Build feature vector matching training columns
        gender_enc  = 1 if gender == "M" else 0
        address_enc = 1 if address == "Urban" else 0

        # Engineered features
        parent_edu_avg    = (mother_edu + father_edu) / 2
        alcohol_total     = workday_alc + weekend_alc
        support_score     = school_sup + family_sup
        academic_history  = (g1 + g2) / 2
        grade_improvement = g2 - g1
        study_efficiency  = study_time / (go_out + 1)
        at_risk_sc        = past_failures * 2 + alcohol_total + absences // 5
        absence_cat       = 0 if absences<=2 else (1 if absences<=7 else (2 if absences<=15 else 3))

        feature_vals = np.array([
            age, gender_enc, address_enc,
            1, 1,                         # family_size, parent_status (default GT3, Together)
            mother_edu, father_edu,
            4, 2,                         # mother_job, father_job (Other, Services)
            1, 0,                         # reason, guardian
            2, study_time,                # travel_time
            past_failures, school_sup, family_sup,
            0, 1, 1,                      # paid_classes, activities, nursery
            1, internet, romantic,        # higher_education_goal
            3, 3, go_out,                 # family_relations, free_time
            workday_alc, weekend_alc, health,
            absences, attendance, g1, g2,
            0,                            # final_grade placeholder
            parent_edu_avg, alcohol_total, support_score,
            academic_history, grade_improvement,
            absence_cat, study_efficiency, at_risk_sc,
        ], dtype=float)

        features  = arts["features"]
        n_feat    = len(features)
        feat_vec  = feature_vals[:n_feat]
        if len(feat_vec) < n_feat:
            feat_vec = np.pad(feat_vec, (0, n_feat - len(feat_vec)))

        imputed = arts["imputer"].transform(feat_vec.reshape(1, -1))
        scaled  = arts["scaler"].transform(imputed)

        model_name_map = {
            "Logistic Regression": arts["lr"],
            "Random Forest":       arts["rf"],
            "XGBoost":             arts["xgb"],
            "LightGBM":            arts["lgb"],
        }
        use_scaled = {"Logistic Regression"}

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

        cols = st.columns(len(model_name_map))
        for i, (mname, model) in enumerate(model_name_map.items()):
            X = scaled if mname in use_scaled else imputed
            pred  = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0, 1])
            color = "#06D6A0" if pred == 1 else "#EF476F"
            icon  = "✅" if pred == 1 else "❌"
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center">
                  <div style="font-size:0.75rem;color:#8888AA;margin-bottom:8px">{mname}</div>
                  <div style="font-size:2rem">{icon}</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{color}">{'PASS' if pred==1 else 'FAIL'}</div>
                  <div style="font-size:0.85rem;color:#8888AA;margin-top:4px">Confidence</div>
                  <div style="font-size:1.3rem;font-weight:800;color:{color}">{proba:.1%}</div>
                  {prob_bar(proba, color)}
                </div>
                """, unsafe_allow_html=True)

        # ── Best model detail ──
        best_model = model_name_map[best]
        X_best = scaled if best in use_scaled else imputed
        pred_b = int(best_model.predict(X_best)[0])
        prob_b = float(best_model.predict_proba(X_best)[0, 1])
        risk_pct = 1 - prob_b

        st.markdown(f"""
        <div class="{'pred-pass' if pred_b==1 else 'pred-fail'}" style="margin-top:20px">
          <div style="font-size:0.8rem;color:#8888AA;margin-bottom:8px">Best Model · {best}</div>
          <div class="pred-title">{'🎉 Predicted to PASS' if pred_b==1 else '⚠️ Predicted to FAIL'}</div>
          <div class="pred-prob">Pass probability: <b>{prob_b:.1%}</b> · Risk score: <b>{at_risk_sc:.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Risk level
        if risk_pct < 0.3:
            badge = '<span class="risk-low">🟢 LOW RISK</span>'
            advice = "Student is on track. Maintain current study habits."
        elif risk_pct < 0.6:
            badge = '<span class="risk-medium">🟡 MEDIUM RISK</span>'
            advice = "Consider additional tutoring and monitoring attendance."
        else:
            badge = '<span class="risk-high">🔴 HIGH RISK</span>'
            advice = "Immediate intervention needed. Schedule counseling session."

        st.markdown(f"""
        <div class="glass-card" style="margin-top:16px">
          <b>Risk Level:</b> {badge}<br><br>
          <b>💡 Recommendation:</b> {advice}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analytics
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    from src.visualization.charts import correlation_heatmap, gauge_chart
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<div class="section-header">📊 Deep Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "👥 Demographics"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = px.histogram(df, x="final_grade", color="passed",
                               color_discrete_map={1:"#06D6A0", 0:"#EF476F"},
                               nbins=20, title="Final Grade Distribution",
                               labels={"passed": "Outcome"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#E0E0E0")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig2 = px.box(df, x="study_time", y="final_grade", color="passed",
                          color_discrete_map={1:"#06D6A0", 0:"#EF476F"},
                          title="Grade by Study Time",
                          labels={"study_time":"Study Level","final_grade":"Final Grade"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#E0E0E0")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig3 = px.scatter(df.sample(400, random_state=1),
                              x="grade_period1", y="grade_period2",
                              color="passed", size="final_grade",
                              color_discrete_map={1:"#06D6A0", 0:"#EF476F"},
                              title="G1 vs G2 Grades")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#E0E0E0")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig4 = px.violin(df, x="past_failures", y="final_grade", color="passed",
                             color_discrete_map={1:"#06D6A0", 0:"#EF476F"},
                             title="Final Grade vs Past Failures")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#E0E0E0")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        num_cols = ["age","study_time","past_failures","absences","attendance_rate",
                    "grade_period1","grade_period2","final_grade","alcohol_total",
                    "support_score","at_risk_score","academic_history"]
        corr = df[num_cols].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,"#EF476F"],[0.5,"#1A1A3A"],[1,"#06D6A0"]],
            zmid=0, text=np.round(corr.values,2), texttemplate="%{text}",
        ))
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#E0E0E0"), height=550,
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Top correlations with target
        corr_target = df[num_cols + ["passed"]].corr()["passed"].drop("passed").abs().sort_values(ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🔝 Top Positive Correlations with Passing**")
            pos = df[num_cols + ["passed"]].corr()["passed"].drop("passed").sort_values(ascending=False).head(6)
            for feat, v in pos.items():
                color = "#06D6A0" if v > 0 else "#EF476F"
                st.markdown(f"<span style='color:{color};font-weight:600'>{v:+.3f}</span> — {feat}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**⬇️ Top Negative Correlations with Passing**")
            neg = df[num_cols + ["passed"]].corr()["passed"].drop("passed").sort_values().head(6)
            for feat, v in neg.items():
                st.markdown(f"<span style='color:#EF476F;font-weight:600'>{v:+.3f}</span> — {feat}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            gender_pass = df.groupby("gender")["passed"].mean()
            fig_g = go.Figure(go.Bar(
                x=["Male (0)", "Female (1)"], y=gender_pass.values,
                marker_color=["#6C63FF","#48CAE4"],
                text=[f"{v:.1%}" for v in gender_pass.values], textposition="outside",
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#E0E0E0", title="Pass Rate by Gender",
                                yaxis=dict(tickformat=".0%", range=[0,1.1]),
                                margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            addr_pass = df.groupby("address")["passed"].mean()
            fig_a = go.Figure(go.Bar(
                x=["Rural (0)", "Urban (1)"], y=addr_pass.values,
                marker_color=["#FFD166","#06D6A0"],
                text=[f"{v:.1%}" for v in addr_pass.values], textposition="outside",
            ))
            fig_a.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#E0E0E0", title="Pass Rate by Address",
                                yaxis=dict(tickformat=".0%", range=[0,1.1]),
                                margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            age_pass = df.groupby("age")["passed"].mean().reset_index()
            fig_age = go.Figure(go.Scatter(
                x=age_pass["age"], y=age_pass["passed"],
                mode="lines+markers",
                line=dict(color="#6C63FF", width=2.5),
                marker=dict(size=8, color="#48CAE4"),
            ))
            fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#E0E0E0", title="Pass Rate by Age",
                                  yaxis=dict(tickformat=".0%"),
                                  margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Explainability
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    import plotly.graph_objects as go

    st.markdown('<div class="section-header">🧠 Model Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8888AA">Understand <i>why</i> the model makes each prediction — trust through transparency.</p>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model for SHAP Analysis",
                                ["Random Forest", "XGBoost", "LightGBM"])

    with st.spinner("⚡ Computing SHAP values..."):
        try:
            from src.explainability.shap_explainer import get_shap_explainer, top_features_df
            _, shap_vals, X_sample, feat_names = get_shap_explainer(model_choice)
            feat_df = top_features_df(shap_vals, feat_names, n=15)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                feat_sorted = feat_df.sort_values("SHAP Importance")
                fig = go.Figure(go.Bar(
                    x=feat_sorted["SHAP Importance"],
                    y=feat_sorted["Feature"],
                    orientation="h",
                    marker=dict(
                        color=feat_sorted["SHAP Importance"],
                        colorscale=[[0,"#6C63FF"],[0.5,"#48CAE4"],[1,"#06D6A0"]],
                        line=dict(width=0),
                    ),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E0E0E0"), title=f"SHAP Feature Importance — {model_choice}",
                    xaxis=dict(title="Mean |SHAP Value|", gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    height=480, margin=dict(l=10,r=10,t=40,b=10),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**📋 SHAP Rankings**")
                for i, row in feat_df.iterrows():
                    pct = row["SHAP Importance"] / feat_df["SHAP Importance"].max()
                    color = "#6C63FF" if pct > 0.6 else ("#48CAE4" if pct > 0.3 else "#8888AA")
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
                      <span style="font-size:0.8rem;color:#E0E0E0">{row['Feature']}</span>
                      <span style="font-size:0.8rem;color:{color};font-weight:600">{row['SHAP Importance']:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── SHAP summary scatter ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🔍 SHAP Value Distribution (Top 10 Features)**")
            top10 = feat_df.head(10)["Feature"].tolist()
            feat_idx = [feat_names.index(f) for f in top10 if f in feat_names]

            fig2 = go.Figure()
            colors = ["#6C63FF","#48CAE4","#06D6A0","#FFD166","#EF476F",
                      "#8B7FFF","#70D6F5","#3DFFA0","#FFDF80","#FF7096"]
            for i, (idx, fname) in enumerate(zip(feat_idx, top10)):
                fig2.add_trace(go.Scatter(
                    x=shap_vals[:, idx],
                    y=[fname] * len(shap_vals),
                    mode="markers",
                    marker=dict(size=5, color=colors[i % len(colors)], opacity=0.6),
                    name=fname,
                ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E0E0E0"),
                title="SHAP Values Distribution per Feature",
                xaxis=dict(title="SHAP Value", gridcolor="rgba(255,255,255,0.06)", zeroline=True,
                           zerolinecolor="rgba(255,255,255,0.2)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                height=420, showlegend=False, margin=dict(l=10,r=10,t=40,b=10),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"SHAP computation error: {e}")
            st.info("Using feature importance from Random Forest as fallback.")
            rf = arts["rf"]
            feat_names = arts["features"]
            imp = rf.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            fallback = pd.DataFrame({"Feature": [feat_names[i] for i in idx],
                                     "Importance": imp[idx]})
            fig = go.Figure(go.Bar(
                x=fallback["Importance"], y=fallback["Feature"],
                orientation="h", marker_color="#6C63FF",
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E0E0E0"), title="RF Feature Importance")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Model Comparison":
    import plotly.graph_objects as go
    from src.visualization.charts import gauge_chart

    st.markdown('<div class="section-header">⚖️ Model Comparison & Selection</div>', unsafe_allow_html=True)

    # ── Metrics cards ──
    cols = st.columns(4)
    model_icons = {"Logistic Regression":"📐","Random Forest":"🌲","XGBoost":"⚡","LightGBM":"💡"}
    for i, (mname, metrics) in enumerate(res.items()):
        with cols[i]:
            is_best = mname == best
            border = "border:2px solid #6C63FF;" if is_best else ""
            badge  = "🏆 BEST" if is_best else ""
            st.markdown(f"""
            <div class="glass-card" style="{border}text-align:center">
              <div style="font-size:2rem">{model_icons.get(mname,'🤖')}</div>
              <div style="font-weight:700;font-size:0.9rem;margin:6px 0">{mname}</div>
              {f'<span style="background:#6C63FF22;color:#6C63FF;padding:2px 10px;border-radius:20px;font-size:0.7rem;font-weight:700">{badge}</span>' if is_best else ''}
              <div style="margin-top:12px">
                <div style="color:#8888AA;font-size:0.7rem">ACCURACY</div>
                <div style="color:#06D6A0;font-size:1.5rem;font-weight:800">{metrics['accuracy']:.3f}</div>
              </div>
              <div style="margin-top:8px">
                <div style="color:#8888AA;font-size:0.7rem">ROC-AUC</div>
                <div style="color:#48CAE4;font-size:1.5rem;font-weight:800">{metrics['auc']:.3f}</div>
              </div>
              <div style="margin-top:8px">
                <div style="color:#8888AA;font-size:0.7rem">F1 SCORE</div>
                <div style="color:#6C63FF;font-size:1.5rem;font-weight:800">{metrics['f1']:.3f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Bar chart ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    names   = list(res.keys())
    metrics = ["accuracy", "auc", "f1"]
    colors  = ["#6C63FF", "#48CAE4", "#06D6A0"]
    fig = go.Figure()
    for m, c in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=m.upper(), x=names, y=[res[n][m] for n in names],
            marker_color=c, opacity=0.88,
            text=[f"{res[n][m]:.3f}" for n in names], textposition="outside",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"), barmode="group",
        title="All Models — Accuracy · AUC · F1",
        yaxis=dict(range=[0.85, 1.01], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10,r=10,t=40,b=10),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Gauge charts for best model ──
    st.markdown(f'<div class="section-header">🏆 Best Model Details — {best}</div>', unsafe_allow_html=True)
    g1c, g2c, g3c = st.columns(3)
    bm = res[best]
    with g1c:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(bm["accuracy"], "Accuracy"), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with g2c:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(bm["auc"], "ROC-AUC"), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with g3c:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(bm["f1"], "F1 Score"), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Why this model? ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"""
    **🤔 Why {best}?**

    The model was selected based on highest **ROC-AUC** score, which measures discriminative power
    independent of classification threshold — crucial for imbalanced educational datasets where
    we care equally about catching at-risk students AND not over-flagging healthy performers.

    | Criterion | Rationale |
    |-----------|-----------|
    | ROC-AUC   | Best for class-imbalanced binary classification |
    | F1 Score  | Balances precision and recall for intervention planning |
    | Accuracy  | Overall correctness across all students |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:30px 0 10px;color:#444466;font-size:0.78rem'>
  🎓 Student Performance Predictor · Developped by BENDIB Mohamed Dhia
</div>
""", unsafe_allow_html=True)
