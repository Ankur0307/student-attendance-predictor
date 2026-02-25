"""
app.py
------
Streamlit dashboard for the Student Attendance Predictive System.

Tabs:
  1. Attendance Overview  — bar chart of % per subject
  2. Gap Report           — colour-coded detention risk table
  3. Predict Next Class   — ML prediction + probability gauge
  4. SHAP Explanation     — global importance + local waterfall
"""

import sys
from pathlib import Path

# ── make sure ml.* imports resolve when launched with streamlit run ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Student Attendance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        border: 1px solid #2e3450; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c9ef5; }
    .metric-label { font-size: 0.8rem; color: #8b92a5; margin-top: 4px; }
    .status-safe     { color: #22c55e; font-weight: 700; }
    .status-caution  { color: #f59e0b; font-weight: 700; }
    .status-risk     { color: #ef4444; font-weight: 700; }
    .status-detained { color: #dc2626; font-weight: 700; }
    .predict-present { background:#16a34a22; border:1px solid #22c55e;
                       border-radius:10px; padding:1rem; text-align:center; }
    .predict-absent  { background:#dc262622; border:1px solid #ef4444;
                       border-radius:10px; padding:1rem; text-align:center; }
    .pred-label { font-size: 2.5rem; font-weight: 800; }
    .pred-prob  { font-size: 1rem; color: #9ca3af; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li, label { color: #9ca3af !important; }
</style>
""", unsafe_allow_html=True)


# ── lazy imports (keep startup fast) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_raw_data():
    from ml.config import DATASET_PATH
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def _get_student_ids(_df):
    return sorted(_df["student_id"].unique().tolist())


@st.cache_data(show_spinner=False)
def _get_subject_codes(_df, student_id):
    return sorted(_df[_df["student_id"] == student_id]["subject_code"].unique().tolist())


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("🎓 Attendance Predictor")
    st.markdown("---")

    raw_df = _load_raw_data()
    all_ids = _get_student_ids(raw_df)

    student_id = st.selectbox("👤 Student ID", all_ids, index=all_ids.index("ST1001") if "ST1001" in all_ids else 0)
    remaining  = st.slider("📅 Classes remaining this semester", 0, 40, 20)

    st.markdown("---")
    st.caption("Model: GradientBoosting  |  F1 = 0.7810")
    st.caption("Dataset: 107,000 records")


# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"## 🎓 Student Attendance Dashboard")
st.markdown(f"**Student ID:** `{student_id}`  |  **Remaining classes:** `{remaining}`")
st.markdown("---")

# ── compute gap report (used by multiple tabs) ────────────────────────────────
from ml.predict import attendance_gap_report

with st.spinner("Loading attendance data..."):
    gap_df = attendance_gap_report(
        student_id=student_id,
        total_classes_remaining=remaining,
    )

if gap_df.empty:
    st.error(f"❌ Student **{student_id}** not found in the dataset.")
    st.stop()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Attendance Overview",
    "⚠️ Gap Report",
    "🔮 Predict Next Class",
    "🧠 SHAP Explanation",
])


# ═══════════════════════════════════════════════════════
# TAB 1 — Attendance Overview
# ═══════════════════════════════════════════════════════
with tab1:
    st.subheader("Attendance % per Subject")

    # Summary metrics
    total_held     = gap_df["classes_held"].sum()
    total_attended = gap_df["classes_attended"].sum()
    overall_pct    = 100 * total_attended / total_held if total_held > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{total_held}</div>
            <div class='metric-label'>Classes Held</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{total_attended}</div>
            <div class='metric-label'>Classes Attended</div></div>""", unsafe_allow_html=True)
    with c3:
        clr = "#22c55e" if overall_pct >= 75 else "#ef4444"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{clr}'>{overall_pct:.1f}%</div>
            <div class='metric-label'>Overall Attendance</div></div>""", unsafe_allow_html=True)
    with c4:
        n_safe = (gap_df["status"].str.startswith("✅")).sum()
        n_caution = (gap_df["status"].str.startswith("💛")).sum()
        n_risk = (gap_df["status"].str.startswith("🔴")).sum()
        n_detained = (gap_df["status"].str.startswith("❌")).sum()
        risk_icon = "✅" if n_detained == 0 and n_risk == 0 else ("❌" if n_detained > 0 else "⚠️")
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{risk_icon}</div>
            <div class='metric-label'>{n_safe} Safe | {n_caution} Caution | {n_risk} At Risk | {n_detained} Detained</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    subjects  = gap_df["subject_code"].tolist()
    pcts      = gap_df["current_pct"].tolist()
    statuses  = gap_df["status"].tolist()

    def _bar_colour(s):
        if s.startswith("✅"): return "#22c55e"
        if s.startswith("💛"): return "#f59e0b"
        if s.startswith("🔴"): return "#ef4444"
        return "#dc2626"

    colours = [_bar_colour(s) for s in statuses]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")
    bars = ax.bar(subjects, pcts, color=colours, edgecolor="#2e3450", linewidth=0.8, zorder=3)
    ax.axhline(75, color="#7c9ef5", linewidth=1.5, linestyle="--", zorder=4, label="75% Threshold")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color="white", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Attendance %", color="#9ca3af")
    ax.set_xlabel("Subject", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    ax.spines[:].set_color("#2e3450")
    ax.yaxis.grid(True, color="#2e3450", linestyle="--", alpha=0.5, zorder=0)
    legend_patches = [
        mpatches.Patch(color="#22c55e", label="Safe"),
        mpatches.Patch(color="#f59e0b", label="Caution"),
        mpatches.Patch(color="#ef4444", label="At Risk"),
        mpatches.Patch(color="#dc2626", label="Detained"),
        mpatches.Patch(color="#7c9ef5", label="75% Threshold"),
    ]
    ax.legend(handles=legend_patches, facecolor="#1e2130", edgecolor="#2e3450",
              labelcolor="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════
# TAB 2 — Gap Report
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("Detention Risk — Gap Report")

    def _colour_status(val):
        if "Safe"    in str(val): return "color: #22c55e; font-weight: bold"
        if "Caution" in str(val): return "color: #f59e0b; font-weight: bold"
        if "At Risk" in str(val): return "color: #ef4444; font-weight: bold"
        if "Detained" in str(val): return "color: #dc2626; font-weight: bold"
        return ""

    def _colour_pct(val):
        try:
            v = float(val)
            if v >= 75: return "color: #22c55e"
            if v >= 65: return "color: #f59e0b"
            return "color: #ef4444"
        except: return ""

    display_df = gap_df[[
        "subject_code", "subject_name", "classes_held",
        "classes_attended", "current_pct", "remaining_classes",
        "classes_needed_more", "status"
    ]].rename(columns={
        "subject_code":       "Subject",
        "subject_name":       "Name",
        "classes_held":       "Held",
        "classes_attended":   "Attended",
        "current_pct":        "Current %",
        "remaining_classes":  "Remaining",
        "classes_needed_more":"Need More",
        "status":             "Status",
    })

    styled = (
        display_df.style
        .map(_colour_status, subset=["Status"])
        .map(_colour_pct,    subset=["Current %"])
        .set_properties(**{"background-color": "#1e2130", "color": "#e2e8f0",
                           "border": "1px solid #2e3450"})
        .set_table_styles([{"selector": "th",
                            "props": [("background-color", "#252b40"),
                                      ("color", "#7c9ef5"),
                                      ("font-weight", "bold")]}])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Plain-english summaries
    st.markdown("### 📋 Plain-English Summary")
    for _, row in gap_df.iterrows():
        needed = row["classes_needed_more"]
        sub    = row["subject_code"]
        pct    = row["current_pct"]
        status = row["status"]
        if "Safe" in status:
            st.success(f"**{sub}** — {pct:.1f}% ✅ Can miss all remaining classes and stay safe.")
        elif "Caution" in status:
            st.warning(f"**{sub}** — {pct:.1f}% 💛 Must attend at least **{needed}** of the {remaining} remaining classes.")
        elif "At Risk" in status:
            st.error(f"**{sub}** — {pct:.1f}% 🔴 Below threshold! Must attend **{needed}** classes to recover.")
        else:
            st.error(f"**{sub}** — {pct:.1f}% ❌ Detained — mathematically impossible to reach 75% in {remaining} remaining classes.")


# ═══════════════════════════════════════════════════════
# TAB 3 — Predict Next Class
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("Predict Attendance for an Upcoming Class")

    subjects_avail = _get_subject_codes(raw_df, student_id)

    col_l, col_r = st.columns(2)
    with col_l:
        pred_subject  = st.selectbox("Subject Code", subjects_avail)
        pred_faculty  = st.text_input("Faculty ID", value="F101")
        pred_semester = st.text_input("Semester", value="6")
        pred_start    = st.text_input("Class Start (HH:MM)", value="09:00")
        pred_end      = st.text_input("Class End (HH:MM)", value="10:00")
    with col_r:
        pred_dow      = st.selectbox("Day of Week", list(range(7)),
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        pred_exam     = st.toggle("Exam Week?", value=False)
        pred_late     = st.toggle("Historically late?", value=False)
        pred_dom      = st.number_input("Day of Month", 1, 31, 21)
        pred_month    = st.number_input("Month", 1, 12, 2)

    # Auto-compute rolling_pct from historical data
    stu_sub = raw_df[(raw_df["student_id"] == student_id) &
                     (raw_df["subject_code"] == pred_subject)]
    auto_rolling = (100.0 * stu_sub["status"].sum() / len(stu_sub)) if len(stu_sub) > 0 else 75.0
    auto_prev    = int(stu_sub["status"].iloc[-1]) if len(stu_sub) > 0 else 1
    auto_streak  = 0
    if len(stu_sub) > 0:
        for s in stu_sub["status"].values[::-1]:
            if s == 0: auto_streak += 1
            else: break
    auto_weekly  = (100.0 * stu_sub["status"].tail(7).sum() / min(7, len(stu_sub))) if len(stu_sub) > 0 else 75.0

    st.info(f"📈 Auto-computed from history — Rolling: **{auto_rolling:.1f}%** | "
            f"Last class attended: **{'Yes' if auto_prev else 'No'}** | "
            f"Consecutive absences: **{auto_streak}** | "
            f"Weekly %: **{auto_weekly:.1f}%**")

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        from ml.predict import predict_attendance
        row_dict = {
            "student_id":             student_id,
            "semester":               pred_semester,
            "subject_code":           pred_subject,
            "faculty_id":             pred_faculty,
            "class_start_time":       pred_start,
            "class_end_time":         pred_end,
            "day_of_week":            pred_dow,
            "is_exam_week":           int(pred_exam),
            "rolling_attendance_pct": auto_rolling,
            "late_entry":             int(pred_late),
            "day_of_month":           int(pred_dom),
            "month":                  int(pred_month),
            "prev_class_attended":    auto_prev,
            "consecutive_absences":   auto_streak,
            "weekly_attendance_pct":  auto_weekly,
        }
        result = predict_attendance(row_dict)
        label  = result["label"]
        prob   = result["probability_present"]

        if label == "Present":
            st.markdown(f"""<div class='predict-present'>
                <div class='pred-label' style='color:#22c55e'>✅ Present</div>
                <div class='pred-prob'>P(Present) = {prob:.1%}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='predict-absent'>
                <div class='pred-label' style='color:#ef4444'>❌ Absent</div>
                <div class='pred-prob'>P(Present) = {prob:.1%}</div>
            </div>""", unsafe_allow_html=True)

        # Probability gauge bar
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 1.2))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1e2130")
        ax.barh(0, prob,  color="#22c55e", height=0.5)
        ax.barh(0, 1-prob, left=prob, color="#ef4444", height=0.5)
        ax.axvline(0.5, color="white", linewidth=1.5, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#9ca3af")
        ax.spines[:].set_visible(False)
        ax.set_title(f"Probability of Attendance — {prob:.1%} Present", color="#e2e8f0", pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Store for SHAP tab
        st.session_state["last_row_dict"] = row_dict
        st.session_state["last_pred_label"] = label
        st.session_state["last_pred_prob"]  = prob


# ═══════════════════════════════════════════════════════
# TAB 4 — SHAP Explanation
# ═══════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _compute_global_shap(n: int = 300):
    """Compute global SHAP values once and cache. Returns (shap_vals, X_bg_sc, feature_names)."""
    import shap
    import joblib
    from ml.config import ALL_FEATURES, BEST_MODEL_PATH, SCALER_PATH
    from ml.feature_engineering import load_and_engineer

    model  = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    eng_df, _ = load_and_engineer()          # get the fully-engineered dataframe
    X_bg   = eng_df[ALL_FEATURES].fillna(0).sample(n=min(n, len(eng_df)), random_state=42)
    X_bg_sc = pd.DataFrame(scaler.transform(X_bg), columns=ALL_FEATURES)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_bg_sc)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals
    return sv, X_bg_sc, ALL_FEATURES


def _render_local_shap(row_dict: dict, pred_label: str, pred_prob: float):
    """Compute + render a SHAP waterfall for one prediction row."""
    import shap
    import joblib
    from ml.config import ALL_FEATURES, CAT_FEATURES, BEST_MODEL_PATH, SCALER_PATH, LABEL_ENCODERS_PATH

    model    = joblib.load(BEST_MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(LABEL_ENCODERS_PATH)

    df = pd.DataFrame([row_dict])

    # Time features
    if "class_start_time" in df.columns:
        t = df["class_start_time"].iloc[0]
        if isinstance(t, str) and ":" in t:
            df["class_hour"] = int(t.split(":")[0])
    if "class_end_time" in df.columns and "class_start_time" in df.columns:
        s, e = df["class_start_time"].iloc[0], df["class_end_time"].iloc[0]
        if isinstance(s, str) and isinstance(e, str):
            s_m = int(s.split(":")[0])*60 + int(s.split(":")[1])
            e_m = int(e.split(":")[0])*60 + int(e.split(":")[1])
            df["class_duration_min"] = max(0, e_m - s_m)

    for col in CAT_FEATURES:
        le  = encoders[col]
        val = str(df[col].iloc[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else -1

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X    = df[ALL_FEATURES].fillna(0)
    X_sc = pd.DataFrame(scaler.transform(X), columns=ALL_FEATURES)

    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_sc)
    sv_row    = sv[1][0] if isinstance(sv, list) else sv[0]

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()
        base = float(ev[1]) if len(ev) >= 2 else float(ev[0])
    else:
        base = float(ev)

    # ── Waterfall bar chart (manual, Agg-safe) ───────────────────────────────
    order         = np.argsort(np.abs(sv_row))[::-1][:12]
    feats_ordered = [ALL_FEATURES[i] for i in order]
    vals_ordered  = sv_row[order]
    colours       = ["#22c55e" if v > 0 else "#ef4444" for v in vals_ordered]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")
    bars = ax.barh(feats_ordered[::-1], vals_ordered[::-1],
                   color=colours[::-1], edgecolor="#2e3450", linewidth=0.6)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.bar_label(bars, fmt="%+.3f", padding=4, color="white", fontsize=9)
    ax.set_xlabel("SHAP Value  (positive = pushes toward Present)", color="#9ca3af")
    ax.set_title(
        f"Why was '{pred_label}' predicted?  "
        f"P(Present) = {pred_prob:.1%}  |  Base rate = {base:.2f}",
        color="#e2e8f0", fontsize=11, pad=10,
    )
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.spines[:].set_color("#2e3450")
    ax.xaxis.grid(True, color="#2e3450", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Text table ───────────────────────────────────────────────────────────
    with st.expander("📋 Full feature contribution table"):
        rows = []
        for i in np.argsort(np.abs(sv_row))[::-1]:
            rows.append({
                "Feature": ALL_FEATURES[i],
                "SHAP Value": f"{sv_row[i]:+.4f}",
                "Direction": "↑ Present" if sv_row[i] > 0 else "↓ Absent",
                "Raw Value": f"{X_sc.values[0][i]:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab4:
    st.subheader("SHAP Feature Importance")

    # ── Global ────────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Global — Which features matter most across all students?")
    with st.spinner("Computing global SHAP values (cached after first run)..."):
        sv_global, X_bg_sc, feat_names = _compute_global_shap(300)

    mean_abs      = np.abs(sv_global).mean(axis=0)
    order_g       = np.argsort(mean_abs)
    feats_ranked  = [feat_names[i] for i in order_g]
    vals_ranked   = mean_abs[order_g]

    fig_g, ax_g = plt.subplots(figsize=(10, 5))
    fig_g.patch.set_facecolor("#0f1117")
    ax_g.set_facecolor("#1e2130")
    bars_g = ax_g.barh(feats_ranked, vals_ranked,
                       color="#4C72B0", edgecolor="#2e3450", linewidth=0.5)
    ax_g.bar_label(bars_g, fmt="%.4f", padding=4, color="white", fontsize=9)
    ax_g.set_xlabel("Mean |SHAP Value|", color="#9ca3af")
    ax_g.set_title("Global SHAP Feature Importance — GradientBoosting", color="#e2e8f0", pad=10)
    ax_g.tick_params(colors="#9ca3af", labelsize=9)
    ax_g.spines[:].set_color("#2e3450")
    ax_g.xaxis.grid(True, color="#2e3450", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_g)
    plt.close(fig_g)

    st.markdown("---")

    # ── Local ─────────────────────────────────────────────────────────────────
    st.markdown("#### � Local — Why was THIS prediction made?")
    if "last_row_dict" in st.session_state:
        with st.spinner("Computing local SHAP explanation..."):
            _render_local_shap(
                st.session_state["last_row_dict"],
                st.session_state["last_pred_label"],
                st.session_state["last_pred_prob"],
            )
    else:
        st.info("👆 Go to the **Predict Next Class** tab, click **🔮 Predict**, then come back here.")

