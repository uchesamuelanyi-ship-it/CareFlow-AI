
import streamlit as st
import pandas as pd
import joblib
import math

st.set_page_config(
    page_title="CareFlow AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- GLOBAL STYLE ----------
st.markdown("""
<style>
:root{
    --navy:#0b1324;
    --navy2:#13233f;
    --teal:#0f766e;
    --blue:#2563eb;
    --ink:#0f172a;
    --muted:#64748b;
    --line:#e2e8f0;
    --soft:#f8fafc;
}
.stApp{
    background: linear-gradient(180deg,#f7fbff 0%,#ffffff 48%,#f8fafc 100%);
}
.block-container{
    padding-top:1.6rem;
    padding-bottom:3rem;
    max-width:1400px;
}
.hero{
    background:
      radial-gradient(circle at 90% 10%, rgba(45,212,191,.26), transparent 24%),
      linear-gradient(135deg,var(--navy) 0%,var(--navy2) 55%,#0f766e 100%);
    color:white;
    border-radius:26px;
    padding:38px 42px;
    box-shadow:0 18px 45px rgba(15,23,42,.18);
    margin-bottom:24px;
}
.hero .eyebrow{
    color:#99f6e4;
    letter-spacing:.12em;
    font-size:.78rem;
    font-weight:700;
    text-transform:uppercase;
}
.hero h1{font-size:2.65rem;margin:.35rem 0 .5rem 0;line-height:1.08}
.hero p{color:#dbeafe;font-size:1.08rem;max-width:780px;margin:0}
.panel{
    background:white;
    border:1px solid var(--line);
    border-radius:20px;
    padding:22px;
    box-shadow:0 10px 28px rgba(15,23,42,.06);
}
.kpi{
    background:white;
    border:1px solid var(--line);
    border-radius:18px;
    padding:20px;
    box-shadow:0 8px 24px rgba(15,23,42,.05);
    min-height:125px;
}
.kpi .label{color:var(--muted);font-size:.9rem}
.kpi .value{color:var(--ink);font-size:1.75rem;font-weight:800;margin:.2rem 0}
.kpi .sub{color:var(--muted);font-size:.84rem}
.section-title{
    font-size:1.35rem;font-weight:800;color:var(--ink);margin-bottom:.25rem
}
.section-sub{color:var(--muted);margin-bottom:1rem}
.login-wrap{
    max-width:1040px;
    margin:0 auto;
}
.login-card{
    background:white;
    border:1px solid var(--line);
    border-radius:24px;
    padding:30px;
    box-shadow:0 16px 40px rgba(15,23,42,.08);
}
.feature{
    background:#fff;
    border:1px solid var(--line);
    border-radius:18px;
    padding:18px;
    min-height:145px;
}
.feature h4{margin:.35rem 0;color:var(--ink)}
.feature p{color:var(--muted);font-size:.92rem}
.result-low{background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;border-radius:18px;padding:22px}
.result-mid{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:18px;padding:22px}
.result-high{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;border-radius:18px;padding:22px}
.step{
    display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-bottom:1px solid #eef2f7
}
.step-num{
    width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    background:#e0f2fe;color:#075985;font-weight:800;flex:0 0 32px
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1324 0%,#13233f 100%);
}
[data-testid="stSidebar"] *{color:#e2e8f0}
div.stButton > button[kind="primary"]{
    background:linear-gradient(90deg,#2563eb,#0f766e);
    border:none;
}
hr{border-color:#e2e8f0}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("best_waiting_time_model.joblib")

model = load_model()

# ---------- AUTH CONFIG ----------
def auth_ready():
    try:
        return (
            "auth" in st.secrets
            and "google" in st.secrets["auth"]
            and bool(st.secrets["auth"]["google"].get("client_id"))
            and bool(st.secrets["auth"]["google"].get("client_secret"))
        )
    except Exception:
        return False

AUTH_READY = auth_ready()

def signed_in():
    try:
        return AUTH_READY and st.user.is_logged_in
    except Exception:
        return False

# ---------- PUBLIC LANDING / LOGIN ----------
if not signed_in():
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Secure Research Portal</div>
      <h1>🏥 CareFlow AI</h1>
      <p>
        A machine-learning decision-support system for estimating outpatient waiting time
        before physician consultation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.15, .85])
    with left:
        st.markdown("""
        <div class="login-card">
          <div class="section-title">Welcome to CareFlow AI</div>
          <div class="section-sub">
            Sign in before accessing the prediction dashboard, model performance,
            research information and patient-flow tools.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if AUTH_READY:
            st.write("")
            if st.button("Continue with Google", type="primary", use_container_width=True):
                st.login("google")
        else:
            st.warning(
                "Google sign-in is not configured on this local copy yet. "
                "After public deployment, add the Google OAuth credentials in Streamlit secrets."
            )
            st.caption("For local interface testing only, use Preview Mode below.")
            if st.button("Preview Dashboard Locally", use_container_width=True):
                st.session_state["local_preview"] = True

        if st.session_state.get("local_preview"):
            st.session_state["preview_access"] = True

        st.markdown("""
        <div style="margin-top:16px;color:#64748b;font-size:.9rem">
          By continuing, users access a research prototype intended for academic evaluation.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
          <div class="section-title">What you can do</div>
          <div class="step"><div class="step-num">1</div><div><b>Predict</b><br><span style="color:#64748b">Estimate expected waiting time from live queue conditions.</span></div></div>
          <div class="step"><div class="step-num">2</div><div><b>Review</b><br><span style="color:#64748b">See the selected model and evaluation metrics.</span></div></div>
          <div class="step"><div class="step-num">3</div><div><b>Understand</b><br><span style="color:#64748b">Review the research purpose, method and limitations.</span></div></div>
          <div class="step" style="border-bottom:none"><div class="step-num">4</div><div><b>Operate</b><br><span style="color:#64748b">Use a clean dashboard designed for GOPD workflow demonstration.</span></div></div>
        </div>
        """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature"><div>⚡</div><h4>Fast Prediction</h4><p>Generate an estimated waiting time from current outpatient-flow variables.</p></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature"><div>📊</div><h4>Model Evidence</h4><p>View the selected XGBoost model and key held-out test metrics.</p></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature"><div>🔒</div><h4>Controlled Access</h4><p>Public deployment can require Google sign-in before dashboard access.</p></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.get("preview_access"):
        st.stop()

# ---------- AUTHENTICATED / PREVIEW APP ----------
user_name = "Preview User"
user_email = "Local preview"
if signed_in():
    try:
        user_name = getattr(st.user, "name", None) or getattr(st.user, "email", "User")
        user_email = getattr(st.user, "email", "")
    except Exception:
        pass

with st.sidebar:
    st.markdown("## 🏥 CareFlow AI")
    st.caption("Patient Waiting-Time Portal")
    st.divider()

    nav = st.radio(
        "Navigation",
        ["Dashboard", "Predict Waiting Time", "Model Performance", "About the Project", "My Profile"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Signed in")
    st.write(f"**{user_name}**")
    st.caption(user_email)

    if signed_in():
        if st.button("Sign out", use_container_width=True):
            st.logout()
    elif st.session_state.get("preview_access"):
        if st.button("Exit Preview", use_container_width=True):
            st.session_state["preview_access"] = False
            st.rerun()

# ---------- DASHBOARD ----------
if nav == "Dashboard":
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">Operations Dashboard</div>
      <h1>Welcome, {user_name}</h1>
      <p>Monitor the research prototype, review model evidence and access the waiting-time prediction tool.</p>
    </div>
    """, unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    cards = [
        ("Selected Model","XGBoost","Regression"),
        ("Test MAE","9.27 min","Average absolute error"),
        ("Test RMSE","12.00 min","Penalises large errors"),
        ("Test R²","0.872","Held-out test fit"),
    ]
    for col,(lab,val,sub) in zip([a,b,c,d],cards):
        with col:
            st.markdown(f'<div class="kpi"><div class="label">{lab}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1,c2 = st.columns([1.1,.9])
    with c1:
        st.markdown("""
        <div class="panel">
          <div class="section-title">System Workflow</div>
          <div class="section-sub">How the prediction pipeline operates.</div>
          <div class="step"><div class="step-num">1</div><div><b>Enter current GOPD conditions</b><br><span style="color:#64748b">Day, patient type, payment type, queue length, doctors and arrival time.</span></div></div>
          <div class="step"><div class="step-num">2</div><div><b>Preprocess inputs</b><br><span style="color:#64748b">Apply the same trained transformations used during model development.</span></div></div>
          <div class="step"><div class="step-num">3</div><div><b>Run XGBoost model</b><br><span style="color:#64748b">Estimate the expected waiting time in minutes.</span></div></div>
          <div class="step" style="border-bottom:none"><div class="step-num">4</div><div><b>Display operational result</b><br><span style="color:#64748b">Show waiting time, congestion level and queue-to-doctor ratio.</span></div></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="panel">
          <div class="section-title">Quick Notes</div>
          <div class="section-sub">Current research status.</div>
          <p><b>Language:</b> Python</p>
          <p><b>Framework:</b> Streamlit</p>
          <p><b>Model:</b> XGBoost Regression</p>
          <p><b>Purpose:</b> Academic research prototype</p>
          <p><b>Deployment:</b> Local now; public deployment can enable Google sign-in.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------- PREDICT ----------
elif nav == "Predict Waiting Time":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Prediction Workspace</div>
      <h1>Patient Waiting Time Predictor</h1>
      <p>Enter the current outpatient-flow conditions to estimate time before physician consultation.</p>
    </div>
    """, unsafe_allow_html=True)

    left,right = st.columns([1.25,.95])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient & Queue Details</div>', unsafe_allow_html=True)
        st.caption("Use values observed when the patient arrives.")
        with st.form("predict_form"):
            c1,c2 = st.columns(2)
            with c1:
                day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday"])
                ptype = st.selectbox("Patient Type", ["Returning","New"])
                payment = st.selectbox("Payment Type", ["NHIA/Insurance","Cash","Other"])
            with c2:
                queue = st.number_input("Queue Length at Arrival", min_value=0, max_value=100, value=15, step=1)
                doctors = st.number_input("Doctors on Duty", min_value=1, max_value=20, value=4, step=1)
                hour = st.slider("Arrival Time", 7.0, 16.99, 9.0, 0.25, help="9.5 represents approximately 9:30 AM")
            go = st.form_submit_button("Predict Waiting Time", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        if go:
            occ = queue / doctors
            row = pd.DataFrame([{
                "day_of_week": day,
                "patient_type": ptype,
                "payment_type": payment,
                "queue_length_at_arrival": queue,
                "doctors_on_duty": doctors,
                "arrival_hour": hour,
                "occupancy_ratio": occ,
            }])
            pred = max(0.0, float(model.predict(row)[0]))

            if pred < 30:
                css, level, msg = "result-low","Low","Current conditions suggest relatively light congestion."
            elif pred < 60:
                css, level, msg = "result-mid","Moderate","Monitor queue growth and physician availability."
            else:
                css, level, msg = "result-high","High","Current conditions suggest a heavier operational load."

            st.markdown(f"""
            <div class="{css}">
              <div style="font-size:.92rem">Estimated Waiting Time</div>
              <div style="font-size:2.7rem;font-weight:800;margin:.1rem 0">{pred:.1f} minutes</div>
              <div><b>Congestion Level:</b> {level}</div>
              <div style="margin-top:.6rem">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            x,y = st.columns(2)
            x.metric("Queue-to-Doctor Ratio", f"{occ:.2f}")
            y.metric("Doctors on Duty", f"{doctors}")
        else:
            st.info("Complete the form and click **Predict Waiting Time** to generate an estimate.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- MODEL PERFORMANCE ----------
elif nav == "Model Performance":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Model Evidence</div>
      <h1>Performance & Model Selection</h1>
      <p>Review the candidate models and the held-out test results used for prototype selection.</p>
    </div>
    """, unsafe_allow_html=True)

    perf = pd.DataFrame({
        "Model":["XGBoost","Linear Regression","Random Forest"],
        "Test MAE":[9.269,9.613,9.842],
        "Test RMSE":[11.999,11.799,12.603],
        "Test R²":[0.872,0.876,0.859]
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="section-title">Why XGBoost?</div><p style="color:#64748b">XGBoost was selected using the lowest test-set MAE as the primary practical error criterion. It achieved an average absolute error of approximately 9.27 minutes on the held-out development test set.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="section-title">Important Interpretation</div><p style="color:#64748b">These metrics describe model behaviour on the generated development dataset used for prototype work. They are not a claim of clinical performance on real Nigerian hospital patients.</p></div>', unsafe_allow_html=True)

    st.bar_chart(perf.set_index("Model")["Test MAE"])

# ---------- ABOUT ----------
elif nav == "About the Project":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Research Overview</div>
      <h1>About the Project</h1>
      <p>Development of a Predictive Model for Reducing Patient Waiting Time in Nigerian Hospitals.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <div class="section-title">Aim</div>
      <p style="color:#64748b">To develop a machine-learning predictive model for estimating patient waiting time in the General Outpatient Department of a Nigerian teaching hospital.</p>
      <hr>
      <div class="section-title">Core Method</div>
      <p style="color:#64748b">The prototype compares Linear Regression, Random Forest and XGBoost, then deploys the selected model through a Streamlit interface.</p>
      <hr>
      <div class="section-title">Inputs</div>
      <p style="color:#64748b">Day of week, patient type, payment type, queue length at arrival, number of doctors on duty, arrival hour and derived occupancy ratio.</p>
      <hr>
      <div class="section-title">Output</div>
      <p style="color:#64748b">Estimated waiting time in minutes before physician consultation.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- PROFILE ----------
elif nav == "My Profile":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Account</div>
      <h1>My Profile</h1>
      <p>View the account currently accessing the CareFlow AI portal.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")
    st.write(f"**Access mode:** {'Google sign-in' if signed_in() else 'Local preview'}")
    st.caption("Public deployment can enforce Google sign-in before dashboard access.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.caption("CareFlow AI • Python • Streamlit • XGBoost • scikit-learn")
