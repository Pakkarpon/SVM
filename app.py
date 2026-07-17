import streamlit as st
import pandas as pd
import pickle

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Smart Credit Evaluator",
    page_icon="💳",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
with open("loan_svm_model.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# CSS
# ==========================
st.markdown("""
<style>
.main {
    background-color: #FAFAFA;
}

.title {
    text-align: center;
    color: #059669; /* สีเขียวมรกต ดูเป็นทางการและเกี่ยวกับการเงิน */
    font-size: 45px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    color: #4B5563;
    font-size: 18px;
    font-style: italic;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown("<div class='title'>💳 Smart Credit Evaluator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Loan Assessment System (SVM Model)</div>", unsafe_allow_html=True)

# ==========================
# Sidebar (ปรับ Emoji นิดหน่อย)
# ==========================
st.sidebar.title("⚙️ Applicant Details")

st.sidebar.subheader("👤 Personal Info")
person_age = st.sidebar.number_input("Age", 18, 100, 25)
person_gender = st.sidebar.selectbox("Gender", ["male", "female"])
person_education = st.sidebar.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.sidebar.number_input("Income", min_value=0, value=50000)
person_emp_exp = st.sidebar.number_input("Employment Experience (Years)", min_value=0, value=2)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

st.sidebar.subheader("🏦 Loan Request")
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0, value=10000)
loan_intent = st.sidebar.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", value=12.5)
loan_percent_income = st.sidebar.number_input("Loan Percent Income", value=0.25)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", min_value=0, value=3)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=700)
previous_loan_defaults_on_file = st.sidebar.selectbox("Previous Loan Default", ["Yes", "No"])

predict = st.sidebar.button("✨ Run Prediction", use_container_width=True)

# ==========================
# Main Page (เปลี่ยนมาใช้ Tabs แทน Columns)
# ==========================

tab1, tab2 = st.tabs(["📋 Data Overview", "📊 AI Prediction Results"])

with tab1:
    st.markdown("### 📝 Summary of Provided Information")
    st.dataframe(pd.DataFrame({
        "Feature": [
            "Age", "Gender", "Education", "Income", "Employment Exp", 
            "Home Ownership", "Loan Amount", "Loan Intent", "Interest Rate", 
            "Loan Percent Income", "Credit History Length", "Credit Score", "Previous Default"
        ],
        "Value": [
            person_age, person_gender, person_education, f"${person_income:,}", 
            f"{person_emp_exp} Years", person_home_ownership, f"${loan_amnt:,}", 
            loan_intent, f"{loan_int_rate}%", loan_percent_income, 
            f"{cb_person_cred_hist_length} Years", credit_score, previous_loan_defaults_on_file
        ]
    }), use_container_width=True)

with tab2:
    if predict:
        st.markdown("### 🎯 System Decision")
        
        input_df = pd.DataFrame({
            "person_age": [person_age],
            "person_gender": [person_gender],
            "person_education": [person_education],
            "person_income": [person_income],
            "person_emp_exp": [person_emp_exp],
            "person_home_ownership": [person_home_ownership],
            "loan_amnt": [loan_amnt],
            "loan_intent": [loan_intent],
            "loan_int_rate": [loan_int_rate],
            "loan_percent_income": [loan_percent_income],
            "cb_person_cred_hist_length": [cb_person_cred_hist_length],
            "credit_score": [credit_score],
            "previous_loan_defaults_on_file": [previous_loan_defaults_on_file]
        })

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        confidence = max(probability[0])

        # ปรับการแสดงผลให้ดูมีลูกเล่นขึ้น
        if prediction[0] == 1:
            st.success(f"✅ **APPROVED** — The model predicts this loan should be approved.")
        else:
            st.error(f"❌ **REJECTED** — The model predicts this loan should be rejected.")

        st.metric("Model Confidence Level", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

        st.write("---")
        st.markdown("#### 🔍 Probability Breakdown")
        prob_df = pd.DataFrame(probability, columns=model.classes_)
        st.dataframe(prob_df, use_container_width=True)

    else:
        st.info("👈 กรุณากรอกข้อมูลทางแถบด้านซ้าย แล้วกดปุ่ม 'Run Prediction' เพื่อดูผลลัพธ์")

# ==========================
# Footer
# ==========================
st.write("")
st.write("")
st.markdown("---")
st.markdown(
    "<center>🎓 Academic Project | 🚀 Powered by Streamlit & Support Vector Machine</center>",
    unsafe_allow_html=True
)