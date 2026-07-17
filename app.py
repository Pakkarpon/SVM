import streamlit as st
import pandas as pd
import pickle

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Loan Prediction using SVM",
    page_icon="💰",
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
    background-color: #f5f7fa;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result-box{
    padding:25px;
    border-radius:15px;
    background:#ffffff;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown("<div class='title'>💰 Loan Prediction using SVM</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Machine Learning Project | Support Vector Machine</div>", unsafe_allow_html=True)

st.write("")

# ==========================
# Sidebar
# ==========================
st.sidebar.title("📋 Input Information")

st.sidebar.subheader("👤 Personal")

person_age = st.sidebar.number_input("Age",18,100,25)

person_gender = st.sidebar.selectbox(
    "Gender",
    ["male","female"]
)

person_education = st.sidebar.selectbox(
    "Education",
    ["High School","Associate","Bachelor","Master","Doctorate"]
)

person_income = st.sidebar.number_input(
    "Income",
    min_value=0,
    value=50000
)

person_emp_exp = st.sidebar.number_input(
    "Employment Experience",
    min_value=0,
    value=2
)

person_home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["RENT","OWN","MORTGAGE","OTHER"]
)

st.sidebar.subheader("💵 Loan")

loan_amnt = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_int_rate = st.sidebar.number_input(
    "Interest Rate",
    value=12.5
)

loan_percent_income = st.sidebar.number_input(
    "Loan Percent Income",
    value=0.25
)

cb_person_cred_hist_length = st.sidebar.number_input(
    "Credit History Length",
    min_value=0,
    value=3
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

previous_loan_defaults_on_file = st.sidebar.selectbox(
    "Previous Loan Default",
    ["Yes","No"]
)

predict = st.sidebar.button(
    "🚀 Predict",
    use_container_width=True
)

# ==========================
# Main Page
# ==========================

left,right = st.columns([1,1])

with left:

    st.subheader("📄 Input Summary")

    st.dataframe(pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "Education",
            "Income",
            "Employment Exp",
            "Home Ownership",
            "Loan Amount",
            "Loan Intent",
            "Interest Rate",
            "Loan Percent Income",
            "Credit History",
            "Credit Score",
            "Previous Default"
        ],

        "Value":[
            person_age,
            person_gender,
            person_education,
            person_income,
            person_emp_exp,
            person_home_ownership,
            loan_amnt,
            loan_intent,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length,
            credit_score,
            previous_loan_defaults_on_file
        ]

    }),use_container_width=True)

with right:

    st.subheader("📊 Prediction Result")

    if predict:

        input_df = pd.DataFrame({

            "person_age":[person_age],
            "person_gender":[person_gender],
            "person_education":[person_education],
            "person_income":[person_income],
            "person_emp_exp":[person_emp_exp],
            "person_home_ownership":[person_home_ownership],
            "loan_amnt":[loan_amnt],
            "loan_intent":[loan_intent],
            "loan_int_rate":[loan_int_rate],
            "loan_percent_income":[loan_percent_income],
            "cb_person_cred_hist_length":[cb_person_cred_hist_length],
            "credit_score":[credit_score],
            "previous_loan_defaults_on_file":[previous_loan_defaults_on_file]

        })

        prediction = model.predict(input_df)

        probability = model.predict_proba(input_df)

        confidence = max(probability[0])

        if prediction[0] == 1:

            st.success("✅ Loan Approved")

        else:

            st.error("❌ Loan Rejected")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(float(confidence))

        st.subheader("Probability")

        prob_df = pd.DataFrame(
            probability,
            columns=model.classes_
        )

        st.dataframe(prob_df,use_container_width=True)

    else:

        st.info("กรุณากรอกข้อมูลทางด้านซ้าย แล้วกดปุ่ม Predict")

# ==========================
# Footer
# ==========================

st.write("")
st.write("")

st.markdown("---")

st.markdown(
    "<center>💻 Developed with Streamlit | 🤖 Machine Learning using Support Vector Machine (SVM)</center>",
    unsafe_allow_html=True
)