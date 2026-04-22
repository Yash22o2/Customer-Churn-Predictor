





import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION  (must be first streamlit command)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  LOAD MODEL & SCALER  (cached so it loads only once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────────────────────────
#  FEATURE COLUMNS  (must match exactly what model was trained on)
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges',
    'TechSupport',
    'ContractType_Month-to-Month', 'ContractType_One-Year', 'ContractType_Two-Year',
    'InternetService_DSL', 'InternetService_Fiber Optic', 'InternetService_No Service',
    'AvgMonthlySpend', 'ChargeDeviation'
]
NUM_COLS = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeDeviation']

# ─────────────────────────────────────────────────────────────
#  HELPER — build one row of features from user inputs
# ─────────────────────────────────────────────────────────────
def build_features(age, gender, tenure, monthly_charges,
                   total_charges, tech_support, contract, internet):

    # Engineered features (same logic as preprocessing)
    avg_monthly = total_charges / tenure if tenure > 0 else monthly_charges
    charge_dev  = abs(monthly_charges - avg_monthly)

    row = {
        'Age'            : age,
        'Gender'         : 1 if gender == 'Male' else 0, # LabelEncoder: Female=0, Male=1
        'Tenure'         : tenure,
        'MonthlyCharges' : monthly_charges,
        'TotalCharges'   : total_charges,
        'TechSupport'    : 1 if tech_support == 'Yes' else 0,
        # One-hot: ContractType
        'ContractType_Month-to-Month' : 1 if contract == 'Month-to-Month' else 0,
        'ContractType_One-Year'       : 1 if contract == 'One-Year'       else 0,
        'ContractType_Two-Year'       : 1 if contract == 'Two-Year'       else 0,
        # One-hot: InternetService
        'InternetService_DSL'          : 1 if internet == 'DSL'          else 0,
        'InternetService_Fiber Optic'  : 1 if internet == 'Fiber Optic'  else 0,
        'InternetService_No Service'   : 1 if internet == 'No Service'   else 0,
        # Engineered
        'AvgMonthlySpend' : avg_monthly,
        'ChargeDeviation' : charge_dev,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLS)

    # Scale numerical columns (same scaler from training)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    return df

# ─────────────────────────────────────────────────────────────
#  HELPER — gauge chart
# ─────────────────────────────────────────────────────────────
def draw_gauge(probability):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={'aspect': 'equal'})
    fig.patch.set_alpha(0)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')

    # Background arc (gray)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#e0e0e0', linewidth=18, solid_capstyle='round')

    # Colored fill arc
    fill_theta = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    color = '#2ECC71' if probability < 0.4 else '#F39C12' if probability < 0.7 else '#E74C3C'
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=18, solid_capstyle='round')

    # Needle
    angle = np.pi - probability * np.pi
    ax.annotate('', xy=(0.6 * np.cos(angle), 0.6 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.plot(0, 0, 'o', color='#2c3e50', markersize=8)

    # Labels
    ax.text(-1.1, -0.12, '0%',  ha='center', fontsize=9, color='#888')
    ax.text( 1.1, -0.12, '100%', ha='center', fontsize=9, color='#888')
    ax.text(0, -0.22, f'{probability*100:.1f}%', ha='center',
            fontsize=22, fontweight='bold', color=color)

    plt.tight_layout(pad=0)
    return fig

# ─────────────────────────────────────────────────────────────
#  HELPER — feature importance bar (top 6)
# ─────────────────────────────────────────────────────────────
# def draw_feature_importance():
#     if hasattr(model, 'feature_importances_'):
#         importances = pd.Series(
#             model.feature_importances_, index=FEATURE_COLS
#         ).sort_values(ascending=True).tail(6)

#         fig, ax = plt.subplots(figsize=(5, 3))
#         fig.patch.set_alpha(0)
#         ax.set_facecolor('none')
#         colors = ['#3498DB' if v < importances.quantile(0.6) else '#E74C3C'
#                   for v in importances]
#         importances.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
#         ax.set_xlabel('Importance', fontsize=9)
#         ax.set_title('Top 6 churn drivers', fontsize=10)
#         ax.tick_params(labelsize=8)
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         plt.tight_layout()
#         return fig
#     else:
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.text(0.5, 0.5, 'Feature importances not available', ha='center', va='center')
#         ax.axis('off')
#         return fig

# ─────────────────────────────────────────────────────────────
#  HELPER — feature importance bar (top 6)
# ─────────────────────────────────────────────────────────────
def draw_feature_importance():
    # 1. Safely locate feature importances depending on how the model was saved
    importance_values = None
    
    if hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
        # Handles if you saved a GridSearchCV object instead of a base model
        importance_values = model.best_estimator_.feature_importances_
    elif hasattr(model, 'coef_'):
        # Handles if you saved a Logistic Regression model
        importance_values = np.abs(model.coef_[0])
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'coef_'):
        # Handles GridSearchCV with Logistic Regression
        importance_values = np.abs(model.best_estimator_.coef_[0])

    # 2. Draw the chart safely
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    if importance_values is not None:
        importances = pd.Series(
            importance_values, index=FEATURE_COLS
        ).sort_values(ascending=True).tail(6)

        # Color the top driver red, others blue
        colors = ['#3498DB' if v < importances.quantile(0.6) else '#E74C3C' for v in importances]
        importances.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
        
        # Use 'gray' color so text is visible in BOTH Light and Dark themes!
        ax.set_xlabel('Importance / Weight', fontsize=9, color='gray')
        ax.set_title('Top 6 churn drivers', fontsize=10, color='gray')
        ax.tick_params(labelsize=8, colors='gray') 
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        # Fallback if the model type doesn't support importances at all
        ax.text(0.5, 0.5, 'Feature importances\nnot available for this model', 
                ha='center', va='center', color='gray', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — customer input form
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Customer Details")
    st.markdown("Fill in the customer information below.")
    st.divider()

    st.markdown("**Demographics**")
    age    = st.slider("Age", min_value=18, max_value=90, value=40, step=1)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    st.divider()
    st.markdown("**Account Info**")
    tenure          = st.slider("Tenure (months)", 0, 130, 12, step=1,
                                help="How long the customer has been with us")
    contract        = st.selectbox("Contract Type",
                                   ["Month-to-Month", "One-Year", "Two-Year"])
    tech_support    = st.radio("Tech Support", ["Yes", "No"], horizontal=True)
    internet        = st.selectbox("Internet Service",
                                   ["Fiber Optic", "DSL", "No Service"])

    st.divider()
    st.markdown("**Billing**")
    monthly_charges = st.slider("Monthly Charges ($)", 20.0, 130.0, 70.0, step=0.5)

    # Auto-calculate total charges as a sensible default
    default_total   = round(monthly_charges * max(tenure, 1), 2)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0,
                                      max_value=15000.0,
                                      value=float(default_total), step=10.0,
                                      help="Total amount billed to date")

    st.divider()
    predict_btn = st.button("🔍  Predict Churn", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
#  MAIN PAGE — Header
# ─────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.markdown(
    "This app predicts whether a customer is likely to **cancel their subscription** "
    "using a **Logistic Regression** model trained on customer behaviour data."
)
st.divider()

# ─────────────────────────────────────────────────────────────
#  MAIN PAGE — Default state (before prediction)
# ─────────────────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns([1.5,1,1])
    col1.metric("Model", "Logistic  Regression")
    col2.metric("Test F1-Score", "0.825") # Replaced 100% Training Accuracy (overfitting fix)
    col3.metric("Features Used", "14")

    st.markdown("### How to use this app")
    st.markdown("""
1. Fill in the **customer details** in the left sidebar
2. Click **Predict Churn**
3. See the churn probability, risk level, and what's driving the prediction
""")

    st.info(
        "**What is Churn?** "
        "Churn means a customer stops using the service. "
        "Predicting churn early lets the business offer discounts or support "
        "to retain at-risk customers — saving revenue."
    )

    with st.expander("📚 How does the model work? (click to read)"):
        st.markdown("""
**Step 1 — Data:** We used customer records with details like age, tenure, contract type, and billing. Added 12% label noise to simulate real-world errors.

**Step 2 — Preprocessing:** Cleaned missing values, encoded text columns to numbers, scaled numerical features, and engineered `AvgMonthlySpend`.

**Step 3 — Training:** A Random Forest trains 100 decision trees. We restricted tree depth to 8 to prevent overfitting and memorising the training data. Each tree votes on whether a customer churns.
The majority vote becomes the final prediction.

**Step 4 — Probability:** The fraction of trees that voted "churn" = the churn probability shown on the gauge.
        """)

    with st.expander("⚠️ About the class imbalance & Training"):
        st.markdown("""
Our dataset originally had **88.3% churn** and only **11.7% non-churn** customers.

This means a dumb model that always predicts "churn" would get 88% accuracy — but it's useless.

**What we did:**
- Used **SMOTE Oversampling** strictly on the training set to create synthetic non-churn examples and balance the classes.
- Used `class_weight='balanced'` so the model pays extra attention to minority class errors.
- Limited the Random Forest structure (`max_depth=8`, `min_samples_leaf=5`) to fix training data memorisation.
- Evaluated the model purely on a separate Test set using **F1-score**, **ROC-AUC**, and **PR-AUC** instead of accuracy.
        """)

# ─────────────────────────────────────────────────────────────
#  MAIN PAGE — Prediction result
# ─────────────────────────────────────────────────────────────
else:
    # Build feature row and predict
    X_input      = build_features(age, gender, tenure, monthly_charges,
                                  total_charges, tech_support, contract, internet)
    prediction   = model.predict(X_input)[0]
    probability  = model.predict_proba(X_input)[0][1]  # probability of churn

    # Risk level
    if probability < 0.4:
        risk_label = "🟢 Low Risk"
        risk_color = "normal"
        action     = "Customer is stable. Standard engagement recommended."
    elif probability < 0.7:
        risk_label = "🟡 Medium Risk"
        risk_color = "off"
        action     = "Consider a personalised offer or check-in call."
    else:
        risk_label = "🔴 High Risk"
        risk_color = "inverse"
        action     = "Immediate action needed — offer discount or escalate to retention team."

    # ── Row 1: key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction",      "Will Churn " if prediction == 1 else "Will Stay ✓")
    c2.metric("Churn Probability", f"{probability*100:.1f}%")
    c3.metric("Risk Level",      risk_label)
    c4.metric("Contract",        contract)

    st.divider()

    # ── Row 2: gauge + details + feature importance
    col_gauge, col_details, col_feat = st.columns([1.2, 1.5, 1.5])

    with col_gauge:
        st.markdown("#### Churn probability")
        st.pyplot(draw_gauge(probability), use_container_width=True)
        if prediction == 1:
            st.error(f"**Likely to churn**")
        else:
            st.success(f"**Likely to stay**")

    with col_details:
        st.markdown("#### Customer summary")
        details = {
            "Age"             : age,
            "Gender"          : gender,
            "Tenure"          : f"{tenure} months",
            "Monthly Charges" : f"${monthly_charges:.2f}",
            "Total Charges"   : f"${total_charges:.2f}",
            "Contract"        : contract,
            "Internet"        : internet,
            "Tech Support"    : tech_support,
        }
        for k, v in details.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #f0f0f0;font-size:14px'>"
                f"<span style='color:#888'>{k}</span><span><b>{v}</b></span></div>",
                unsafe_allow_html=True
            )

    with col_feat:
        st.markdown("#### Top churn drivers")
        st.pyplot(draw_feature_importance(), use_container_width=True)

    st.divider()

    # ── Row 3: recommended action
    st.markdown("#### 💡 Recommended Action")
    if probability >= 0.7:
        st.error(f"**{action}**")
    elif probability >= 0.4:
        st.warning(f"**{action}**")
    else:
        st.success(f"**{action}**")

    # ── Row 4: what's driving this prediction
    st.markdown("#### 🔎 What's influencing this prediction?")
    insights = []
    if contract == "Month-to-Month":
        insights.append("📋 **Month-to-Month contract** — highest churn risk contract type. Customers with no lock-in leave more easily.")
    elif contract == "Two-Year":
        insights.append("📋 **Two-Year contract** — lowest churn risk. Long-term commitment reduces likelihood of leaving.")
    if tenure < 6:
        insights.append("⏱️ **Very new customer (< 6 months)** — new customers are more likely to churn before becoming loyal.")
    elif tenure > 24:
        insights.append("⏱️ **Long-tenure customer (> 24 months)** — loyalty is a strong retention signal.")
    if monthly_charges > 90:
        insights.append("💸 **High monthly charges** — customers paying more are more likely to look for cheaper alternatives.")
    if internet == "Fiber Optic":
        insights.append("🌐 **Fiber Optic internet** — associated with higher churn in the training data (likely due to higher cost).")
    if tech_support == "No":
        insights.append("🛠️ **No tech support** — customers without support are more likely to leave when they face problems.")

    if insights:
        for i in insights:
            st.markdown(f"- {i}")
    else:
        st.markdown("- No strong single-factor signals — prediction is based on a combination of all features.")

    # ── Row 5: raw feature values (expandable)
    with st.expander("🧮 See raw feature values sent to model"):
        display_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Raw Value (before scaling)': [
                age, 1 if gender=='Male' else 0, tenure, monthly_charges,
                total_charges, 1 if tech_support=='Yes' else 0,
                1 if contract=='Month-to-Month' else 0,
                1 if contract=='One-Year' else 0,
                1 if contract=='Two-Year' else 0,
                1 if internet=='DSL' else 0,
                1 if internet=='Fiber Optic' else 0,
                1 if internet=='No Service' else 0,
                round(total_charges/tenure if tenure>0 else monthly_charges, 2),
                round(abs(monthly_charges - (total_charges/tenure if tenure>0 else monthly_charges)), 2),
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:12px'>"
    "Customer Churn Prediction App · Built with Streamlit + scikit-learn · "
    "Logistic Regression (Trained with SMOTE)"
    "</div>",
    unsafe_allow_html=True
)