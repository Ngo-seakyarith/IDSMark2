import streamlit as st

# Page Config
st.set_page_config(
    page_title="Data Science Project Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title Section
st.markdown("""
<div style="background-color:#4CAF50;padding:15px;border-radius:10px">
<h1 style="color:white;text-align:center;">Welcome to the Data Science Project Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.write("")

# Group Introduction
st.subheader("👨‍👩‍👦‍👦 **Group Members**")
team_members = [
    "📌 Sou Pichchomrong",
    "📌 Sroeun Bunnarith",
    "📌 Sorng Seyha",
    "📌 Thorn Davin",
    "📌 Ngo Seakyarith"
]
for member in team_members:
    st.write(f"- {member}")

st.divider()

# Visualizations Section
st.header("📊 **Exploratory Data Analysis**")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Correlation Heatmap")
    st.image("images/correlation_heatmap.png", caption="Correlation Heatmap", use_container_width=True)

    st.subheader("Feature Importance")
    st.image("images/feature_importance.png", caption="Feature Importance", use_container_width=True)

with col2:
    st.subheader("Pairplot")
    st.image("images/pairplot.png", caption="Pairplot of Variables", use_container_width=True)

st.divider()

# Model Training Summary
st.header("📋 **Status of Each Model**")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔹 **Linear Models**")
    st.markdown("""
    - Linear Regression ✅ (Trained)
    - Ridge Regression ✅ (Trained)
    - Lasso Regression ✅ (Trained)
    - ElasticNet Regression ✅ (Trained)
    """)

    st.subheader("🔹 **Bayesian Methods**")
    st.markdown("""
    - Bayesian Ridge Regression ✅ (Trained)
    """)

    st.subheader("🔹 **Other Specialized Models**")
    st.markdown("""
    - K-Nearest Neighbors (KNN) ✅ (Trained)
    """)

with col2:
    st.subheader("🔹 **Tree-Based Models**")
    st.markdown("""
    - Decision Tree ✅ (Trained)
    - Random Forest ✅ (Trained)
    - Gradient Boosting ✅ (Trained)
    - Extreme Gradient Boosting (XGBoost) ❌ (Not trained; requires xgboost library)
    - LightGBM ❌ (Not trained; requires lightgbm library)
    - CatBoost ❌ (Not trained; requires catboost library)
    """)

    st.subheader("🔹 **Ensemble Methods**")
    st.markdown("""
    - AdaBoost ✅ (Trained)
    - Stacking Regressor ❌ (Not trained; can be implemented if needed)
    """)

    st.subheader("🔹 **Neural Networks**")
    st.markdown("""
    - Multi-Layer Perceptron (MLP) ✅ (Trained)
    - TensorFlow/Keras models ❌ (Not implemented; requires TensorFlow library)
    """)

st.divider()

# Navigation Button
if st.button("🚀 Go to Sales Predictor"):
    st.switch_page("pages/Sales_Predictor.py")

# Footer Section
st.write("")
st.markdown("""
<div style="text-align:center;color:gray;font-size:12px;">
    Built with ❤️ by Group 9 | Powered by Streamlit
</div>
""", unsafe_allow_html=True)


