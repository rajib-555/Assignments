import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load model and sample data
model = joblib.load("house_price_model.pkl")
X_sample = pd.read_csv("train.csv")

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")
st.markdown("Enter property details to get an estimated house price.")

# Function to get user input
def user_input_features():
    st.sidebar.header("Input Features")
    data = {
        'OverallQual': st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5),
        'GrLivArea': st.sidebar.number_input("Above Grade Living Area (sq ft)", 500, 5000, 1500),
        'GarageCars': st.sidebar.slider("Garage Capacity (Cars)", 0, 4, 2),
        'TotalBsmtSF': st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800),
        'FullBath': st.sidebar.slider("Full Bathrooms", 0, 4, 2),
        'YearBuilt': st.sidebar.slider("Year Built", 1870, 2023, 2000),
        'Neighborhood': st.sidebar.selectbox("Neighborhood", ['CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst','NWAmes','OldTown','BrkSide','Sawyer','NAmes','SawyerW','IDOTRR','MeadowV']),
        'HouseStyle': st.sidebar.selectbox("House Style", ['2Story','1Story','1.5Fin','SLvl','SFoyer','2.5Unf','2.5Fin'])
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Create full model input by copying a sample row ---
default_row = X_sample.drop(columns=['SalePrice'], errors='ignore').iloc[0].copy()

for col in input_df.columns:
    default_row[col] = input_df[col].values[0]

model_input = pd.DataFrame([default_row])

# --- Predict and display ---
if st.button("Predict Sale Price"):
    prediction = model.predict(model_input)
    st.subheader("🏷️ Predicted Sale Price:")
    st.success(f"${prediction[0]:,.2f}")

# --- Advanced Visualization Section ---
st.markdown("---")
st.header("📊 Model Insights and Visualizations")

# --- Feature Importance ---
with st.expander("🔍 Feature Importance"):
    st.write("These are the top 10 features the model considers most important for price prediction.")
    try:
        rf_model = model.named_steps['model']
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Top 10 Important Features", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Feature importance not available. Model may not be a tree-based regressor.")

# --- Price Distribution ---
with st.expander("💰 Price Distribution (from training data)"):
    st.write("Distribution of house prices in the training dataset.")
    if 'SalePrice' in X_sample.columns:
        fig2 = px.histogram(X_sample, x="SalePrice", nbins=40,
                            title="Sale Price Distribution", color_discrete_sequence=["#33A1FD"])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("SalePrice column not found in training data.")

st.caption("Built with Streamlit, RandomForest, and Plotly 📈")
