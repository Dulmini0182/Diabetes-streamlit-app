# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =====================
# Load Dataset
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("Healthcare-Diabetes.csv")

df = load_data()

# =====================
# Sidebar Navigation
# =====================
st.sidebar.title("Navigation")
options = ["Dataset Overview", "Visualizations", "Model Training", "Prediction"]
choice = st.sidebar.radio("Go to", options)

# =====================
# Dataset Overview
# =====================
if choice == "Dataset Overview":
    st.title("📊 Diabetes Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("Column details:")
    st.write(df.dtypes)
    st.write("Sample data:")
    st.dataframe(df.head())

    # Filtering option
    if st.checkbox("Show filtered data"):
        cols = st.multiselect("Select columns", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[cols])

# =====================
# Visualizations
# =====================
elif choice == "Visualizations":
    st.title("📈 Data Visualizations")

    # 1. Class distribution
    fig1 = px.histogram(df, x='Outcome', title="Diabetes Outcome Distribution")
    st.plotly_chart(fig1)

    # 2. Correlation heatmap
    fig2, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

    # 3. Pairplot (optional)
    if st.checkbox("Show Pairplot (slow)"):
        st.write("Generating pairplot...")
        fig3 = sns.pairplot(df, hue='Outcome')
        st.pyplot(fig3)

# =====================
# Model Training
# =====================
elif choice == "Model Training":
    st.title("🤖 Model Training & Evaluation")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    st.write("### Model Performance:")
    for name, acc in results.items():
        st.write(f"{name}: **{acc:.2f}**")

    st.success(f"Best model: {best_model_name} with accuracy {results[best_model_name]:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    fig4, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig4)

    # Save best model
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    st.info("Best model saved as model.pkl")

# =====================
# Prediction
# =====================
elif choice == "Prediction":
    st.title("🔍 Diabetes Prediction")

    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.write("Enter patient details:")

    input_data = {}
    for col in df.columns:
        if col != "Outcome":
            val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val

    if st.button("Predict"):
        features = np.array(list(input_data.values())).reshape(1, -1)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][prediction]

        st.write("Prediction:", "Diabetic" if prediction == 1 else "Non-Diabetic")
        st.write(f"Confidence: {prob*100:.2f}%")
