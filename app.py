# =====================
# Prediction (New UI)
# =====================
elif choice == "Prediction":
    st.title("🔍 Diabetes Prediction")
    st.write("Fill in the details below to predict diabetes risk.")

    # Load saved best model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Organize input fields into two columns
    col1, col2 = st.columns(2)

    input_data = {}
    for i, col in enumerate(df.columns):
        if col != "Outcome":
            if i % 2 == 0:
                val = col1.number_input(
                    f"{col}",
                    float(df[col].min()), 
                    float(df[col].max()), 
                    float(df[col].mean())
                )
            else:
                val = col2.number_input(
                    f"{col}",
                    float(df[col].min()), 
                    float(df[col].max()), 
                    float(df[col].mean())
                )
            input_data[col] = val

    # Add a styled Predict button
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Now", use_container_width=True)

    if predict_btn:
        features = np.array(list(input_data.values())).reshape(1, -1)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][prediction] * 100

        # Show results in a nice card
        if prediction == 1:
            st.error(f"⚠️ **Prediction:** Diabetic\n\n**Confidence:** {prob:.2f}%")
        else:
            st.success(f"✅ **Prediction:** Non-Diabetic\n\n**Confidence:** {prob:.2f}%")

