import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Title
# -------------------------------
st.title("AI Risk Scoring with Feedback Loop")

st.markdown("""
### Flow:
Input Features → Initial Prediction → User Feedback → Retraining → Updated Prediction
""")

# -------------------------------
# Dataset (Synthetic)
# -------------------------------
data = pd.DataFrame({
    "user_id": [1,2,3,4,5],
    "suspicious_activity": [0.9, 0.6, 0.2, 0.8, 0.5],
    "file_access_freq": [0.8, 0.5, 0.3, 0.7, 0.4],
    "off_hours_access": [1, 1, 0, 1, 0],
    "expected_score": [0.95, 0.60, 0.25, 0.90, 0.55]
})

st.info("Expected score represents benchmark (ground truth) for evaluation")

# -------------------------------
# Baseline Model
# -------------------------------
def predict(features, weights):
    return sum(w * x for w, x in zip(weights, features))

weights = [0.4, 0.3, 0.3]

st.subheader("Initial Model Weights")
st.write(weights)

# -------------------------------
# Initial Prediction
# -------------------------------
predictions = []
for _, row in data.iterrows():
    features = [
        row['suspicious_activity'],
        row['file_access_freq'],
        row['off_hours_access']
    ]
    predictions.append(predict(features, weights))

data["predicted_score"] = predictions

# 👉 SHOW TABLE (you were missing this)
st.subheader("Risk Score Dashboard")
st.dataframe(data)

# -------------------------------
# Feedback Input
# -------------------------------
st.subheader("User Feedback Input")

feedback_list = []

for i, row in data.iterrows():
    fb = st.slider(
        f"User {row['user_id']} Feedback",
        0.0, 1.0,
        float(row["predicted_score"]),
        key=f"user_{i}"
    )
    feedback_list.append(fb)

# -------------------------------
# AIRS Formula
# -------------------------------
def apply_feedback(ai_score, user_score, alpha=0.6):
    return ai_score + alpha * (user_score - ai_score)

# -------------------------------
# Retraining
# -------------------------------
def retrain(weights, X, feedbacks, lr=0.01, lambda_=5):
    new_weights = weights.copy()
    for i in range(len(X)):
        pred = sum(w * x for w, x in zip(new_weights, X[i]))
        error = pred - feedbacks[i]

        new_weights = [
            w - lr * lambda_ * error * x
            for w, x in zip(new_weights, X[i])
        ]
    return new_weights

# -------------------------------
# Retrain Button
# -------------------------------
if st.button("Retrain Model"):

    X = data[[
        "suspicious_activity",
        "file_access_freq",
        "off_hours_access"
    ]].values

    new_weights = retrain(weights, X, feedback_list)

    st.subheader("Updated Weights")
    st.write(new_weights)

    updated_scores = []

    for i, row in data.iterrows():
        features = [
            row['suspicious_activity'],
            row['file_access_freq'],
            row['off_hours_access']
        ]

        ai_pred = predict(features, new_weights)
        updated = apply_feedback(ai_pred, feedback_list[i])
        updated_scores.append(updated)

    data["updated_score"] = updated_scores

    st.success("Model Retrained Successfully!")

    # -------------------------------
    # Results
    # -------------------------------
    st.subheader("Model Output Comparison")

    data["original_error"] = abs(data["predicted_score"] - data["expected_score"])
    data["updated_error"] = abs(data["updated_score"] - data["expected_score"])

    data["improved"] = data["updated_error"] < data["original_error"]

    st.dataframe(data)

    # -------------------------------
    # Graph
    # -------------------------------
    st.subheader("Error Improvement Graph")

    fig, ax = plt.subplots()
    ax.plot(data["original_error"], label="Original Error")
    ax.plot(data["updated_error"], label="Updated Error")
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Click 'Retrain Model' to see updated results.")
