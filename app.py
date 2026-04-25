import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("AI Risk Scoring with Feedback Loop")

st.markdown("""
### Flow:
Input Features → Initial Prediction → User Feedback → Retraining → Updated Prediction
""")

data = pd.DataFrame({
    "user_id": [1,2,3,4,5],
    "suspicious_activity": [0.9, 0.6, 0.2, 0.8, 0.5],
    "file_access_freq": [0.8, 0.5, 0.3, 0.7, 0.4],
    "off_hours_access": [1, 1, 0, 1, 0],
    "expected_score": [0.95, 0.60, 0.25, 0.90, 0.55]
})

st.info("Expected score represents benchmark (ground truth) for evaluation")

def predict(features, weights):
    return sum(w * x for w, x in zip(weights, features))

weights = [0.4, 0.3, 0.3]

st.subheader("Model Explainability")
st.write("Initial Weights:", weights)


predictions = []
for i, row in data.iterrows():
    features = [
        row['suspicious_activity'],
        row['file_access_freq'],
        row['off_hours_access']
    ]
    pred = predict(features, weights)
    predictions.append(pred)

data["predicted_score"] = predictions


st.subheader("User Feedback Input")

feedback_list = []

for i, row in data.iterrows():
    fb = st.slider(
        f"User {row['user_id']} Feedback",
        0.0, 1.0,
        float(row["predicted_score"])
    )
    feedback_list.append(fb)

def apply_feedback(ai_score, user_score, alpha=0.6):
    return ai_score + alpha * (user_score - ai_score)

def retrain(weights, X, feedbacks, lr=0.01, lambda_=5):
    for i in range(len(X)):
        pred = sum(w * x for w, x in zip(weights, X[i]))
        error = pred - feedbacks[i]

        weights = [
            w - lr * lambda_ * error * x
            for w, x in zip(weights, X[i])
        ]
    return weights


if st.button("Retrain Model"):

    X = data[[
        "suspicious_activity",
        "file_access_freq",
        "off_hours_access"
    ]].values

    new_weights = retrain(weights, X, feedback_list)

    st.write("Updated Weights:", new_weights)

    updated_scores = []

    for i, row in data.iterrows():
        features = [
            row['suspicious_activity'],
            row['file_access_freq'],
            row['off_hours_access']
        ]

        ai_pred = predict(features, new_weights)

        # Apply AIRS feedback adjustment
        updated = apply_feedback(ai_pred, feedback_list[i])

        updated_scores.append(updated)

    data["updated_score"] = updated_scores

    st.success("Model Retrained Successfully!")

st.subheader("Model Output Comparison")

if "updated_score" in data:

    data["original_error"] = abs(data["predicted_score"] - data["expected_score"])
    data["updated_error"] = abs(data["updated_score"] - data["expected_score"])

    data["improved"] = data["updated_error"] < data["original_error"]

    # Table
    st.dataframe(data[[
        "user_id",
        "predicted_score",
        "expected_score",
        "updated_score",
        "original_error",
        "updated_error",
        "improved"
    ]])

    # Scores
    st.write("Original Scores:", data["predicted_score"].tolist())
    st.write("Updated Scores:", data["updated_score"].tolist())

    # Error Comparison
    st.subheader("Error Comparison")

    st.write("Average Original Error:", data["original_error"].mean())
    st.write("Average Updated Error:", data["updated_error"].mean())

    # Graph
    st.subheader("Error Improvement Graph")

    plt.figure()
    plt.plot(data["original_error"], label="Original Error")
    plt.plot(data["updated_error"], label="Updated Error")
    plt.legend()
    st.pyplot(plt)

else:
    st.write("Click 'Retrain Model' to see updated results.")
