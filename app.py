import streamlit as st
import pandas as pd

st.title("Risk Score Feedback App")

# Sample data
data = pd.DataFrame({
    "user_id": [1, 2, 3],
    "activity": ["File Delete", "USB Access", "Login"],
    "predicted_score": [80, 40, 20]
})

# Show table
st.subheader("Risk Dashboard")
st.dataframe(data)

# Feedback input
st.subheader("Provide Feedback")
feedback = []

for i in range(len(data)):
    score = st.slider(f"User {data['user_id'][i]} feedback", 0, 100)
    feedback.append(score)

# Button
if st.button("Retrain Model"):
    st.success("Model retrained (simulated)")
    data["updated_score"] = feedback
    st.write(data)
