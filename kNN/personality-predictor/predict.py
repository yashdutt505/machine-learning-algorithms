import joblib
import numpy as np
import pandas as pd

# Load the saved model and encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Example: function to predict personality
def predict_personality(age, gender, education, introversion, sensing, thinking, judging, interest):
    """
    Predict personality type from given inputs.
    Inputs:
      - age: int
      - gender: 0 for Female, 1 for Male
      - education: int/encoded value
      - introversion, sensing, thinking, judging: int scores
      - interest: one of ['Arts', 'Sports', 'Technology', 'Others', 'Unknown']
    """

    # Create DataFrame for consistency with training columns
    input_dict = {
        "Age": [age],
        "Gender": [gender],
        "Education": [education],
        "Introversion Score": [introversion],
        "Sensing Score": [sensing],
        "Thinking Score": [thinking],
        "Judging Score": [judging],
        "Interest_Arts": [1 if interest == "Arts" else 0],
        "Interest_Sports": [1 if interest == "Sports" else 0],
        "Interest_Technology": [1 if interest == "Technology" else 0],
        "Interest_Others": [1 if interest == "Others" else 0],
        "Interest_Unknown": [1 if interest == "Unknown" else 0],
    }

    input_df = pd.DataFrame(input_dict)

    # Prediction
    prediction_encoded = model.predict(input_df)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    return prediction


if __name__ == "__main__":
    # Example run
    result = predict_personality(
        age=21,
        gender=1,
        education=2,
        introversion=70,
        sensing=55,
        thinking=60,
        judging=45,
        interest="Technology"
    )
    print("Predicted Personality Type:", result)
