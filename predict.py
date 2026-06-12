import joblib
import pandas as pd

def predict(inputs: dict) -> dict:
    model = joblib.load('./models/final_model.pkl')
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = max(probability) * 100

    if prediction == 1:
        signal = "Buy"
        advice = "Model suggests upward movement tomorrow. Consider entering a position."
    else:
        signal = "Sell / Stay Out"
        advice = "Model suggests downward movement tomorrow. Consider avoiding or exiting."

    if confidence >= 70:
        strength = "Strong Signal"
    elif confidence >= 55:
        strength = "Moderate Signal"
    else:
        strength = "Weak Signal"

    return {
        'signal'    : signal,
        'confidence': f"{confidence:.1f}%",
        'strength'  : strength,
        'advice'    : advice
    }