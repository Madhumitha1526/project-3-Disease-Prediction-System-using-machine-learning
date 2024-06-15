def preprocess_input(data):
    # Example preprocessing steps
    features = [
        int(data['age']),
        float(data['bmi']),
        float(data['blood_pressure']),
        int(data['cholesterol']),
        int(data['family_history'])
    ]
    return features
