import joblib
from sklearn.impute import SimpleImputer

MODEL_FILE = "src/Models/svc_model_affectnet.pkl"
SCALER_FILE = "src/Models/scaler_affectnet.pkl"

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

def predict_emotion(au_features) -> int:
    """
    Predict the emotion of a person in an image using the given detector and model.
    :param image_path: Path to the image file.
    :param detector: The detector object.
    :param model: The emotion classification model.
    :param scaler: The scaler object.
    :return: The predicted emotion.
    """
    emotion_index = -1
    # try:
        
    # au_features = au_features.reshape(1, -1)

    imputer = SimpleImputer(strategy="mean")
    au_features_imputed = imputer.fit_transform(au_features[0])

    au_features_scaled = scaler.transform(au_features_imputed)

    emotion_index = model.predict(au_features_scaled)[0]
    # except Exception as e:
    #     print(f"Error during detection: {e}")
    return emotion_index
