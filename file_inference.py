from extract_features import extract_features
import numpy as np
import joblib
from tensorflow.keras.models import load_model

class_labels = {0: 'no_noise', 1: 'mic_noise', 2: 'Wind'}

mel_cols = [f'mel_spec_{i}' for i in range(1, 33)]
non_mel_cols = [
    'rms', 'zcr', 'contrast_7',
    'ACT', 'EVN', 'BI'
]
feature_cols = mel_cols + non_mel_cols  

SEGMENT_FRAMES = 196
def predict_audio_class(model, wav_file, scaler, selected_features, segment_frames=SEGMENT_FRAMES):
    # Step 1: Extract selected features from the .wav file
    features = extract_features(wav_file, selected_features=selected_features)

    if features is None or features.shape[0] < segment_frames:
        print(f"❌ Not enough frames in {wav_file}. Got {features.shape[0]} but need {segment_frames}")
        return None, None

    # Step 2: Trim or pad features
    if features.shape[0] > segment_frames:
        features = features[:segment_frames, :]
    elif features.shape[0] < segment_frames:
        pad_len = segment_frames - features.shape[0]
        features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')

    # Step 3: Normalize and reshape
    features_flat = features.reshape(1, -1)
    features_scaled = scaler.transform(features_flat).reshape(1, segment_frames, len(selected_features), 1)

    # Step 4: Predict
    prediction = model.predict(features_scaled)
    predicted_class_idx = np.argmax(prediction, axis=-1)[0]
    predicted_class_name = class_labels[predicted_class_idx]

    return predicted_class_name, prediction




if __name__ == "__main__":
    # === SHAPE CONFIG ===
    pad_size = (SEGMENT_FRAMES, len(feature_cols)) 

    # === LOAD MODEL AND SCALER ===
    model = load_model("most_stable.h5")  
    scaler = joblib.load("most_stable_model_scaler.pkl")
    # === AUDIO PREDICTION ===

    wav_file = "/mnt/d/clean/High/Copy of FN_181124_0827_KeJqO3xsAMbdA6t3aDCT.wav"
    predicted_class_name, prediction = predict_audio_class(
        model=model,
        wav_file=wav_file,
        scaler=scaler,
        selected_features=feature_cols,
        segment_frames=SEGMENT_FRAMES
    )

    # === OUTPUT ===
    if predicted_class_name:
        print(f"🎯 Predicted Class: {predicted_class_name}")
        print(f"📊 Prediction Probabilities: {prediction}")