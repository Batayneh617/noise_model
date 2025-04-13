from file_inference import predict_audio_class
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np

class_labels = {0: 'no_noise', 1: 'mic_noise', 2: 'Wind'}

mel_cols = [f'mel_spec_{i}' for i in range(1, 33)]
non_mel_cols = [
    'rms', 'zcr', 'contrast_7',
    'ACT', 'EVN', 'BI'
]
feature_cols = mel_cols + non_mel_cols  

SEGMENT_FRAMES = 196

sample_path = Path('./sample_files')

    # === SHAPE CONFIG ===
pad_size = (SEGMENT_FRAMES, len(feature_cols)) 

# === LOAD MODEL AND SCALER ===
model = load_model("most_stable.h5")  
scaler = joblib.load("most_stable_model_scaler.pkl")

data = []

for file in Path.rglob(sample_path, '*.wav'):
    print(f"Processing {file.name}...")
    predicted_class_name, prediction = predict_audio_class(
        model=model,
        wav_file=str(file),
        scaler=scaler,
        selected_features=feature_cols,
        segment_frames=SEGMENT_FRAMES
    )
    if predicted_class_name is not None:
        data.append({
            'file_path': str(file),
            'file_name': str(file.name),
            'original_class': str(file.parent.name),
            'predicted_class': predicted_class_name,
            'prediction_probabilities': prediction.tolist(),
            'predicted_class_idx': np.argmax(prediction, axis=-1)[0],
        })

data = pd.DataFrame(data)
data.to_csv('predictions.csv', index=False)

