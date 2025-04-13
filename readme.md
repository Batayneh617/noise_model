# 🎧 Audio Classification Pipeline Documentation

## 🗂 File Usage

### `file_inference.py`
- Modify the `wav_file` variable to point to the `.wav` file you wish to classify.
- Example:
```python
wav_file = "path/to/your_audio.wav"
```

### `evaluate.py`
- Pass the directory containing subdirectories of `.wav` files categorized by class labels.
- Sample folder structure:
```
sample_files/
    class1/
        file1.wav
        file2.wav
    class2/
        file3.wav
        file4.wav
    class3/
        file5.wav
        file6.wav
```

---

## 🧠 Model Architecture
- Activation Function: `tanh` (replacing traditional `ReLU`)
- Total Parameters: **286,147**  
- Model Size: **1.09 MB**  
- Previously: **2,887,299 parameters** (11.01 MB)

---

## 📊 Dataset
- Added additional recordings under the `very_low` category to enhance robustness.

---

## 🏋️ Training Setup
- Training Duration: **40 epochs** (initial 20 + 20 additional epochs)
- Callback: `ReduceLROnPlateau` enabled for dynamic learning rate adjustment

---

## ✅ Results
- **Accuracy:** `0.969`
- **Precision:** `0.9710`
- **Recall:** `0.9695`
- **AUC Score:** `0.9989`

---

## 📝 Notes
- The model is sensitive to wind-related noise.
- To mitigate false positives in `low wind` conditions, consider removing frequencies below **~800 Hz**.
- This step may be feasible if the **multi-class larvae classifier** allows it.

---

## 📂 Author & Version
- Author: `Mohammad Batayneh & Besan Musallam`
- Version: `v1.0`
- Last Updated: `April 2025`

---

## 📌 Recommendations
- Apply bandpass filtering during preprocessing to suppress wind sensitivity.
- Evaluate multi-class classifier sensitivity with and without low-frequency suppression.
