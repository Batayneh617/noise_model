
import librosa
import numpy as np
from maad import features
import pywt
from scipy.signal import chirp, hilbert

SR = 44100

# Window size (50ms)
N_FFT = round(0.05 * SR)  

# Hop length 
HOP_LENGTH = round((2.5 * SR) / 98)  

# Number of Mel bands
N_MELS = 33 



# FMIN and FMAX: The minimum and maximum frequencies used for the Mel spectrogram.
FMIN = 300      # Minimum frequency (in Hz) to consider
FMAX = 10000   # Maximum frequency (in Hz) to consider

# POWER: The exponent used for the power spectrogram (higher values emphasize higher frequencies).
POWER = 2.0

# SEGMENT_DURATION: The duration of each audio segment for processing (in seconds). UNUSED YET
SEGMENT_DURATION = 20 

class_labels = {0: 'no_noise', 1: 'mic_noise', 2: 'Wind'}

mel_cols = [f'mel_spec_{i}' for i in range(1, 33)]
non_mel_cols = [
    'rms', 'zcr', 'contrast_7',
    'ACT', 'EVN', 'BI'
]
feature_cols = mel_cols + non_mel_cols  

def chirp_group_delay_cepstrum(audio, sr, frame_length, hop_length):
    t = np.arange(len(audio)) / sr
    chirp_signal = chirp(t, f0=100, f1=sr/2, t1=t[-1], method='linear')
    analytic_signal = hilbert(audio * chirp_signal)
    group_delay = np.unwrap(np.angle(analytic_signal))
    # Use vectorized framing to avoid explicit Python loop
    frames = librosa.util.frame(group_delay, frame_length=frame_length, hop_length=hop_length)
    fft_frames = np.fft.fft(frames, axis=0)
    cepstrum = np.fft.ifft(np.log1p(np.abs(fft_frames)), axis=0).real
    cgdc = np.mean(cepstrum, axis=0)
    return cgdc


def extract_features(wav_file,
                     selected_features,  # List of features you want
                     sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
                     fmin=FMIN, fmax=FMAX, power=POWER):
    try:
        y, sr = librosa.load(wav_file, sr=sr)
    except Exception as e:
        print(f"Error reading {wav_file}: {e}")
        return None

    if len(y) == 0:
        print(f"Warning: {wav_file} is empty.")
        return None

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_abs = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_abs, ref=np.max)

    mel_spec_db = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=power
    ))
    n_frames = mel_spec_db.shape[1]

    # Extract all features
    feature_dict = {}

    # Mel
    for i in range(n_mels):
        feature_dict[f'mel_spec_{i+1}'] = mel_spec_db[i, :].reshape(-1, 1)

    # Non-mel
    feature_dict['rms'] = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).T
    feature_dict['zcr'] = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length).T
    feature_dict['centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).T
    feature_dict['bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).T
    feature_dict['flatness'] = librosa.feature.spectral_flatness(y=y, hop_length=hop_length).T
    contrast = librosa.feature.spectral_contrast(S=stft_abs, sr=sr)
    for i in range(contrast.shape[0]):
        feature_dict[f'contrast_{i+1}'] = contrast[i, :].reshape(-1, 1)

    snr_mean = np.mean(stft_db, axis=0, keepdims=True).T
    feature_dict['snr_mean'] = snr_mean

    try:
        feature_dict['ACT'] = np.full((n_frames, 1), features.temporal_activity(y, dB_threshold=3, mode='fast', Nt=n_fft)[1])
        feature_dict['EVN'] = np.full((n_frames, 1), features.temporal_events(y, fs=sr, dB_threshold=3, mode='fast', Nt=n_fft)[2])
        feature_dict['ACI'] = np.full((n_frames, 1), features.acoustic_complexity_index(stft_abs)[2])
        feature_dict['BI']  = np.full((n_frames, 1), features.bioacoustics_index(stft_abs, np.linspace(0, sr / 2, stft_abs.shape[0])))
    except:
        feature_dict['ACT'] = feature_dict['EVN'] = feature_dict['ACI'] = feature_dict['BI'] = np.zeros((n_frames, 1))

    try:
        ste = np.sum(librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)**2, axis=0)
        feature_dict['STE'] = ste.reshape(-1, 1)
    except:
        feature_dict['STE'] = np.zeros((n_frames, 1))

    try:
        cgdc = chirp_group_delay_cepstrum(y, sr, frame_length=n_fft, hop_length=hop_length)
        feature_dict['CGDC'] = np.resize(cgdc.reshape(-1, 1), (n_frames, 1))
    except:
        feature_dict['CGDC'] = np.zeros((n_frames, 1))

    try:
        dwt_mean = np.mean(pywt.wavedec(y, 'db4', level=3)[0])
    except:
        dwt_mean = 0.0
    feature_dict['DWT_mean'] = np.full((n_frames, 1), dwt_mean)

    # Final stacking based on selected features
    selected_feature_arrays = []
    for feat in selected_features:
        if feat in feature_dict:
            selected_feature_arrays.append(np.resize(feature_dict[feat], (n_frames, 1)))
        else:
            print(f"⚠ Feature '{feat}' not found. Skipping.")

    if not selected_feature_arrays:
        print("❌ No valid features selected.")
        return None

    return np.hstack(selected_feature_arrays) 