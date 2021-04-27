import librosa

audio_file = "noise.wav"

# 8khz sampling rate
sample_rate = 8000

# 25ms frame time => 8000 * 0.025 = 200 samples frame length,
# 10ms overlap time => 8000 * 0.010 = 80 samples of overlap,
# so hop_length should be 200 - 80
hop_length = 120

# Load the audio file, note with specified sample rate
signal, sample_rate = librosa.load(audio_file, sr=sample_rate)

# Extract mfccs from loaded audio signal, given calculated hop_length
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=1, sr=sample_rate, hop_length=hop_length)

# Prints 67 frames in the audio
print(mfccs.shape)
