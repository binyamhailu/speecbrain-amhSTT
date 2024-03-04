from speechbrain.inference.ASR import EncoderASR

# Load the ASR model
asr_model = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-dvoice-amharic",
    savedir="pretrained_models/asr-wav2vec2-dvoice-amharic"
)

# Transcribe Amharic speech from an audio file
transcription = asr_model.transcribe_file('am.wav')

# Print the transcription
print("Transcription:", transcription)
