import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import librosa
import joblib
import os
import uuid
from pydub import AudioSegment
import numpy as np
import pandas as pd
import subprocess
from speechbrain.pretrained import EncoderClassifier

# Load pretrained speaker embeddings and metadata
speaker_data = np.load("/speaker_embeddings.npz")
speaker_ids = list(speaker_data.keys())
embeds = np.array([speaker_data[sid].squeeze() for sid in speaker_ids])
meta = pd.read_csv('/vox1_meta.csv', sep='\t')
id2name = pd.Series(meta['VGGFace1 ID'].values, index=meta['VoxCeleb1 ID']).to_dict()
# Initialize speaker recognition model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)

def convert_to_pcm_wav(src_path, dst_path):
    cmd = [
        'ffmpeg', '-y', '-i', src_path,
        '-ar', '16000', '-ac', '1', '-f', 'wav', dst_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return os.path.exists(dst_path)

def extract_embedding(audio_path):
    signal = classifier.load_audio(audio_path)
    emb = classifier.encode_batch(signal.unsqueeze(0)).squeeze().detach().numpy()
    return emb

def match_top3(audio_path):
    query_emb = extract_embedding(audio_path)
    sim = np.dot(embeds, query_emb) / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(query_emb))
    top3_idx = np.argsort(sim)[-3:][::-1]
    results = []
    for i in top3_idx:
        sid = speaker_ids[i]
        name = id2name.get(sid, sid)
        similarity = float(sim[i])
        results.append([name, similarity])
    return results

# Initialize emotion recognition model
SAMPLE_RATE = 16000
MODEL_NAME = "facebook/wav2vec2-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
w2v2_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()
# Load label encoders for emotion, intensity and gender
le_e = joblib.load("/le_emotion.pkl")
le_i = joblib.load("/le_intensity.pkl")
le_g = joblib.load("/le_gender.pkl")

class DeeperMultiTaskCNNClassifier(nn.Module):
    def __init__(self, emb_dim, n_emotion, n_intensity, n_gender, cnn_channels=128, kernel_size=5, hidden=256, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(emb_dim, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels*2, cnn_channels*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels*2, cnn_channels, kernel_size=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_emotion   = nn.Linear(hidden, len(le_e.classes_))
        self.head_intensity = nn.Linear(hidden, len(le_i.classes_))
        self.head_gender    = nn.Linear(hidden, len(le_g.classes_))

    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        feat = self.fc(cnn_out)
        return self.head_emotion(feat), self.head_intensity(feat), self.head_gender(feat)

emotion_model = DeeperMultiTaskCNNClassifier(
    emb_dim=768,
    n_emotion=len(le_e.classes_),
    n_intensity=len(le_i.classes_),
    n_gender=len(le_g.classes_)
).to(device)
emotion_model.load_state_dict(torch.load("w2v2_multitask_cnn_XAI_best.pt", map_location=device))
emotion_model.eval()

def safe_audio_to_wav(src_path):
    fixed_path = f"{os.path.splitext(src_path)[0]}_fixed.wav"
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio.export(fixed_path, format="wav")
    return fixed_path

def predict_audio(path, le_e, le_i, le_g):
    fixed_path = safe_audio_to_wav(path)
    y, sr = sf.read(fixed_path)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if y.ndim > 1:
        y = y.mean(axis=1)
    inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb_seq = w2v2_model(input_values=inputs.input_values.to(device)).last_hidden_state.cpu().numpy()[0]
        emb_seq = (emb_seq - emb_seq.mean()) / (emb_seq.std() + 1e-8)
        single_x_tensor = torch.tensor(emb_seq, dtype=torch.float32).unsqueeze(0).to(device)
        out_e, out_i, out_g = emotion_model(single_x_tensor)
        pred_e = out_e.argmax(1).cpu().numpy()[0]
        pred_i = out_i.argmax(1).cpu().numpy()[0]
        pred_g = out_g.argmax(1).cpu().numpy()[0]
    try:
        os.remove(fixed_path)
    except Exception:
        pass
    return {
        "emotion": le_e.inverse_transform([pred_e])[0],
        "intensity": le_i.inverse_transform([pred_i])[0],
        "gender": le_g.inverse_transform([pred_g])[0]
    }

# Initialize Flask app with CORS support
app = Flask(__name__)
CORS(app)

@app.route('/api/similarity', methods=['POST'])
def api_similarity():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        temp_id = str(uuid.uuid4())
        temp_path = f'temp_{temp_id}.wav'
        file.save(temp_path)
        pcm_path = f'temp_{temp_id}_pcm.wav'
        if not convert_to_pcm_wav(temp_path, pcm_path):
            raise Exception("ffmpeg failed to convert to PCM wav.")
        result = match_top3(pcm_path)
        try:
            os.remove(temp_path)
            os.remove(pcm_path)
        except Exception:
            pass
        return jsonify({"match": result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/eig', methods=['POST'])
def api_eig():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        temp_id = str(uuid.uuid4())
        temp_path = f'temp_{temp_id}.wav'
        file.save(temp_path)
        result = predict_audio(temp_path, le_e, le_i, le_g)
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return 'API is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
