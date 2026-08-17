# 🌶️ Paprika

**A web-based speech analysis tool that predicts emotion from voice and matches you to the celebrity you sound like.**

Upload or record a few seconds of speech. Paprika predicts the speaker's **emotion**, **emotional intensity**, and **gender** with a multi-task model built on wav2vec 2.0, then retrieves the **top-3 most similar celebrity voices** from VoxCeleb1 using ECAPA-TDNN speaker embeddings.


<img width="400" height="214" alt="Kapture 2026-08-15 at 19 13 07" src="https://github.com/user-attachments/assets/966cee76-b9df-4c39-a065-7e8d63765082" />




---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Datasets & Licensing](#datasets--licensing)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Multi-task emotion recognition** — a single model jointly predicts emotion, intensity, and gender, sharing a wav2vec 2.0 representation across three heads.
- **Celebrity voice matching** — cosine retrieval over pre-computed ECAPA-TDNN embeddings for <N> VoxCeleb1 speakers, returning the top-3 matches with similarity scores.
- **Browser-based** — record directly in the page or upload a file; no local setup needed for the end user.
- **Format-agnostic input** — audio is normalised to 16 kHz mono PCM WAV server-side via ffmpeg, so `.mp3`, `.m4a`, and `.webm` all work.

---

## How It Works
```

                 ┌──────────────────────────────┐
  audio ────────▶│  ffmpeg → 16 kHz mono PCM    │
                 └───────────────┬──────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │  wav2vec 2.0 (base)  │              │  ECAPA-TDNN          │
   │  768-d frame seq     │              │  192-d speaker emb   │
   └──────────┬───────────┘              └──────────┬───────────┘
              ▼                                     ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │  4-layer 1D CNN      │              │  cosine similarity   │
   │  + 3 task heads      │              │  vs. VoxCeleb1 bank  │
   └──────────┬───────────┘              └──────────┬───────────┘
              ▼                                     ▼
   emotion / intensity / gender              top-3 celebrities
```

### System Design

<img width="2332" height="1295" alt="image" src="https://github.com/user-attachments/assets/1cbebdfa-8c10-4d13-bc54-1282d3041c36" />

### Emotion model

Frame-level wav2vec 2.0 hidden states (768-d) are instance-normalised and
passed to a four-layer 1D CNN (128 → 256 → 256 → 128 channels, kernel size 5,
BatchNorm + ReLU + Dropout 0.4), pooled to a fixed vector by adaptive average
pooling, and projected through a 256-d shared bottleneck into three linear
heads for emotion, intensity, and gender.

### Speaker matching

Speaker embeddings for the VoxCeleb1 bank are pre-computed offline and stored in a single `.npz` archive, so inference is one forward pass plus a vectorised cosine similarity, matching runs in <X> ms on CPU.

---

## Results

**Dataset:** RAVDESS — 8 emotion classes · **Reported on:** validation set

<img width="4000" height="3200" alt="summary_confusion_XAI_FIXED" src="https://github.com/user-attachments/assets/d5d585e0-86a6-4be4-bb19-f528f3619ea9" />
The model frequently confuses neutral with calm and happy, reflecting semantic and acoustic proximity among these classes.

<img width="4000" height="880" alt="f1_heatmap_XAI_FIXED" src="https://github.com/user-attachments/assets/be6f54b8-ca4f-42e0-a861-8c0c6a6262ea" />

The F1 heatmap reveals that angry, sad, and fearful achieved relatively higher F1 scores, while happy and neutral had lower values due to overlaps.



---

## Tech Stack

| Layer | Technology |
|:--|:--|
| Models | PyTorch, HuggingFace Transformers (`facebook/wav2vec2-base`), SpeechBrain (`spkrec-ecapa-voxceleb`) |
| Audio | librosa, soundfile, pydub, ffmpeg |
| Backend | Flask, Flask-CORS |
| Frontend | Vanilla HTML / CSS / JavaScript, Web Audio API |
| Analysis | Jupyter, scikit-learn, pandas, matplotlib |

---

## Getting Started

### Prerequisites

- Python 3.9+
- **ffmpeg** on your `PATH` — `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Debian/Ubuntu)

### Installation

```bash
git clone https://github.com/DawnishL/Paprika.git
cd Paprika

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download model weights

Trained weights and the pre-computed embedding bank are distributed via
[**GitHub Releases**](https://github.com/DawnishL/Paprika/releases) — they are
too large for the repository. *(Release upload in progress.)*

```bash
mkdir -p assets
# Download the following from the release and place them in assets/:
#   w2v2_multitask_cnn_XAI_best.pt
#   speaker_embeddings.npz
#   vox1_meta.csv
#   le_emotion.pkl  le_intensity.pkl  le_gender.pkl
```

### Run

```bash
python app.py                # API → http://localhost:5000
open index.html              # or serve with any static file server
```

---

## API Reference

#### `POST /api/eig`

Predicts emotion, intensity, and gender.

```bash
curl -X POST -F "file=@sample.wav" http://localhost:5000/api/eig
```

```json
{ "emotion": "happy", "intensity": "strong", "gender": "female" }
```

#### `POST /api/similarity`

Returns the three closest celebrity voices with cosine similarity scores.

```bash
curl -X POST -F "file=@sample.wav" http://localhost:5000/api/similarity
```

```json
{ "match": [["Emma Watson", 0.71], ["Keira Knightley", 0.68], ["Kate Winslet", 0.64]] }
```

Both endpoints return `400` if no file is attached and `500` with an `error` field on failure.

---

## Project Structure

```
Paprika/
├── app.py                  # Flask API — /api/eig, /api/similarity
├── index.html              # frontend: upload / record UI
├── style.css
├── notebooks/              # feature extraction, training, XAI analysis
├── visualization_SER/      # evaluation and explainability plots
├── UML&DFD.pdf             # class diagram and data flow diagram
├── requirements.txt
├── LICENSE
└── assets/                 # create this and place downloaded weights here (gitignored)
```

---

## Limitations & Future Work

Being upfront about what doesn't work yet:

- **Acted vs. natural emotion.** The model is trained on acted speech, where emotions are exaggerated and clearly separated. Performance on spontaneous, conversational audio is expected to degrade substantially.
- **English-only.** Both the training corpus and the wav2vec 2.0 checkpoint are English; cross-lingual transfer is untested.
- **Latency.** Each request shells out to ffmpeg and writes temporary files to disk. Moving to in-memory decoding and batching would cut round-trip time considerably.
- **No streaming.** Inference runs on complete utterances only; real-time frame-by-frame prediction would require a causal architecture.
- **Celebrity matching is a similarity ranking, not identification.** A high cosine score reflects vocal-timbre proximity within the VoxCeleb1 bank, not a claim about identity.

---

## Datasets & Licensing

- **RAVDESS** — emotion / intensity / gender labels.
- **[VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)** — speaker embedding bank. Research use only.

No raw audio from either dataset is redistributed in this repository. This project is for research and demonstration purposes and is not intended for identification of individuals.

Code released under the [MIT License](LICENSE).

---

## Acknowledgements

Built on [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (Baevski et al., 2020) and the [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) speaker verification model (Desplanques et al., 2020), via HuggingFace Transformers and SpeechBrain.

<!-- 如果是课程项目或团队作业，在这里注明合作者和你个人负责的部分。
     面试官很在意"这个项目里哪些是你做的"，主动说明比被追问好。 -->
