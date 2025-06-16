import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import opensmile
from transformers import AutoTokenizer, AutoModel
import soundfile as sf

# ------------------------------
# CONFIG
# ------------------------------
CSV_PATH = r"C:\Users\yunju\study\bk21\utils\saved_files_with_speaker.csv"
ROOT_DIR = r"C:\Users\yunju\study\bk21\data\working"  # base root dir for raw .pt files AND save
SAVE_DIR = r"C:\Users\yunju\study\bk21\data\pts"  # same as ROOT_DIR for clarity

chunk_len = 16000  # 1 sec if sampling_rate=16k
stride = 8000      # 50% overlap
sampling_rate = 16000

roberta_name = "FacebookAI/roberta-base"

# ------------------------------
# Load CSV
# ------------------------------
df = pd.read_csv(CSV_PATH)

# ------------------------------
# Init Extractors
# ------------------------------
print("[INFO] Loading RoBERTa and openSMILE...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

tokenizer = AutoTokenizer.from_pretrained(roberta_name)
roberta = AutoModel.from_pretrained(roberta_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
roberta.to(device)
roberta.eval()

# ------------------------------
# Preprocess all rows
# ------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
    dataset = row.Dataset
    emotion = row.Emotion
    file_id = row.File
    speaker_id = int(row.SpeakerID)
    transcript = str(row.Transcript) if "Transcript" in row else "This is a dummy text."

    # === 1) Load wave ===
    wave_path = os.path.join(ROOT_DIR, dataset, emotion, file_id + ".pt")
    wave = torch.load(wave_path).squeeze(0)  # [N]

    # === 2) Chunk wave ===
    chunks = []
    pos = 0
    N = wave.size(0)
    while pos + chunk_len <= N:
        chunk = wave[pos:pos + chunk_len]
        chunks.append(chunk)
        pos += stride
    if not chunks:
        # pad if too short
        pad_size = chunk_len - N
        chunk = torch.nn.functional.pad(wave, (0, pad_size))
        chunks = [chunk]

    wave_seq = torch.stack(chunks)  # [T, chunk_len]
    T = wave_seq.size(0)

    # === 3) Extract GLOBAL openSMILE feature ===
    tmp_wave_path = f"tmp_{file_id}.wav"
    sf.write(tmp_wave_path, wave.cpu().numpy(), samplerate=sampling_rate)
    smile_df = smile.process_file(tmp_wave_path)
    os.remove(tmp_wave_path)
    audio_feat = torch.tensor(smile_df.values.squeeze(), dtype=torch.float32)  # [D]
    audio_seq = audio_feat.unsqueeze(0).expand(T, -1)  # [T, D]

    # === 4) Extract GLOBAL RoBERTa feature ===
    inputs = tokenizer(transcript, return_tensors="pt").to(device)
    with torch.no_grad():
        output = roberta(**inputs)
        text_feat = output.last_hidden_state[:, 0, :].cpu().squeeze(0)  # [D]
    text_seq = text_feat.unsqueeze(0).expand(T, -1)  # [T, D]

    # === 5) Prepare meta ===
    label_seq = [emotion] * T
    spk_seq = [speaker_id] * T

    # === 6) Save ===
    save_subdir = os.path.join(SAVE_DIR, dataset, emotion)
    os.makedirs(save_subdir, exist_ok=True)
    torch.save({
        "wave": wave_seq,         # [T, chunk_len]
        "audio_feat": audio_seq,  # [T, D]
        "text_feat": text_seq,    # [T, D]
        "labels": label_seq,      # [T] (str)
        "speakers": spk_seq       # [T] (int)
    }, os.path.join(save_subdir, f"{file_id}.pt"))

print("✅ All features cached successfully.")
