import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_loader import AudioChunkDataset, collate_fn, IDX2EMO
from models.elr_gnn import ELR_GNN

# ------------------------------
# Load config
# ------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ------------------------------
# Dataset & Loader
# ------------------------------
csv_path = r"C:\Users\yunju\study\bk21\utils\saved_files_with_speaker.csv"
root_dir = os.path.abspath(config["dataset_root"])
batch_size = config["batch_size"]

test_ds = AudioChunkDataset(csv_path, root_dir)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

# ------------------------------
# Model
# ------------------------------
model = ELR_GNN(config).to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# ------------------------------
# Inference
# ------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for audio_feats, text_feats, labels, spks in tqdm(test_loader, desc="Evaluating"):
        audio_feats = audio_feats.to(device)
        text_feats  = text_feats.to(device)
        labels = labels.to(device)
        spks = spks.to(device)

        logits = model(text_feats, audio_feats, spks)  # [B, T, num_classes]
        probs = torch.softmax(logits, dim=-1)          # [B, T, num_classes]
        preds = probs.argmax(dim=-1)                   # [B, T]

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

# ------------------------------
# Flatten & Convert
# ------------------------------
all_preds = torch.cat(all_preds, dim=0).view(-1).numpy()
all_labels = torch.cat(all_labels, dim=0).view(-1).numpy()

# Convert to int for sklearn
all_preds = all_preds.astype(int)
all_labels = all_labels.astype(int)

# ------------------------------
# Mapping to Emotion Labels
# ------------------------------
emo_labels = [IDX2EMO[i] for i in sorted(IDX2EMO.keys())]
y_true = [IDX2EMO[i] for i in all_labels]
y_pred = [IDX2EMO[i] for i in all_preds]

# ------------------------------
# Classification Report
# ------------------------------
report = classification_report(y_true, y_pred, labels=emo_labels, digits=4)
print("\n=== Classification Report ===")
print(report)

# Save to file
with open("evaluation_report.txt", "w") as f:
    f.write(report)
print("✅ Saved detailed report to evaluation_report.txt")

# ------------------------------
# Macro F1
# ------------------------------
macro_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Macro F1: {macro_f1:.4f}")

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred, labels=emo_labels)
cm_df = pd.DataFrame(cm, index=emo_labels, columns=emo_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Saved confusion matrix as confusion_matrix.png")
