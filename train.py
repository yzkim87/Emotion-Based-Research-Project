import os
import yaml
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_loader import AudioChunkDataset, collate_fn
from models.elr_gnn import ELR_GNN

# ------------------------------
# Load config
# ------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Reproducibility
seed = config.get("seed", 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ------------------------------
# Dataset & Loader
# ------------------------------
csv_path   = r"C:\Users\yunju\study\bk21\utils\saved_files_with_speaker.csv"
root_dir   = os.path.abspath(config["dataset_root"])
batch_size = config["batch_size"]

train_ds = AudioChunkDataset(csv_path, root_dir)
val_ds   = AudioChunkDataset(csv_path, root_dir)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

# ------------------------------
# Model & Optimizer
# ------------------------------
model = ELR_GNN(config).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

# ------------------------------
# Training & Validation
# ------------------------------
def train_epoch():
    model.train()
    total_loss = total_correct = total_count = 0

    for audio_feats, text_feats, labels, spks in tqdm(train_loader, desc="Training"):
        audio_feats = audio_feats.to(device)  # [B, T, audio_dim]
        text_feats  = text_feats.to(device)   # [B, T, text_dim]
        labels = labels.to(device)            # [B, T]
        spks = spks.to(device)                # [B, T]

        logits = model(text_feats, audio_feats, spks)  # [B, T, num_classes]

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss   += loss.item() * labels.numel()
        total_correct+= (preds == labels).sum().item()
        total_count  += labels.numel()

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def validate():
    model.eval()
    total_loss = total_correct = total_count = 0

    with torch.no_grad():
        for audio_feats, text_feats, labels, spks in tqdm(val_loader, desc="Validation"):
            audio_feats = audio_feats.to(device)
            text_feats  = text_feats.to(device)
            labels = labels.to(device)
            spks = spks.to(device)

            logits = model(text_feats, audio_feats, spks)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            preds = logits.argmax(dim=-1)
            total_loss   += loss.item() * labels.numel()
            total_correct+= (preds == labels).sum().item()
            total_count  += labels.numel()

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc

# ------------------------------
# Main Loop
# ------------------------------
best_acc = 0
for epoch in range(1, config["epochs"] + 1):
    tr_loss, tr_acc = train_epoch()
    val_loss, val_acc = validate()

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save every epoch
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f">>> [Best Updated] Epoch {epoch:02d} | Val Acc: {best_acc:.4f}")
