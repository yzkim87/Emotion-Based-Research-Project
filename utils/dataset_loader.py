import os
import pandas as pd
import torch
from torch.utils.data import Dataset

# ------------------------------
# Emotion Label Map
# ------------------------------
EMOTIONS = [
    "Angry", "Disgust", "Fearful", "Frustration", "Happy",
    "Neutral", "Sad", "Surprised", "Excited"
]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMO = {i: e for e, i in EMO2IDX.items()}


# ------------------------------
# Dataset
# ------------------------------

class AudioChunkDataset(Dataset):
    """
    Loads preprocessed .pt files containing:
      - audio_feat: [T, audio_dim]
      - text_feat: [T, text_dim]
      - labels: list of strings or ints [T]
      - speakers: list of ints [T]
    """

    def __init__(self, csv_path: str, root_dir: str):
        """
        Args:
            csv_path (str): CSV file with columns: Dataset, Emotion, File, SpeakerID, Transcript (optional)
            root_dir (str): Root directory where preprocessed .pt files are stored
        """
        self.df = pd.read_csv(csv_path)
        self.root = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dataset = row.Dataset
        emotion = row.Emotion
        file_id = row.File

        # Path to .pt file
        pt_path = os.path.join(self.root, dataset, emotion, file_id + ".pt")
        obj = torch.load(pt_path)

        # Load tensors
        audio_seq = obj["audio_feat"]   # [T, audio_dim]
        text_seq  = obj["text_feat"]    # [T, text_dim]

        # Labels: map to int if they are str
        if isinstance(obj["labels"][0], str):
            label_seq = torch.tensor([EMO2IDX[l] for l in obj["labels"]], dtype=torch.long)
        else:
            label_seq = torch.tensor(obj["labels"], dtype=torch.long)

        spk_seq = torch.tensor(obj["speakers"], dtype=torch.long)  # [T]

        return audio_seq, text_seq, label_seq, spk_seq


# ------------------------------
# Collate Function
# ------------------------------

def collate_fn(batch):
    """
    batch: list of tuples from AudioChunkDataset
    Returns:
        audio_feats: [B, T, audio_dim]
        text_feats: [B, T, text_dim]
        labels: [B, T]
        spk_ids: [B, T]
    """
    audio_seqs, text_seqs, label_seqs, spk_seqs = zip(*batch)

    audio_feats = torch.stack(audio_seqs)  # [B, T, audio_dim]
    text_feats  = torch.stack(text_seqs)   # [B, T, text_dim]
    labels      = torch.stack(label_seqs)  # [B, T]
    spk_ids     = torch.stack(spk_seqs)    # [B, T]

    return audio_feats, text_feats, labels, spk_ids
