import os
import sys

# ensure project root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn

from models.modules.lstm_encoder import LSTMEncoder
from models.modules.graph_constructor import GraphConstructor
from models.modules.graph_random_network import GraphRandomNetwork
from models.modules.aim_module import AIMModule


class ELR_GNN(nn.Module):
    """
    Efficient Long-distance Latent Relation-aware Graph Neural Network (ELR-GNN)
    Combines LSTM context, speaker graph, GFPush latent relations, AIM fusion, and final classification.
    """

    def __init__(self, config):
        super().__init__()
        self.text_dim = config["text_dim"]
        self.audio_dim = config["audio_dim"]
        self.lstm_hidden = config["lstm_hidden"]
        self.grn_hops = config["grn_hops"]
        self.aim_hidden = config["aim_hidden"]
        self.num_classes = config["num_classes"]
        self.graph_window = config["graph_window"]

        # LSTM Encoder for contextual features
        self.encoder = LSTMEncoder(
            input_size=self.text_dim + self.audio_dim, hidden_size=self.lstm_hidden
        )
        # Graph constructor + random network
        self.graph_constructor = GraphConstructor(window_size=self.graph_window)
        self.graph_random_network = GraphRandomNetwork(num_hops=self.grn_hops)
        # Auxiliary Information Module
        self.aim = AIMModule(
            input_dim=self.encoder.output_size,  # LSTM out dim
            graph_dim=self.encoder.output_size,  # GRN out dim
            hidden_dim=self.aim_hidden,  # desired fused hidden size
        )
        # Final classifier
        self.classifier = nn.Linear(self.aim_hidden, self.num_classes)

    def forward(
        self,
        text_embeds: torch.Tensor,
        audio_feats: torch.Tensor,
        speaker_ids: torch.Tensor,
    ):
        """
        Args:
            text_embeds : [B, T, D1]
            audio_feats : [B, T, D2]
            speaker_ids : [B, T]
        Returns:
            logits      : [B, T, num_classes]
        """
        B, T, _ = text_embeds.size()
        # concatenate modalities
        x = torch.cat([text_embeds, audio_feats], dim=-1)  # [B, T, D1+D2]
        # LSTM encoding
        lstm_feats = self.encoder(x)  # [B, T, 2*hidden]

        # build graph per sample
        graph_outs = []
        for b in range(B):
            node_feats = lstm_feats[b]  # [T, D]
            spk_ids = speaker_ids[b].tolist()
            # adjacency
            edge_index, _ = self.graph_constructor.build_adjacency(spk_ids)
            # apply GRN
            out = self.graph_random_network(node_feats, edge_index)
            graph_outs.append(out)
        graph_feats = torch.stack(graph_outs, dim=0)  # [B, T, D]

        # AIM fusion
        fused = self.aim(lstm_feats, graph_feats)  # [B, T, H]
        # classification
        logits = self.classifier(fused)  # [B, T, num_classes]
        return logits


# # ==============================
# #       Test script
# # ==============================
# if __name__ == "__main__":
#     # Dummy config
#     config = {
#         "text_dim": 768,
#         "audio_dim": 88,
#         "lstm_hidden": 128,
#         "grn_hops": 3,
#         "aim_hidden": 64,
#         "num_classes": 6,
#         "graph_window": 5,
#     }
#     model = ELR_GNN(config)
#     model.eval()

#     # Create dummy inputs: batch_size=2, seq_len=4
#     B, T = 2, 4
#     text_embeds = torch.randn(B, T, config["text_dim"])
#     audio_feats = torch.randn(B, T, config["audio_dim"])
#     # Speaker IDs: all same speaker per batch or random
#     speaker_ids = torch.tensor([[1001, 1001, 1001, 1001], [1002, 1002, 1002, 1002]])

#     # Forward pass
#     with torch.no_grad():
#         logits = model(text_embeds, audio_feats, speaker_ids)
#     print("Logits shape:", logits.shape)  # expected [2,4,6]
