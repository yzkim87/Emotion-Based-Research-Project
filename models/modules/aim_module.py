import torch
import torch.nn as nn
import torch.nn.functional as F

class AIMModule(nn.Module):
    def __init__(self, input_dim, graph_dim, hidden_dim):
        """
        input_dim: output dim of LSTM encoder (i.e. contextual feature)
        graph_dim: output dim of GRN (i.e. latent graph feature)
        hidden_dim: final fused feature dim
        """
        super(AIMModule, self).__init__()

        self.conv_gate = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_filter = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)

        # ✅ CORRECT in_features: gated_output(H) + graph_feats(graph_dim)
        self.fusion_fc = nn.Linear(hidden_dim + graph_dim, hidden_dim)

        self.attn_query = nn.Parameter(torch.randn(hidden_dim))
        self.attn_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, context_feats, graph_feats):
        """
        Args:
            context_feats: [B, T, D]
            graph_feats: [B, T, D]
        Returns:
            fused_feats: [B, T, hidden_dim]
        """
        # --- Gated Conv ---
        # Conv1d expects [B, D, T]
        x = context_feats.transpose(1, 2)  # [B, D, T]

        gate = torch.sigmoid(self.conv_gate(x))
        filter = torch.tanh(self.conv_filter(x))
        gated_output = gate * filter  # [B, H, T]

        gated_output = gated_output.transpose(1, 2)  # [B, T, H]

        # --- Early Fusion ---
        # Gated Conv Output + Graph Output concat
        fusion_input = torch.cat([gated_output, graph_feats], dim=-1)
        fused = torch.tanh(self.fusion_fc(fusion_input))  # [B, T, H]

        # --- Adaptive Late Fusion ---
        # Attention weight to combine
        q = self.attn_query  # [H]
        attn_logits = torch.tanh(self.attn_fc(fused))  # [B, T, H]
        attn_logits = attn_logits @ q  # [B, T]
        attn_weight = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, T, 1]

        # Weighted sum: fused * attn + residual fused
        output = fused * attn_weight + fused  # [B, T, H]

        return output
