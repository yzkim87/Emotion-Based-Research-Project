import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    텍스트 + 오디오 피처 concat → Bi-LSTM → Contextual encoding
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.1):
        """
        Args:
            input_size (int): 텍스트+오디오 concat dimension
            hidden_size (int): LSTM hidden size
        """
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bidirectional이면 output dimension = 2*hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, T, D] - B: batch, T: time steps, D: input_size
            lengths: optional, 길이 리스트 for pack_padded_sequence

        Returns:
            output: [B, T, output_size]
        """
        # 길이 지원 가능 (옵션)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, _ = self.lstm(x)

        return output
