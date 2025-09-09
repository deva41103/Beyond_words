import torch
import torch.nn as nn

class Gloss2PoseTransformer(nn.Module):
    def __init__(self, vocab_size, pose_dim, hidden=256, num_layers=4, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.pose_dim = pose_dim
        self.hidden = hidden

        # Positional encoding matches hidden dim
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, hidden))

        # Projection layers
        self.pose_proj = nn.Linear(pose_dim, hidden)
        self.fc_out = nn.Linear(hidden, pose_dim)

        # Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=1024
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, gloss_id, tgt_seq):
        # Ensure proper input shapes
        if tgt_seq.dim() == 2:
            tgt_seq = tgt_seq.unsqueeze(0)  # Add batch dim if missing [1, seq_len, pose_dim]

        # Project pose to hidden dimension
        tgt_seq = self.pose_proj(tgt_seq)  # [B, seq_len, hidden]

        # Add positional encoding
        tgt_seq = tgt_seq + self.pos_encoder

        # Prepare memory (gloss embedding)
        if gloss_id.dim() == 0:
            gloss_id = gloss_id.unsqueeze(0)  # Add batch dim if needed
        memory = self.embedding(gloss_id).unsqueeze(0)  # [1, B, hidden]

        # Transformer expects [seq_len, batch, features]
        tgt_seq = tgt_seq.permute(1, 0, 2)  # [seq_len, B, hidden]

        # Transformer processing
        out = self.transformer(tgt_seq, memory)
        out = out.permute(1, 0, 2)  # [B, seq_len, hidden]

        return self.fc_out(out)  # [B, seq_len, pose_dim]
