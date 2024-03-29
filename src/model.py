import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
    """A module to add positional encoding to the input feature vectors."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)

class SequenceWiseAttention(nn.Module):
    """Applies sequence-wise attention to enhance important features."""
    def __init__(self, num_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 2, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1)
        attention = self.fc(avg_out)
        return x * attention.unsqueeze(-1)

class DTIModel(nn.Module):
    """A model for Drug-Target Interaction prediction using convolutional and transformer layers."""
    def __init__(self, protein_vocab_size, ligand_vocab_size, embedding_dim, conv_out_channels, kernel_size, nhead, nhid, nlayers, output_dim):
        super().__init__()
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim)
        self.ligand_embedding = nn.Embedding(ligand_vocab_size, embedding_dim)

        self.protein_pos_encoder = PositionalEncoding(embedding_dim)
        self.ligand_pos_encoder = PositionalEncoding(embedding_dim)

        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_out_channels, kernel_size=kernel_size, padding=kernel_size//2)

        encoder_layer = TransformerEncoderLayer(d_model=conv_out_channels, nhead=nhead, dim_feedforward=nhid, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        self.seq_wise_attn = SequenceWiseAttention(conv_out_channels)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_out_channels * 2, output_dim)

    def forward(self, protein_seq, ligand_seq):
        protein_embedded = self.protein_pos_encoder(self.protein_embedding(protein_seq)).permute(0, 2, 1)
        ligand_embedded = self.ligand_pos_encoder(self.ligand_embedding(ligand_seq)).permute(0, 2, 1)

        protein_features = F.relu(self.conv(protein_embedded))
        ligand_features = F.relu(self.conv(ligand_embedded))

        protein_encoded = self.transformer_encoder(protein_features.permute(0, 2, 1))
        ligand_encoded = self.transformer_encoder(ligand_features.permute(0, 2, 1))

        protein_attn = self.seq_wise_attn(protein_encoded)
        ligand_attn = self.seq_wise_attn(ligand_encoded)

        protein_pooled = self.adaptive_pool(protein_attn.permute(0, 2, 1)).squeeze(-1)
        ligand_pooled = self.adaptive_pool(ligand_attn.permute(0, 2, 1)).squeeze(-1)

        combined_features = torch.cat((protein_pooled, ligand_pooled), dim=1)
        output = self.fc(combined_features)
        return torch.sigmoid(output)


if __name__ == "__main__":
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Use MPS if available
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Model instantiation and the rest of the script remain the same
        model = DTIModel(
            protein_vocab_size=20,  # Example sizes, adjust according to your dataset
            ligand_vocab_size=50,
            embedding_dim=128,
            conv_out_channels=128,
            kernel_size=3,
            nhead=4,
            nhid=512,
            nlayers=6,
            output_dim=1
        ).to(device)

        print("Model architecture:")
        print(model)

