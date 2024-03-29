import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import selfies as sf
from sklearn.model_selection import train_test_split

# Special Tokens
SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2

# Maximum length settings based on EDA
MAX_SELFIES_LENGTH = 350  # Slightly above max to accommodate special tokens
MAX_SEQ_LENGTH = 780  # Slightly above max to accommodate special tokens

class ProteinSELFIESDataset(Dataset):
    def __init__(self, dataframe, max_seq_length=MAX_SEQ_LENGTH, max_selfies_length=MAX_SELFIES_LENGTH):
        self.dataframe = dataframe
        self.max_seq_length = max_seq_length
        self.max_selfies_length = max_selfies_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        seq, selfies = row['Seq'], row['SELFIES']

        seq_encoded = self.encode_sequence(seq)
        selfies_encoded = self.encode_selfies(selfies)

        return {
            'seq': seq_encoded,
            'selfies': selfies_encoded,
            'isActive': torch.tensor(row['isActive'], dtype=torch.float)
        }

    def encode_sequence(self, seq):
        encoded = [SOS_TOKEN] + [ord(c) - ord('A') + 3 for c in seq.upper()[:self.max_seq_length - 2]] + [EOS_TOKEN]
        padded = encoded + [PAD_TOKEN] * (self.max_seq_length - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    def encode_selfies(self, selfies):
        decoded = sf.decoder(selfies)
        encoded = [SOS_TOKEN] + [ord(c) for c in decoded[:self.max_selfies_length - 2]] + [EOS_TOKEN]
        padded = encoded + [PAD_TOKEN] * (self.max_selfies_length - len(encoded))
        return torch.tensor(padded, dtype=torch.long)


def load_dataset(csv_file, test_size=0.2, batch_size=32):
    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_dataset = ProteinSELFIESDataset(train_df)
    test_dataset = ProteinSELFIESDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    csv_file = "/Users/will/Documents/PubChem/7GeneSubSet/Enriched_Set_60percent_similarity.csv"
    train_loader, test_loader = load_dataset(csv_file)
    for batch in train_loader:
        print(batch['seq'], batch['selfies'], batch['isActive'])
        break  # Just show the first batch for demonstration
