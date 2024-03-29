import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import load_dataset  # Ensure this matches your dataset loading function
from model import CascadedConvTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Train DTI Prediction Model")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--dataset_path", type=str, default="/Users/will/Documents/SELFSeq/SELFSeq/data/raw/Enriched_Set_60percent_similarity.csv", help="Path to dataset")
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "protein_vocab_size": 23,
        "selfies_vocab_size": 112,
        "embedding_dim": 256,
        "nhead": 8,
        "nhid": 4096,
        "nlayers": 12,
        "output_dim": 1,
    }

    train_loader, val_loader = load_dataset(args.dataset_path, test_size=0.2)

    model = CascadedConvTransformer(
        protein_vocab_size=config["protein_vocab_size"],
        selfies_vocab_size=config["selfies_vocab_size"],
        embedding_dim=config["embedding_dim"],
        nhead=config["nhead"],
        nhid=config["nhid"],
        nlayers=config["nlayers"],
        output_dim=config["output_dim"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    writer = SummaryWriter()  # For TensorBoard logging

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.epochs, writer)
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    torch.save(model.state_dict(), 'drug_target_interaction_model.pth')
    print('Model training complete.')
    writer.close()

def train(model, loader, optimizer, criterion, device, epoch, total_epochs, writer):
    # Now use `total_epochs` instead of `args.epochs` within the function
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        protein_seq, selfies_seq, labels = batch['seq'].to(device), batch['selfies'].to(device), batch['isActive'].to(device)
        optimizer.zero_grad()
        outputs = model(protein_seq, selfies_seq)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 10 == 0:  # Adjust the logging frequency as needed
            print(f'Epoch [{epoch}/{total_epochs}], Step [{i}/{len(loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(loader) + i)

    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch} Training Complete. Avg Loss: {avg_loss:.4f}\n')
    return avg_loss


def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            protein_seq, selfies_seq, labels = batch['seq'].to(device), batch['selfies'].to(device), batch['isActive'].to(device)
            outputs = model(protein_seq, selfies_seq)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()

            # Calculate accuracy (if applicable)
            predicted = outputs.round()  # Assuming binary classification
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)

            if i % 10 == 0:  # Adjust the logging frequency as needed
                writer.add_scalar('Validate/Loss', loss.item(), epoch * len(loader) + i)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples * 100
    print(f'Validation Complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    writer.add_scalar('Validate/Accuracy', accuracy, epoch)
    return avg_loss



if __name__ == "__main__":
    main()
