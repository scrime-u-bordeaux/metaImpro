import torch
from torch.utils.data import DataLoader, TensorDataset
from model import PianoGenie, compute_losses, build_model_and_features

def train_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    for features, seq_lens, targets in dataloader:
        features, seq_lens, targets = features.to(device), seq_lens.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features, seq_lens)
        losses = compute_losses(outputs, targets, seq_lens)
        losses['total_loss'].backward()
        optimizer.step()
        total_loss += losses['total_loss'].item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, seq_lens, targets in dataloader:
            features, seq_lens, targets = features.to(device), seq_lens.to(device), targets.to(device)
            outputs = model(features, seq_lens)
            losses = compute_losses(outputs, targets, seq_lens)
            total_loss += losses['total_loss'].item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    seq_len = 128
    num_epochs = 10
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss_plot = []
    val_loss_plot = []
    print("device is ", device)
    # Generate dummy dataset (replace with real data)
    pitches = torch.randint(0, 89, (1000, seq_len))
    _, feat = build_model_and_features(pitches, pitches.size(0), seq_len)
    dataset = TensorDataset(feat['features'], feat['seq_lens'], feat['targets'])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Model & optimizer
    model = PianoGenie().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, optimizer, train_loader, device)
        val_loss = eval_epoch(model, val_loader, device)
        train_loss_plot.append(train_loss)
        val_loss_plot.append(val_loss_plot)
        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        

    