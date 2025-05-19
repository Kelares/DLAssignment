import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from math import ceil

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# 1. Collect File Paths and Labels
def collect_files_and_labels(folder_path):
    file_paths = []
    labels = []

    for fname in os.listdir(folder_path):
        if fname.endswith('.wav') and len(fname) >= 2:
            accent_digit = fname[0]
            try:
                label = int(accent_digit) - 1  # labels: 0–4
            except ValueError:
                continue  # skip if not a number

            full_path = os.path.join(folder_path, fname)
            file_paths.append(full_path)
            labels.append(label)

    return file_paths, labels

# 2. Preprocess raw waveform
def preprocess_raw_audio(path, target_duration, sample_rate=16000):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        print("Not the correct sample rate")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    # Pad
    target_len = sample_rate * target_duration
    pad = (0, ceil(target_len - waveform.size(0)))
    waveform = F.pad(input=waveform, pad=pad)

    # Standardize
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

    return waveform

# 3. Dataset
class RawAudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_duration):
        self.file_paths = file_paths
        self.labels = labels
        self.max_duration = max_duration
        self.x = {}
        self.y = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = preprocess_raw_audio(self.file_paths[idx], self.max_duration)
        y = self.labels[idx]
        return x, y
        # I tried saving the results to memory but there are tooo many files to store them and program crushed.
        # if idx in self.x and idx in self.y:
        #     return self.x, self.y
        # else:
        #     self.x[idx] = preprocess_raw_audio(self.file_paths[idx], self.max_duration)
        #     self.y[idx] = self.labels[idx]
        #     return self.x[idx], self.y[idx]

# 4. CNN for 1D audio
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 5. Train and evaluate
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return correct / total

# 6. Main
def main():
    BATCH_SIZE = 32
    EPOCHS = 10
    MODEL_PATH = "model_weights.pth"
    file_paths, labels = collect_files_and_labels("Train")
    
    max_duration = 0
    for path in file_paths:
        waveform, sr = torchaudio.load(path)
        duration = waveform.shape[1] / sr
        if duration >= max_duration:
            max_duration = duration
    print(max_duration)

    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, stratify=labels, random_state=42)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    train_dataset = RawAudioDataset(train_paths, train_labels, max_duration)
    val_dataset = RawAudioDataset(val_paths, val_labels, max_duration)
    test_dataset = RawAudioDataset(test_paths, test_labels, max_duration)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=5).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except FileNotFoundError:
        print("NO MODEL TO LOAD")
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
if __name__ == "__main__":
    main()

