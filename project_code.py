
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!pip install jiwer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import time
from jiwer import wer
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
import pickle
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = 29  # a-z + space + blank+'
lr = 0.001 #learning rate (how large the update step is)
epochs =256 #how many times we want to interate
sample_rate = 16000
n_mels = 80
batch_size = 8
data_root = "/content/drive/MyDrive/ASR/new_dev_clean"
#checkpoint_path = '/content/drive/MyDrive/Augmented_BiLSTM_ASR_1988.pth'
test_audio ="/content/drive/MyDrive/ASR/TESTING_SET/3170/137823/3170-137823-0013.flac"
test_audio_text="WE'D BETTER NOT TRY TO MOVE HIM TOM DECIDED WE'LL GET AN AMBULANCE"
test_audio_text=test_audio_text.lower()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=512,
    hop_length=160,
    win_length=400,
    n_mels=n_mels
)

char_map_str = [" "] + list("abcdefghijklmnopqrstuvwxyz") + ["'", "<blank>"]
char_to_idx = {c:i for i,c in enumerate(char_map_str)}
blank_idx = num_classes - 1

flac_files = sorted(glob(os.path.join(data_root, "**", "*.flac"), recursive=True))
print("Found FLAC files:", len(flac_files))
print(flac_files)

transcripts = {}
for trans_path in glob(os.path.join(data_root, "**", "*.trans.txt"), recursive=True):
    with open(trans_path, "r") as f:
        for line in f:
            fname, *text = line.strip().split()
            transcripts[fname] = " ".join(text)

print("Loaded transcripts:", len(transcripts))
print(transcripts)

missing = []
for f in flac_files:
    name = f.split('/')[-1].replace('.flac', '')
    if name not in transcripts:
        missing.append(name)

if missing:
    print("Missing transcripts for:", missing)
else:
    print("All audio files have transcripts")

def text_to_labels(text):
    return [char_to_idx[c] for c in text.lower() if c in char_to_idx]

def load_audio(path, target_sr=16000, training=True):
    waveform, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(waveform)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    waveform = waveform.unsqueeze(0)  # [1, T]

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    # Speed perturbation (TRAINING ONLY)
    if training and torch.rand(1) < 0.5:
        rate = 0.9 if torch.rand(1) < 0.5 else 1.1
        new_sr = int(target_sr * rate)

        waveform = torchaudio.functional.resample(
            waveform, target_sr, new_sr
        )

        waveform = torchaudio.functional.resample(
            waveform, new_sr, target_sr
        )

    return waveform, target_sr

class TemporalCNN_BiLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=256, lstm_layers=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d((2,2), stride=(2,1))
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))


        self.lstm = nn.LSTM(
            input_size=64 * 20,   # 64 *freq/4
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        B, C, Freq, T = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * Freq)

        x, _ = self.lstm(x)
        #x = self.dropout(x)

        x = self.fc(x)

        return x

model = TemporalCNN_BiLSTM(num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

class ASRDataset(Dataset):
    def __init__(self, flac_files, transcripts, mel_transform, training=True):
        self.flac_files = flac_files
        self.transcripts = transcripts
        self.mel_transform = mel_transform
        self.training = training

    def __len__(self):
        return len(self.flac_files)

    def __getitem__(self, idx):
        flac_path = self.flac_files[idx]
        name = flac_path.split('/')[-1].replace('.flac', '')

        waveform, _ = load_audio(flac_path, training=self.training)

        mel_spec = self.mel_transform(waveform)  # [1, n_mels, T]

        # SpecAugment (TRAINING ONLY)
        if self.training:
            n_mels = mel_spec.shape[1]
            num_frames = mel_spec.shape[2]

            freq_mask = int(n_mels * 0.2)
            time_mask = max(1, int(num_frames * 0.05))

            mel_spec = T.FrequencyMasking(freq_mask)(mel_spec)
            mel_spec = T.TimeMasking(time_mask)(mel_spec)

        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        label = torch.tensor(
            text_to_labels(self.transcripts[name]),
            dtype=torch.long
        )

        return mel_spec.squeeze(0), label  # [80, T], [L]

def collate_fn(batch):
        batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        specs, labels = zip(*batch)

        max_len = specs[0].shape[1]
        padded_specs = []
        input_lengths = []
        for spec in specs:
          T = spec.shape[1]
          pad = max_len - T
          padded = F.pad(spec, (0, pad))
          padded_specs.append(padded)
          input_lengths.append(T)
        padded_specs = torch.stack(padded_specs)  # [B, 80, max_T]

        # CNN time reduction = 2× with floor rounding
        input_lengths = torch.tensor([(T // 2) for T in input_lengths], dtype=torch.long)



        targets = torch.cat(labels)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

        return padded_specs.unsqueeze(1), targets, input_lengths, target_lengths

dataset = ASRDataset(flac_files, transcripts, mel_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
print(len(loader))

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=42)

'''fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n===== Fold {fold+1} =====")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    model = TemporalCNN_BiLSTM(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    checkpoint_path = f"/content/drive/MyDrive/WEIGHTS/checkpoint_fold_{fold}.pt"
    best_val_loss = float("inf")

    train_losses = []
    start = time.time()

  for epoch in range(start_epoch, epochs):
    total_loss = 0.0
    model.train()
    for mel_specs, targets, input_lengths, target_lengths in loader:
        mel_specs = mel_specs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()# removes grdient of previous

        logits = model(mel_specs)            # [B, T, C]
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2) # [T, B, C] permute because ctc_loss required input in this format and dim=-1 because we apply softmax on num_class

        input_lengths = torch.clamp(input_lengths, max=log_probs.size(0))

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss in batch, skipping")
            continue

        loss.backward()#calculate change in loss means gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()#changes lr accordingly

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)
    if epoch % 10 ==0
      model.eval()
      val_loss = 0.0

      with torch.no_grad():
        for mel_specs, targets, input_lengths, target_lengths in val_loader:
          mel_specs = mel_specs.to(device)
          targets = targets.to(device)
          input_lengths = input_lengths.to(device)
          target_lengths = target_lengths.to(device)

          logits = model(mel_specs)
          log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
          input_lengths = torch.clamp(input_lengths, max=log_probs.size(0))

          loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
          val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Fold {fold} | Epoch {epoch} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

      if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
        }, checkpoint_path)



  end = time.time()
  print(f"Training finished in {(end - start):.2f}s")'''

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n===== Fold {fold+1} =====")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = TemporalCNN_BiLSTM(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    checkpoint_path = f"/content/drive/MyDrive/new_checkpoints_fold_{fold}.pt"

    if os.path.exists(checkpoint_path):
      print(f"Fold {fold+1} already trained, skipping")
      continue
    best_val_loss = float("inf")

    start_epoch = 0
    start_time = time.time()
    train_losses=[]
    val_losses=[]
    for epoch in range(start_epoch, epochs):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for mel_specs, targets, input_lengths, target_lengths in train_loader:
            mel_specs = mel_specs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            logits = model(mel_specs)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.clamp(input_lengths, max=log_probs.size(0))

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss in batch, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATE --------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for mel_specs, targets, input_lengths, target_lengths in val_loader:
                mel_specs = mel_specs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(mel_specs)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                input_lengths = torch.clamp(input_lengths, max=log_probs.size(0))

                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(
            f"Fold {fold+1} | Epoch {epoch+1} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}"
        )

        # -------- SAVE BEST --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, checkpoint_path)

    elapsed = time.time() - start_time
    print(f"Fold {fold+1} finished in {elapsed:.2f}s")

    fold_results.append(best_val_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Train Loss")

    'if len(val_losses) > 0:
      plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label="Val Loss")

    plt.title(f"Training Loss vs Epochs (Fold {fold+1})")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.grid(True)
    plt.legend()

    plot_path = f"loss_plots/fold_{fold+1}_loss.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {plot_path}")

def ctc_greedy_decode(pred_ids, char_map_str, blank_idx):
    pred_ids = pred_ids.tolist()
    decoded = []
    prev = blank_idx
    for p in pred_ids:
        if p != prev and p != blank_idx:
            decoded.append(char_map_str[p])
        prev = p
    return "".join(decoded)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])

model.eval()

waveform,sr = load_audio(test_audio,target_sr=16000,training=False)
print("Waveform shape:", waveform.shape)  # must be [1, T]

sample_mel = mel_transform(waveform)
print("Mel shape:", sample_mel.shape)     # [1, n_mels, time]

sample_mel = (sample_mel - sample_mel.mean()) / (sample_mel.std() + 1e-5)

sample_mel = sample_mel.unsqueeze(0).to(device)  # [1, 1, Freq, Time]

with torch.no_grad():
    logits = model(sample_mel)
    predicted_ids = torch.argmax(logits, dim=-1).cpu()

decoded_text = ctc_greedy_decode(predicted_ids[0], char_map_str, blank_idx)
print("Predicted text:", decoded_text)

wer_score = wer(test_audio_text, decoded_text)
print("WER Score:", wer_score)

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Training Loss vs Epochs (Fixed LR)")
plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.grid(True)
plt.show()

!pip install nemo-toolkit[asr]

!pip uninstall -y numpy
!pip install numpy==2.1.3

from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_small")
print(model)

pretrainM_root = "/content/drive/MyDrive/ASR/dev_clean"
save_model = "/content/drive/MyDrive/ASR/asr_conformer_finetuned.nemo"

train_files, val_files = train_test_split(
    pretrainM_flac_files,
    test_size=0.15,
    random_state=42
)
len(train_files), len(val_files)

pretrainM_flac_files = sorted(glob(os.path.join(pretrainM_root, "**", "*.flac"), recursive=True))
print("Found FLAC files:", len(pretrainM_flac_files))
print(pretrainM_flac_files)

Pre_transcripts = {}
for trans_path in glob(os.path.join(pretrainM_root, "**", "*.trans.txt"), recursive=True):
    with open(trans_path, "r") as f:
        for line in f:
            fname, *text = line.strip().split()
            Pre_transcripts[fname] = " ".join(text)

print("Loaded transcripts:", len(Pre_transcripts))
print(Pre_transcripts)

missing = []
for f in pretrainM_flac_files:
    name = f.split('/')[-1].replace('.flac', '')
    if name not in Pre_transcripts:
        missing.append(name)

if missing:
    print("Missing transcripts for:", missing)
else:
    print("All audio files have transcripts")

with open("training.json", "w") as fout:
    for fname in (train_files):

        utt_id = os.path.basename(fname).replace(".flac", "")

        if utt_id not in Pre_transcripts:
            print("Skipping", utt_id)
            continue

        #duration = librosa.get_duration(path=audio_path)
        info = sf.info(fname)
        duration = info.frames / info.samplerate
        entry = {
            "audio_filepath": fname,
            "duration": round(duration, 3),
            "text": Pre_transcripts[utt_id]
        }

        fout.write(json.dumps(entry) + "\n")

with open("val.json", "w") as fout:
    for fname in (val_files):

        utt_id = os.path.basename(fname).replace(".flac", "")

        if utt_id not in Pre_transcripts:
            print("Skipping", utt_id)
            continue

        #duration = librosa.get_duration(path=audio_path)
        info = sf.info(fname)
        duration = info.frames / info.samplerate
        entry = {
            "audio_filepath": fname,
            "duration": round(duration, 3),
            "text": Pre_transcripts[utt_id]
        }

        fout.write(json.dumps(entry) + "\n")

# Training
model.setup_training_data(
    train_data_config={
        "manifest_filepath": "training.json",
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 2
    }
)

# Validation
model.setup_validation_data(
    val_data_config={
        "manifest_filepath": "val.json",
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 2
    }
)

model.setup_optimization(
    optim_config={
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "betas": [0.9, 0.98],
    }
)

from lightning.pytorch import Trainer

trainer=Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=1.0,
    precision=16,
    log_every_n_steps=10
)

model.set_trainer(trainer)

trainer.fit(model)
model.save_to(save_model)

from nemo.collections.asr.models import EncDecCTCModelBPE

model = EncDecCTCModelBPE.restore_from(save_model)
model.eval()

pred=model.transcribe([test_audio])
print(pred[0])

