import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BimaOCRDataset, ctc_collate_fn
from model import CRNN
from decoder import greedy_decoder, levenshtein_distance

def train_crnn(epochs=25, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Device: {device}")
    
    if not os.path.exists("data_crnn/train.csv"):
        import subprocess
        subprocess.run(["python", "01_generate_synthetic_words.py"], check=True)
        
    charset = ["<blank>"] + sorted(list("abcdefghijklmnopqrstuvwxyz"))
    char2idx = {c: i for i, c in enumerate(charset)}
    idx2char = {i: c for i, c in enumerate(charset)}
    
    with open("charset.json", "w") as f:
        json.dump(charset, f, indent=2)
    
    train_dataset = BimaOCRDataset("data_crnn/train.csv", "data_crnn/images", char2idx)
    val_dataset   = BimaOCRDataset("data_crnn/val.csv", "data_crnn/images", char2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)
    
    print(f"Data Latih: {len(train_dataset)} sampel | Data Validasi: {len(val_dataset)} sampel")
    
    model = CRNN(num_classes=len(charset), hidden_size=256).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_cer = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for images, flat_targets, in_lens, tgt_lens, _ in train_loader:
            images = images.to(device)
            flat_targets = flat_targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)
            
            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, flat_targets, in_lens, tgt_lens)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                
        model.eval()
        total_dist = 0
        total_ref_len = 0
        with torch.no_grad():
            for images, _, in_lens, _, ref_texts in val_loader:
                images = images.to(device)
                log_probs = model(images)
                preds = greedy_decoder(log_probs, idx2char, in_lens)
                for pred, ref in zip(preds, ref_texts):
                    total_dist += levenshtein_distance(pred, ref)
                    total_ref_len += max(1, len(ref))
                    
        cer = total_dist / total_ref_len
        avg_loss = total_loss / max(1, len(train_loader))
        scheduler.step(cer)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_loss:.4f} | Val CER: {cer * 100:.2f}%")
        
        if cer < best_cer:
            best_cer = cer
            torch.save(model.state_dict(), "best_crnn.pt")
            print(f"  -> Checkpoint disimpan (CER: {cer * 100:.2f}%)")
            
    print(f"\nPelatihan Selesai. Model disimpan di 'best_crnn.pt' (CER: {best_cer * 100:.2f}%).")

if __name__ == "__main__":
    train_crnn(epochs=20, batch_size=32, lr=1e-3)
