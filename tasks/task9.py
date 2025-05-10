from rouge_score import rouge_scorer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

# Model parameters
d_model = 384
nhead = 6
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024
dropout = 0.1

# Sequence lengths
max_input_len = 256
max_output_len = 64

dataset = load_dataset("samsum")
train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

print("BOS token id:", tokenizer.bos_token_id)
print("EOS token id:", tokenizer.eos_token_id)
start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id


def preprocess(example):
    inputs = tokenizer(
        example["dialogue"],
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    outputs = tokenizer(
        example["summary"],
        max_length=max_output_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": outputs["input_ids"].squeeze()
    }


class SamsumDataset(Dataset):
    def __init__(self, data):
        self.data = [preprocess(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v for k, v in self.data[idx].items()}


train_dataset = SamsumDataset(train_data)
valid_dataset = SamsumDataset(valid_data)
test_dataset = SamsumDataset(test_data)


class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                 num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_input_len, d_model))
        self.pos_decoder = nn.Parameter(torch.zeros(1, max_output_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        src_emb = self.embedding(src)
        src_emb = self.emb_dropout(self.emb_norm(src_emb)) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.emb_dropout(self.emb_norm(tgt_emb)) + self.pos_decoder[:, :tgt.size(1), :]
        out = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.fc_out(out)


def create_pad_mask(matrix, pad_token_id):
    return matrix == pad_token_id


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask


def summarize(dialogue, max_len=max_output_len):
    model.eval()
    with torch.no_grad():
        input_ids = \
            tokenizer(dialogue, return_tensors="pt", max_length=max_input_len, truncation=True, padding="max_length")[
                "input_ids"].to(device)
        src_mask = create_pad_mask(input_ids, tokenizer.pad_token_id).bool()
        generated = torch.full((1, 1), start_token_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(generated.size(1)).to(device)
            out = model(
                input_ids, generated,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=create_pad_mask(generated, tokenizer.pad_token_id).bool(),
                tgt_mask=tgt_mask
            )
            logits = out[:, -1, :]
            next_token = logits.argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        summary = tokenizer.decode(generated[0, 1:], skip_special_tokens=True)
        return summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
model = TransformerSummarizer(vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Vytvoření adresáře pro výstupy, pokud neexistuje
output_dir = os.path.join("..", "output", "task9")
os.makedirs(output_dir, exist_ok=True)

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
epochs = 20
best_rouge1 = 0.0
best_model_path = os.path.join(output_dir, "transformer_summarizer_best.pth")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    batch_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for i, batch in enumerate(pbar):
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = create_pad_mask(src, tokenizer.pad_token_id).bool()
        tgt_pad_mask = create_pad_mask(tgt_input, tokenizer.pad_token_id).bool()
        causal_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        logits = model(
            src, tgt_input,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=causal_mask
        )
        loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        batch_losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0
    val_rouge_scores = []
    with torch.no_grad():
        for batch in valid_loader:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask = create_pad_mask(src, tokenizer.pad_token_id).bool()
            tgt_pad_mask = create_pad_mask(tgt_input, tokenizer.pad_token_id).bool()
            causal_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            logits = model(
                src, tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                tgt_mask=causal_mask
            )
            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            val_loss += loss.item()

            # ROUGE evaluation
            for i in range(src.size(0)):
                dialogue = tokenizer.decode(src[i], skip_special_tokens=True)
                reference = tokenizer.decode(tgt_output[i], skip_special_tokens=True)
                prediction = summarize(dialogue)
                score = scorer.score(reference, prediction)
                val_rouge_scores.append(score)

    val_loss /= len(valid_loader)
    rouge1_f = np.mean([s["rouge1"].fmeasure for s in val_rouge_scores])

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation ROUGE-1: {rouge1_f:.4f}")

    # Save the best model based on ROUGE-1
    if rouge1_f > best_rouge1:
        best_rouge1 = rouge1_f
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path} (val_rouge1={rouge1_f:.4f})")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
scores = []
for i in range(20):
    dialog = test_data[i]["dialogue"]
    ref = test_data[i]["summary"]
    pred = summarize(dialog)
    score = scorer.score(ref, pred)
    scores.append(score)

rouge1 = np.mean([s["rouge1"].fmeasure for s in scores])
rouge2 = np.mean([s["rouge2"].fmeasure for s in scores])
print(f"ROUGE-1: {rouge1:.3f}, ROUGE-2: {rouge2:.3f}")

print("\n--- Ukázky predikcí modelu ---")
for i in range(10):
    dialog = test_data[i]["dialogue"]
    ref = test_data[i]["summary"]
    pred = summarize(dialog)
    print(f"\nDialog:\n{dialog}\n")
    print(f"Reference:\n{ref}\n")
    print(f"Predikce:\n{pred}\n")

# Save the trained model (poslední stav po všech epochách)
model_save_path = os.path.join(output_dir, "transformer_summarizer_last.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
