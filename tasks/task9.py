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

# Training parameters
total_epochs = 25
batch_size = 16


def load_data():
    dataset = load_dataset("samsum")
    train_data = dataset["train"]
    valid_data = dataset["validation"]
    test_data = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    return dataset, train_data, valid_data, test_data, tokenizer, start_token_id


def preprocess(example, tokenizer):
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
    def __init__(self, data, tokenizer):
        self.data = [preprocess(x, tokenizer) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v for k, v in self.data[idx].items()}


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


def summarize_greedy(model, dialogue, tokenizer, start_token_id, device, max_len=max_output_len):
    """Generuje shrnutí pomocí greedy decodingu (vybírá token s nejvyšší pravděpodobností)."""
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(
            dialogue,
            return_tensors="pt",
            max_length=max_input_len,
            truncation=True,
            padding="max_length"
        )["input_ids"].to(device)

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


def summarize_beam(model, dialogue, tokenizer, start_token_id, device, beam_width=3, max_len=max_output_len):
    """Generuje shrnutí pomocí beam search algoritmu."""
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(
            dialogue,
            return_tensors="pt",
            max_length=max_input_len,
            truncation=True,
            padding="max_length"
        )["input_ids"].to(device)

        src_mask = create_pad_mask(input_ids, tokenizer.pad_token_id).bool()

        # Inicializace prvního tokenu
        generated = torch.full((1, 1), start_token_id, dtype=torch.long, device=device)

        # Inicializace beam search
        # Každý paprsek má tvar [sekvence, log_prob]
        beams = [(generated, 0.0)]
        finished_beams = []

        # Beam search
        for _ in range(max_len):
            candidates = []

            # Pro každý současný paprsek
            for seq, score in beams:
                if seq[0, -1].item() == tokenizer.eos_token_id:
                    # Pokud sekvence končí EOS, přidáme ji mezi dokončené
                    # Normalizace skóre podle délky
                    normalized_score = score / (seq.size(1) - 1)  # -1 kvůli start tokenu
                    finished_beams.append((seq, normalized_score))
                    continue

                # Předpovíme další token
                tgt_mask = generate_square_subsequent_mask(seq.size(1)).to(device)
                out = model(
                    input_ids, seq,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=create_pad_mask(seq, tokenizer.pad_token_id).bool(),
                    tgt_mask=tgt_mask
                )

                # Získáme logity pro poslední token
                logits = out[:, -1, :]

                # Aplikujeme log_softmax pro získání log pravděpodobností
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Získáme top_k nejpravděpodobnějších tokenů
                topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)

                # Pro každý token v top_k
                for i in range(beam_width):
                    token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    token_log_prob = topk_log_probs[0, i].item()

                    # Vytvoříme novou sekvenci přidáním tokenu
                    new_seq = torch.cat([seq, token_id], dim=1)

                    # Aktualizujeme skóre (součet log pravděpodobností)
                    new_score = score + token_log_prob

                    # Přidáme nového kandidáta
                    candidates.append((new_seq, new_score))

            # Pokud všechny paprsky skončily, ukončíme generování
            if not candidates:
                break

            # Seřadíme kandidáty podle skóre
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Vybereme top beam_width kandidátů jako nové paprsky
            beams = candidates[:beam_width]

        # Přidáme nedokončené paprsky mezi dokončené
        for seq, score in beams:
            # Normalizace skóre podle délky
            normalized_score = score / (seq.size(1) - 1)  # -1 kvůli start tokenu
            finished_beams.append((seq, normalized_score))

        # Seřadíme dokončené paprsky podle skóre
        finished_beams.sort(key=lambda x: x[1], reverse=True)

        # Vybereme nejlepší sekvenci
        best_seq = finished_beams[0][0] if finished_beams else generated

        # Dekódujeme ji
        summary = tokenizer.decode(best_seq[0, 1:], skip_special_tokens=True)
        return summary


def summarize(model, dialogue, tokenizer, start_token_id, device, method="greedy", beam_width=3,
              max_len=max_output_len):
    """Wrapper funkce pro generování shrnutí pomocí různých metod."""
    if method == "greedy":
        return summarize_greedy(model, dialogue, tokenizer, start_token_id, device, max_len)
    elif method == "beam":
        return summarize_beam(model, dialogue, tokenizer, start_token_id, device, beam_width, max_len)
    else:
        raise ValueError(f"Neznámá metoda dekódování: {method}")


def train_epoch(model, device, train_loader, optimizer, criterion, tokenizer, vocab_size, epoch, total_epochs):
    model.train()
    epoch_loss = 0
    batch_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Training]")
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

    return epoch_loss / len(train_loader)


def validate(model, device, valid_loader, criterion, tokenizer, start_token_id, vocab_size, epoch, total_epochs,
             decoding_method="greedy", beam_width=3):
    model.eval()
    val_loss = 0
    val_rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Validating]", leave=False)
        for batch in pbar:
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
            pbar.set_postfix(loss=loss.item())

            # ROUGE evaluation
            for i in range(src.size(0)):
                dialogue = tokenizer.decode(src[i], skip_special_tokens=True)
                reference = tokenizer.decode(tgt_output[i], skip_special_tokens=True)
                prediction = summarize(model, dialogue, tokenizer, start_token_id, device, method=decoding_method,
                                       beam_width=beam_width)
                score = scorer.score(reference, prediction)
                val_rouge_scores.append(score)

    val_loss /= len(valid_loader)
    rouge1_f = np.mean([s["rouge1"].fmeasure for s in val_rouge_scores])

    return val_loss, rouge1_f


def train(model, device, train_loader, valid_loader, tokenizer, start_token_id, vocab_size, epochs,
          decoding_method="greedy", beam_width=3):
    output_dir = os.path.join("..", "output", "task9")
    os.makedirs(output_dir, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_rouge1 = float('-inf')
    best_model_path = os.path.join(output_dir, f"transformer_summarizer_best_{decoding_method}.pth")

    for epoch in range(epochs):
        # Trénovací fáze
        avg_loss = train_epoch(model, device, train_loader, optimizer, criterion, tokenizer, vocab_size, epoch, epochs)

        # Validační fáze
        val_loss, rouge1_f = validate(model, device, valid_loader, criterion, tokenizer, start_token_id, vocab_size,
                                      epoch, epochs, decoding_method, beam_width)

        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_loss:.4f}, Val loss: {val_loss:.4f}, "
              f"ROUGE-1: {rouge1_f:.4f}")

        # Uložení nejlepšího modelu podle ROUGE-1
        if rouge1_f > best_rouge1:
            best_rouge1 = rouge1_f
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (val_rouge1={rouge1_f:.4f})")

    # Uložení posledního modelu
    model_save_path = os.path.join(output_dir, f"transformer_summarizer_last_{decoding_method}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model


def evaluate(model, test_data, tokenizer, start_token_id, device, decoding_method="greedy", beam_width=3,
             num_examples=20, display=10):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = []

    for i in tqdm(range(num_examples), desc=f"Evaluating model (method={decoding_method})", leave=False):
        dialog = test_data[i]["dialogue"]
        ref = test_data[i]["summary"]
        pred = summarize(model, dialog, tokenizer, start_token_id, device, method=decoding_method,
                         beam_width=beam_width)
        score = scorer.score(ref, pred)
        scores.append(score)

    rouge1 = np.mean([s["rouge1"].fmeasure for s in scores])
    rouge2 = np.mean([s["rouge2"].fmeasure for s in scores])
    print(f"Výsledky ROUGE na testovacích datech (metoda: {decoding_method}):")
    print(f"ROUGE-1: {rouge1:.3f}, ROUGE-2: {rouge2:.3f}")

    print(f"\n--- Ukázky predikcí modelu (metoda: {decoding_method}) ---")
    for i in range(display):
        dialog = test_data[i]["dialogue"]
        ref = test_data[i]["summary"]
        pred = summarize(model, dialog, tokenizer, start_token_id, device, method=decoding_method,
                         beam_width=beam_width)
        print(f"\nDialog:\n{dialog}\n")
        print(f"Reference:\n{ref}\n")
        print(f"Predikce ({decoding_method}):\n{pred}\n")


def main():
    # Inicializace a načtení dat
    dataset, train_data, valid_data, test_data, tokenizer, start_token_id = load_data()

    # Příprava datasetů a dataloaderů
    train_dataset = SamsumDataset(train_data, tokenizer)
    valid_dataset = SamsumDataset(valid_data, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.vocab_size

    # Inicializace modelu
    model = TransformerSummarizer(vocab_size).to(device)

    # Příprava dataloaderů
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Nastavení dekódování
    decoding_method = "greedy"  # Možnosti: "greedy" nebo "beam"
    # decoding_method = "beam"
    beam_width = 3

    # Trénování nebo načtení modelu
    output_dir = os.path.join("..", "output", "task9")
    model_path = os.path.join(output_dir, f"transformer_summarizer_best_{decoding_method}.pth")
    if os.path.exists(model_path):
        print(f"Načítám existující model z {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Trénuji nový model (metoda dekódování: {decoding_method})...")
        model = train(model, device, train_loader, valid_loader, tokenizer, start_token_id, vocab_size,
                      total_epochs, decoding_method, beam_width)

    # Vyhodnocení modelu
    print("\n\033[94m--- Vyhodnocení pomocí greedy search ---\033[0m")
    evaluate(model, test_data, tokenizer, start_token_id, device,
             decoding_method="greedy")

    print("\n\033[94m--- Vyhodnocení pomocí beam search ---\033[0m")
    evaluate(model, test_data, tokenizer, start_token_id, device,
             decoding_method="beam", beam_width=beam_width)


if __name__ == "__main__":
    main()
