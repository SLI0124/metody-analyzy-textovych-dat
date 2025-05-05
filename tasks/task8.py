import os
import requests
import gzip
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from adjustText import adjust_text
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMBER_OF_WORKERS = multiprocessing.cpu_count() - 1


def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    print(f"Stahuji soubor z {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=Path(local_path).name) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Soubor stažen do {local_path}")


def extract_gzip(gzip_path, output_path):
    """Extract a gzip file to a specified output path."""
    print(f"Extrahuji {gzip_path} do {output_path}...")
    try:
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                chunk_size = 8192
                chunk = f_in.read(chunk_size)
                while chunk:
                    f_out.write(chunk)
                    chunk = f_in.read(chunk_size)
        print(f"Soubor byl úspěšně extrahován do {output_path}")
    except EOFError:
        print(f"Chyba: Soubor {gzip_path} je poškozený nebo neúplný. Mazání souboru.")
        os.remove(gzip_path)
        raise EOFError("Soubor je poškozený nebo neúplný.")


def tokenize(text):
    return re.findall(r"[a-záčďéěíňóřšťúůýž]+", text.lower(), re.UNICODE)


def process_chunk(chunk):
    """Helper function to tokenize a chunk of input text."""
    tokens = []
    for line in chunk:
        tokens.extend(tokenize(line))
    return tokens


def parallel_tokenize(lines):
    """Tokenize text in parallel using multiprocessing."""
    num_processes = NUMBER_OF_WORKERS
    chunk_size = max(1, len(lines) // num_processes)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    print(f"Paralelní zpracování textu pomocí {num_processes} procesů...")
    tokens = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizace chunků"):
            tokens.extend(future.result())

    return tokens


def tokenize_and_build_vocab(input_path, vocab_size, max_lines):
    """Tokenize text and build a vocabulary."""
    print("Načítám a tokenizuji text...")

    lines = []
    line_count = 0
    f = open(input_path, 'r', encoding='utf-8')
    while line_count < max_lines:
        line = f.readline()
        if line.strip():  # skip empty lines
            lines.append(line)
            line_count += 1
    f.close()

    tokens = parallel_tokenize(lines)
    print(f"Celkem tokenů: {len(tokens)}")

    print(f"Vytvářím slovník (velikost {vocab_size})...")
    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(vocab_size - 1)]
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    word_to_idx['<UNK>'] = len(vocab)
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    actual_vocab_size = len(word_to_idx)
    print(f"Slovník vytvořen, obsahuje {actual_vocab_size} slov")

    return word_to_idx, idx_to_word, tokens, counts


def process_cbow_chunk(chunk, word_to_idx, window_size):
    data = []
    unk_idx = word_to_idx['<UNK>']
    token_indices = [word_to_idx.get(word, unk_idx) for word in chunk]

    for i in range(window_size, len(token_indices) - window_size):
        context_indices = (
                token_indices[i - window_size: i] +
                token_indices[i + 1: i + 1 + window_size]
        )
        target_index = token_indices[i]
        data.append((context_indices, target_index))

    return data


def create_cbow_training_data(tokens, word_to_idx, window_size):
    print(f"Vytvářím CBOW trénovací data (window_size={window_size}) paralelně...")
    num_processes = NUMBER_OF_WORKERS
    chunk_size = max(1, len(tokens) // num_processes)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    data = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_cbow_chunk, chunk, word_to_idx, window_size) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Zpracování CBOW chunků"):
            data.extend(future.result())

    print(f"Počet trénovacích vzorků: {len(data)}")
    return data


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_indices):
        context_embeddings = self.embeddings(context_indices)
        mean_embeddings = torch.mean(context_embeddings, dim=1)
        out = self.linear(mean_embeddings)
        return out


def save_model(model, output_dir):
    model_path = output_dir / "cbow_model.pth"
    print(f"Ukládám model do {model_path}...")
    torch.save(model.state_dict(), model_path)
    print(f"Model uložen")


def load_model(output_dir, vocab_size, embedding_dim):
    model_path = output_dir / "cbow_model.pth"
    if not model_path.exists():
        return None

    print(f"Načítám model z {model_path}...")

    model = CBOW(vocab_size, embedding_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model úspěšně načten")
        return model
    except Exception as e:
        print(f"Chyba při načítání modelu: {e}")
        return None


def train_cbow_model(cbow_data, word_to_idx, output_dir, embedding_dim, learning_rate, epochs, batch_size):
    print(f"Používám zařízení: {DEVICE}")
    vocab_size = len(word_to_idx)

    model = CBOW(vocab_size, embedding_dim).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Příprava trénovacích dat pro PyTorch...")
    context_tensor = torch.tensor([item[0] for item in cbow_data], dtype=torch.long).to(DEVICE)
    target_tensor = torch.tensor([item[1] for item in cbow_data], dtype=torch.long).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(context_tensor, target_tensor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epocha {epoch + 1}/{epochs}")
        for context_batch, target_batch in progress_bar:
            context_batch = context_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            optimizer.zero_grad()
            log_probs = model(context_batch)
            loss = loss_function(log_probs, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(dataloader)
        print(f"Epocha {epoch + 1}/{epochs}, Průměrná ztráta: {avg_loss:.4f}")

    print("Trénování dokončeno.")
    save_model(model, output_dir)
    embeddings = model.embeddings.weight.data.cpu().numpy()
    save_embeddings(embeddings, output_dir)

    return embeddings


def get_nearest_neighbors(words, word_to_idx, idx_to_word, embeddings, n=5):
    """Finds the nearest neighbors for one word or multiple words based on cosine similarity."""
    # Pokud je vstupem jedno slovo, vrať přímo výsledek pro to slovo
    if isinstance(words, str):
        word = words
        if word not in word_to_idx:
            return f"Slovo '{word}' není ve slovníku."

        word_idx = word_to_idx[word]
        word_vec = embeddings[word_idx].reshape(1, -1)

        other_indices = [i for i in range(embeddings.shape[0]) if i != word_idx]
        other_vecs = embeddings[other_indices]

        similarities = cosine_similarity(word_vec, other_vecs)[0]

        sorted_indices = np.argsort(similarities)[::-1]
        neighbors = [(idx_to_word[other_indices[i]], similarities[i]) for i in sorted_indices[:n]]
        return neighbors

    # Pokud je vstupem seznam slov, zpracuj každé slovo postupně
    results = {}
    for word in words:
        results[word] = get_nearest_neighbors(word, word_to_idx, idx_to_word, embeddings, n)
    return results


def visualize_embeddings(embeddings, idx_to_word, output_dir, vis_embeddings_count):
    sns.set_theme(style="whitegrid")
    print("Generuji t-SNE vizualizaci embeddingů...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, vis_embeddings_count // 4),
                learning_rate=200, max_iter=1000, init='pca')
    vocab_size = embeddings.shape[0]
    random_indices = random.sample(range(vocab_size), min(vis_embeddings_count, vocab_size))
    selected_embeddings = embeddings[random_indices]

    # Normalize embeddings for better t-SNE stability
    selected_embeddings = selected_embeddings / np.linalg.norm(selected_embeddings, axis=1, keepdims=True)

    embeddings_2d = tsne.fit_transform(selected_embeddings)

    # Scale to a consistent range
    embeddings_2d = (embeddings_2d - embeddings_2d.min(axis=0)) / (
            embeddings_2d.max(axis=0) - embeddings_2d.min(axis=0))
    embeddings_2d = (embeddings_2d * 2) - 1  # Scale to [-1, 1]

    plt.figure(figsize=(20, 20))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=15)

    texts = []
    for i, idx in enumerate(random_indices):
        word = idx_to_word[idx]
        texts.append(plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=10))

    # Zvýšená síla pro lepší oddělení textů
    adjust_text(texts,
                arrowprops=dict(arrowstyle='-', color='black', lw=0.7),
                force_text=0.7,
                force_points=0.7,
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2))

    plt.title("t-SNE vizualizace náhodně vybraných embeddingů", fontsize=14)
    plt.tight_layout()
    viz_path = output_dir / "word_embeddings.png"
    plt.savefig(viz_path, dpi=300)
    print(f"Vizualizace uložena do {viz_path}")
    plt.show()


def save_vocab(word_to_idx, output_dir):
    vocab_path = output_dir / "vocab.txt"
    print(f"Ukládám slovník do {vocab_path}...")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word, idx in word_to_idx.items():
            f.write(f"{word}\t{idx}\n")
    print(f"Slovník uložen")


def load_vocab(output_dir):
    vocab_path = output_dir / "vocab.txt"
    print(f"Načítám slovník z {vocab_path}...")
    word_to_idx = {}
    idx_to_word = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            idx = int(idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word

    print(f"Slovník načten, obsahuje {len(word_to_idx)} slov")
    return word_to_idx, idx_to_word


def save_embeddings(embeddings, output_dir):
    embeddings_path = output_dir / "embeddings.npy"
    print(f"Ukládám embeddings do {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    print(f"Embeddings uloženy")


def load_embeddings(output_dir):
    embeddings_path = output_dir / "embeddings.npy"
    print(f"Načítám embeddings z {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Embeddings načteny, tvar: {embeddings.shape}")
    return embeddings


def main():
    url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2735/cs.txt.gz?sequence=54&isAllowed=y"
    input_dir = Path("../input/task8")
    output_dir = Path("../output/task8")

    vocab_path = output_dir / "vocab.txt"
    embeddings_path = output_dir / "embeddings.npy"
    model_path = output_dir / "cbow_model.pth"
    input_gzip_path = input_dir / "cs.txt.gz"
    output_gzip_path = input_dir / "cs.txt"

    vocab_size = 10_000
    max_lines = 500_000
    window_size = 2
    embedding_dim = 100
    learning_rate = 0.01
    epochs = 5
    batch_size = 128
    visualize_sample_count = 750

    input_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    word_to_idx = None
    idx_to_word = None
    embeddings = None
    tokens = None

    if not os.path.exists(input_gzip_path):
        print(f"Soubor {input_gzip_path} neexistuje, stahuji...")
        download_file(url, input_gzip_path)
    else:
        print(f"Soubor {input_gzip_path} již existuje, přeskakuji stahování.")

    if not os.path.exists(output_gzip_path):
        print(f"Soubor {output_gzip_path} neexistuje, extrahuji...")
        extract_gzip(input_gzip_path, output_gzip_path)
    else:
        print(f"Soubor {output_gzip_path} již existuje, přeskakuji extrakci.")

    if os.path.exists(vocab_path):
        print(f"Soubor {vocab_path} již existuje, načítám slovník...")
        word_to_idx, idx_to_word = load_vocab(output_dir)
        # TODO: load tokens from this branch too
    else:
        print(f"Soubor {vocab_path} neexistuje, vytvářím slovník...")
        word_to_idx, idx_to_word, tokens, counts = tokenize_and_build_vocab(output_gzip_path, vocab_size, max_lines)

    if os.path.exists(embeddings_path) and os.path.exists(model_path):
        print(f"Soubor {embeddings_path} a {model_path} již existují, načítám embeddings...")
        embeddings = load_embeddings(output_dir)
    else:
        print(f"Soubor {embeddings_path} nebo {model_path} neexistují, trénuji model a vytvářím embeddings...")
        cbow_data = create_cbow_training_data(tokens, word_to_idx, window_size)
        embeddings = train_cbow_model(cbow_data, word_to_idx, output_dir, embedding_dim, learning_rate, epochs,
                                      batch_size)

    print("\n--- Vyhodnocení modelu (nejbližší sousedé) ---")
    test_words = ["muž", "žena", "král", "královna", "praha", "řeka", "pes", "kočka", "škola", "auto"]

    neighbors_results = get_nearest_neighbors(test_words, word_to_idx, idx_to_word, embeddings)

    for word, neighbors in neighbors_results.items():
        neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
        print(f"Nejbližší k '{word}': {neighbor_str}")

    visualize_embeddings(embeddings, idx_to_word, output_dir, visualize_sample_count)


if __name__ == "__main__":
    main()
