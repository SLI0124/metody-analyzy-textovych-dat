import os
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from adjustText import adjust_text
import random


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
                shutil.copyfileobj(f_in, f_out)
        print(f"Soubor byl úspěšně extrahován do {output_path}")
    except EOFError:
        print("Chyba: Komprimovaný soubor je poškozený nebo neúplný. Mažu a stahuji znovu.")
        os.remove(gzip_path)
        raise


def download_and_extract(url, input_dir, output_dir):
    """Download and extract a file if it doesn't already exist."""
    gzip_path = input_dir / "cs.txt.gz"
    output_path = input_dir / "cs.txt"

    if not gzip_path.exists():
        download_file(url, gzip_path)
    else:
        print(f"Soubor {gzip_path} již existuje, přeskakuji stahování.")

    if not output_path.exists():
        while not output_path.exists():
            try:
                extract_gzip(gzip_path, output_path)
            except EOFError:
                print("Opakuji stažení a extrakci...")
                download_file(url, gzip_path)
            else:
                break
    else:
        print(f"Soubor {output_path} již existuje, přeskakuji extrakci.")


def tokenize(text):
    return re.findall(r"[a-záčďéěíňóřšťúůýž]+", text.lower(), re.UNICODE)


def process_chunk(chunk):
    tokens = []
    for line in chunk:
        tokens.extend(tokenize(line))
    return tokens


def build_vocab(tokens, vocab_size):
    """Build a vocabulary from tokens."""
    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(vocab_size - 1)]
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    word_to_ix['<UNK>'] = len(vocab)
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return word_to_ix, ix_to_word, counts


def parallel_tokenize(lines):
    """Tokenize text in parallel using multiprocessing."""
    num_processes = multiprocessing.cpu_count()
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
    tokens = []

    if input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line for i, line in enumerate(f) if i < max_lines]

        tokens = parallel_tokenize(lines)

        print(f"Celkem tokenů: {len(tokens)}")
    else:
        print("Chyba: Extrahovaný soubor nenalezen.")
        return None, None, None

    print(f"Vytvářím slovník (velikost {vocab_size})...")
    word_to_ix, ix_to_word, word_counts = build_vocab(tokens, vocab_size)
    actual_vocab_size = len(word_to_ix)
    print(f"Slovník vytvořen, obsahuje {actual_vocab_size} slov")

    return word_to_ix, ix_to_word, tokens


def create_cbow_training_data(tokens, word_to_ix, window_size):
    """Create CBOW training data from tokens."""
    print(f"Vytvářím CBOW trénovací data (window_size={window_size})...")
    data = []
    unk_ix = word_to_ix['<UNK>']
    token_indices = [word_to_ix.get(word, unk_ix) for word in tokens]

    for i in range(window_size, len(token_indices) - window_size):
        context_indices = (
            token_indices[i - window_size: i] +
            token_indices[i + 1: i + 1 + window_size]
        )
        target_index = token_indices[i]
        data.append((context_indices, target_index))

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


def train_cbow_model(cbow_data, word_to_ix, output_dir, embedding_dim, learning_rate, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")
    vocab_size = len(word_to_ix)
    model = CBOW(vocab_size, embedding_dim).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Příprava trénovacích dat pro PyTorch...")
    context_tensor = torch.tensor([item[0] for item in cbow_data], dtype=torch.long)
    target_tensor = torch.tensor([item[1] for item in cbow_data], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(context_tensor, target_tensor)
    use_pin_memory = (device.type == 'cuda')
    num_workers = 0 if device.type == 'cuda' else min(4, multiprocessing.cpu_count())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epocha {epoch + 1}/{epochs}")
        for context_batch, target_batch in progress_bar:
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
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
    return model, embeddings


def get_nearest_neighbors(word, word_to_ix, ix_to_word, embeddings, n=5):
    if word not in word_to_ix:
        return f"Slovo '{word}' není ve slovníku."

    word_ix = word_to_ix[word]
    word_vec = embeddings[word_ix].reshape(1, -1)

    other_indices = [i for i in range(embeddings.shape[0]) if i != word_ix]
    other_vecs = embeddings[other_indices]

    similarities = cosine_similarity(word_vec, other_vecs)[0]

    sorted_indices = np.argsort(similarities)[::-1]
    neighbors = [(ix_to_word[other_indices[i]], similarities[i]) for i in sorted_indices[:n]]
    return neighbors


def parallel_nearest_neighbors(test_words, word_to_ix, ix_to_word, embeddings, n=5):
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                get_nearest_neighbors,
                word,
                word_to_ix,
                ix_to_word,
                embeddings,
                n): word for word in test_words}
        for future in as_completed(futures):
            word = futures[future]
            results[word] = future.result()
    return results


def visualize_embeddings(embeddings, ix_to_word, output_dir, vis_embeddings_count=200):
    sns.set_theme(style="whitegrid")
    print("Generuji t-SNE vizualizaci embeddingů...")
    tsne = TSNE(n_components=2, random_state=42)
    vocab_size = embeddings.shape[0]
    random_indices = random.sample(range(vocab_size), min(vis_embeddings_count, vocab_size))
    selected_embeddings = embeddings[random_indices]
    embeddings_2d = tsne.fit_transform(selected_embeddings)
    plt.figure(figsize=(16, 16))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    texts = []
    for i, idx in enumerate(random_indices):
        word = ix_to_word[idx]
        texts.append(plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=8))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    plt.title("t-SNE vizualizace náhodně vybraných embeddingů")
    plt.tight_layout()
    viz_path = output_dir / "word_embeddings.png"
    plt.savefig(viz_path, dpi=300)
    print(f"Vizualizace uložena do {viz_path}")
    plt.show()


def check_model_exists(output_dir):
    model_path = output_dir / "cbow_model.pth"
    return model_path.exists()


def check_vocab_exists(output_dir):
    vocab_path = output_dir / "vocab.txt"
    return vocab_path.exists()


def check_embeddings_exist(output_dir):
    embeddings_path = output_dir / "embeddings.npy"
    return embeddings_path.exists()


def save_vocab(word_to_ix, output_dir):
    vocab_path = output_dir / "vocab.txt"
    print(f"Ukládám slovník do {vocab_path}...")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word, ix in word_to_ix.items():
            f.write(f"{word}\t{ix}\n")
    print(f"Slovník uložen")


def load_vocab(output_dir):
    vocab_path = output_dir / "vocab.txt"
    if not vocab_path.exists():
        return None, None

    print(f"Načítám slovník z {vocab_path}...")
    word_to_ix = {}
    ix_to_word = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, ix = line.strip().split('\t')
            ix = int(ix)
            word_to_ix[word] = ix
            ix_to_word[ix] = word

    print(f"Slovník načten, obsahuje {len(word_to_ix)} slov")
    return word_to_ix, ix_to_word


def save_embeddings(embeddings, output_dir):
    embeddings_path = output_dir / "embeddings.npy"
    print(f"Ukládám embeddings do {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    print(f"Embeddings uloženy")


def load_embeddings(output_dir):
    embeddings_path = output_dir / "embeddings.npy"
    if not embeddings_path.exists():
        return None

    print(f"Načítám embeddings z {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Embeddings načteny, tvar: {embeddings.shape}")
    return embeddings


def main():
    url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2735/cs.txt.gz?sequence=54&isAllowed=y"
    input_dir = Path("../input/task8")
    output_dir = Path("../output/task8")
    vocab_size = 10000
    max_lines = 200_000
    window_size = 2
    embedding_dim = 100
    learning_rate = 0.01
    epochs = 5
    batch_size = 128
    vis_embeddings_count = 200

    input_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = None
    word_to_ix = None
    ix_to_word = None
    embeddings = None

    vocab_exists = check_vocab_exists(output_dir)
    embeddings_exist = check_embeddings_exist(output_dir)
    model_exists = check_model_exists(output_dir)

    if vocab_exists:
        word_to_ix, ix_to_word = load_vocab(output_dir)
    if embeddings_exist:
        embeddings = load_embeddings(output_dir)
    if model_exists and word_to_ix is not None:
        vocab_size = len(word_to_ix)
        embedding_dim = embeddings.shape[1] if embeddings is not None else 100
        model = load_model(output_dir, vocab_size, embedding_dim)

    if model is not None and word_to_ix is not None and embeddings is not None:
        print("Všechny komponenty již existují, přeskočeno zpracování a trénink.")
    else:
        print("Některé komponenty chybí, zahajuji zpracování dat a trénink...")
        if not vocab_exists or not embeddings_exist:
            download_and_extract(url, input_dir, output_dir)
            word_to_ix, ix_to_word, tokens = tokenize_and_build_vocab(input_dir / "cs.txt", vocab_size, max_lines)
            save_vocab(word_to_ix, output_dir)
        if not model_exists or not embeddings_exist:
            if word_to_ix is not None and ix_to_word is not None:
                cbow_data = create_cbow_training_data(tokens, word_to_ix, window_size)
                model, embeddings = train_cbow_model(
                    cbow_data, word_to_ix, output_dir, embedding_dim, learning_rate, epochs, batch_size)

    print("\n--- Vyhodnocení modelu (nejbližší sousedé) ---")
    test_words = ["muž", "žena", "král", "královna", "praha", "řeka", "pes", "kočka", "škola", "auto"]

    neighbors_results = parallel_nearest_neighbors(test_words, word_to_ix, ix_to_word, embeddings)

    for word, neighbors in neighbors_results.items():
        if isinstance(neighbors, str):
            print(neighbors)
        else:
            neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
            print(f"Nejbližší k '{word}': {neighbor_str}")

    visualize_embeddings(embeddings, ix_to_word, output_dir)


if __name__ == "__main__":
    main()
