from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from adjustText import adjust_text
import random
import nltk
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBOW_Scratch:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Initialize weights with small random values
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.output_weights = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.output_bias = np.zeros(vocab_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_negative_words(self, target_idx, num_negatives=5):
        negative_indices = []
        while len(negative_indices) < num_negatives:
            idx = np.random.randint(0, self.vocab_size)
            if idx != target_idx and idx not in negative_indices:
                negative_indices.append(idx)
        return negative_indices

    def train_step_negative(self, context_indices, target_idx, num_negatives=5):
        # Get mean embedding for context
        context_embeddings = self.embedding_matrix[context_indices]
        mean_embedding = np.mean(context_embeddings, axis=0)

        # Calculate score for target word
        target_score = np.dot(mean_embedding,
                              self.output_weights[target_idx]) + self.output_bias[target_idx]
        target_sigmoid = self.sigmoid(target_score)

        # Sample negative words
        negative_indices = self.sample_negative_words(target_idx, num_negatives)

        # Calculate scores for negative words
        negative_scores = np.array([np.dot(mean_embedding, self.output_weights[neg_idx]) +
                                    self.output_bias[neg_idx] for neg_idx in negative_indices])
        negative_sigmoid = self.sigmoid(-negative_scores)  # Negative samples should have low scores

        # Calculate loss
        loss = -np.log(target_sigmoid + 1e-10) - np.sum(np.log(negative_sigmoid + 1e-10))

        # Calculate gradients
        # For target word
        d_target = target_sigmoid - 1  # target should be 1
        d_output_weights_target = d_target * mean_embedding
        d_output_bias_target = d_target

        # For negative words
        d_negative = 1 - negative_sigmoid  # negative should be 0

        # Gradient for mean embedding
        d_mean_embedding = d_target * self.output_weights[target_idx]
        for k, neg_idx in enumerate(negative_indices):
            d_mean_embedding += d_negative[k] * self.output_weights[neg_idx]

        # Update weights
        # Update target word weights
        self.output_weights[target_idx] -= self.learning_rate * d_output_weights_target
        self.output_bias[target_idx] -= self.learning_rate * d_output_bias_target

        # Update negative word weights
        for k, neg_idx in enumerate(negative_indices):
            d_output_weight_neg = d_negative[k] * mean_embedding
            d_output_bias_neg = d_negative[k]
            self.output_weights[neg_idx] -= self.learning_rate * d_output_weight_neg
            self.output_bias[neg_idx] -= self.learning_rate * d_output_bias_neg

        # Update embeddings for context words
        for idx in context_indices:
            self.embedding_matrix[idx] -= self.learning_rate * d_mean_embedding / len(context_indices)

        return loss

    def get_embeddings(self):
        return self.embedding_matrix


def train_cbow_model_from_scratch(training_data, vocab_size, embedding_dim=100,
                                  learning_rate=0.01, epochs=5, batch_size=128, num_negatives=20):
    print("Training CBOW model from scratch using negative sampling...")
    model = CBOW_Scratch(vocab_size, embedding_dim, learning_rate)

    # Implement custom training loop using train_step_negative
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)

        progress_bar = tqdm(range(0, len(training_data), batch_size),
                            desc=f"Epoch {epoch + 1}/{epochs}")

        for i in progress_bar:
            batch = training_data[i:i + batch_size]
            batch_loss = 0

            for context_indices, target_idx in batch:
                loss = model.train_step_negative(context_indices, target_idx, num_negatives)
                batch_loss += loss

            avg_batch_loss = batch_loss / len(batch)
            total_loss += batch_loss
            progress_bar.set_postfix({'loss': f"{avg_batch_loss:.4f}"})

        avg_epoch_loss = total_loss / len(training_data)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Average loss: {avg_epoch_loss:.4f}")

    embeddings = model.get_embeddings()
    return model, embeddings, losses


def tokenize_text(text):
    return re.findall(r"[a-záčďéěíňóřšťúůýž]+", text.lower(), re.UNICODE)


def clean_token(token):
    return re.sub(r"[^a-zA-Zá-žÁ-Ž]", "", token)


def load_and_preprocess_data(dataset_name="wikimedia/wikipedia", dataset_config="20231101.cs", split="train[:1%]",
                             use_nltk=False):
    print(f"Loading dataset {dataset_name} ({dataset_config}, {split})...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    all_texts = []
    for item in tqdm(dataset, desc="Text extraction"):
        all_texts.append(item['text'])

    tokenized_texts = []
    all_tokens = []

    if use_nltk:
        print("Tokenizing texts using NLTK...")
        nltk.download('punkt', quiet=True)
        for text in tqdm(all_texts, desc="NLTK tokenization"):
            tokens = nltk.word_tokenize(text, language="czech")
            cleaned_tokens = []
            for token in tokens:
                token = clean_token(token).lower()
                if token and token.isalpha() and len(token) > 2:
                    cleaned_tokens.append(token)
            tokenized_texts.append(cleaned_tokens)
            all_tokens.extend(cleaned_tokens)
    else:
        print("Tokenizing texts using custom tokenizer...")
        for text in tqdm(all_texts, desc="Tokenization"):
            tokens = tokenize_text(text)
            tokenized_texts.append(tokens)
            all_tokens.extend(tokens)

    print(f"Total processed texts: {len(all_texts)}")
    print(f"Total tokens: {len(all_tokens)}")

    return tokenized_texts, all_tokens


def build_vocabulary(tokens, vocab_size=10000):
    print(f"Creating vocabulary (max size {vocab_size})...")
    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(vocab_size - 1)]

    # Přidání speciálního tokenu pro neznámá slova
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    word_to_idx['<UNK>'] = len(vocab)
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    actual_vocab_size = len(word_to_idx)
    print(f"Vocabulary created, contains {actual_vocab_size} words")

    return word_to_idx, idx_to_word, counts


def create_cbow_training_data(tokens, word_to_idx, window_size=2):
    print(f"Creating CBOW training data (window_size={window_size})...")
    data = []
    unk_idx = word_to_idx['<UNK>']

    token_indices = [word_to_idx.get(word, unk_idx) for word in tokens]

    for i in tqdm(range(window_size, len(token_indices) - window_size), desc="Creating CBOW data"):
        context_indices = (
                token_indices[i - window_size: i] +
                token_indices[i + 1: i + 1 + window_size]
        )
        target_index = token_indices[i]
        data.append((context_indices, target_index))

    print(f"Created {len(data)} training samples")
    return data


def prepare_training_tensors(cbow_data, batch_size=128):
    print("Preparing training data for PyTorch...")

    # Check if cbow_data is empty
    if not cbow_data or len(cbow_data) == 0:
        print("ERROR: CBOW data is empty. No training samples available.")
        print("Please check that the CBOW data was properly created and loaded.")
        raise ValueError("CBOW data is empty. Cannot create DataLoader with empty dataset.")

    # Print data sample for debugging
    print(f"CBOW data contains {len(cbow_data)} samples")
    print(f"Sample data point: {cbow_data[0]}")

    context_tensor = torch.tensor([item[0] for item in cbow_data], dtype=torch.long)
    target_tensor = torch.tensor([item[1] for item in cbow_data], dtype=torch.long)

    print(f"Created tensors - context shape: {context_tensor.shape}, target shape: {target_tensor.shape}")

    dataset = TensorDataset(context_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class CBOW_PyTorch(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_PyTorch, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, context_indices):
        # Získání embeddingů pro kontextová slova
        context_embeddings = self.embeddings(context_indices)  # [batch_size, context_size, emb_dim]
        # Zprůměrování embeddingů kontextu
        mean_embeddings = torch.mean(context_embeddings, dim=1)  # [batch_size, emb_dim]
        # Finální predikce
        logits = self.output_layer(mean_embeddings)  # [batch_size, vocab_size]
        return logits

    def get_embeddings(self):
        return self.embeddings.weight.data


def train_cbow_model(dataloader, vocab_size, embedding_dim=100, learning_rate=0.01,
                     epochs=5, output_dir=None, num_negatives=10):
    # Výpis informace o použitém zařízení
    print(f"Using device: {DEVICE}")
    model = CBOW_PyTorch(vocab_size, embedding_dim).to(DEVICE)

    # Definice kritéria a optimizéru
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print(f"Starting training with negative sampling...")
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for context_batch, target_batch in progress_bar:
            # Přesun dat na GPU, pokud je dostupná
            context_batch = context_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            # Dopředný průchod
            optimizer.zero_grad()
            outputs = model(context_batch)

            # Výpočet standardní cross-entropy ztráty
            loss = criterion(outputs, target_batch)

            # Přidání negativního samplování, pokud je požadováno
            if num_negatives > 0:
                batch_size = context_batch.size(0)
                # Vytvoření náhodných negativních příkladů
                noise_targets = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
                # Zajištění, že negativní příklady jsou odlišné od cílů
                same_mask = noise_targets == target_batch
                if same_mask.any():
                    noise_targets[same_mask] = (noise_targets[same_mask] + 1) % vocab_size

                # Výpočet ztráty pro negativní příklady
                noise_loss = -torch.mean(torch.log(1 - torch.softmax(outputs, dim=1).gather(1,
                                                                                            noise_targets.unsqueeze(
                                                                                                1)).squeeze()))

                # Kombinace ztrát s váhami
                loss = loss * 0.8 + noise_loss * 0.2

            # Zpětný průchod
            loss.backward()

            # Aktualizace parametrů
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            progress_bar.set_postfix({'loss': batch_loss})

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Average loss: {avg_loss:.4f}")

    print("Training completed.")

    # Získání natrénovaných embeddingů
    embeddings = model.get_embeddings().cpu().numpy()

    # Uložení výsledků, pokud je zadaný výstupní adresář
    if output_dir:
        save_model(model, output_dir)
        save_embeddings(embeddings, output_dir)
        loss_path = Path(output_dir) / "losses.npy"
        np.save(loss_path, np.array(losses))
        print(f"Losses saved to {loss_path}")

    return model, embeddings, losses


def save_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_path = output_dir / "cbow_model.pth"
    print(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved")


def load_model(output_dir, vocab_size, embedding_dim=100):
    output_dir = Path(output_dir)
    model_path = output_dir / "cbow_model.pth"

    print(f"Loading model from {model_path}...")

    model = CBOW_PyTorch(vocab_size, embedding_dim)
    try:
        if DEVICE == "cuda" and torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model successfully loaded")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_vocabulary(word_to_idx, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    vocab_path = output_dir / "vocab.txt"
    print(f"Saving vocabulary to {vocab_path}...")

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word, idx in word_to_idx.items():
            f.write(f"{word}\t{idx}\n")

    print(f"Vocabulary saved")


def load_vocabulary(output_dir):
    output_dir = Path(output_dir)
    vocab_path = output_dir / "vocab.txt"

    print(f"Loading vocabulary from {vocab_path}...")
    word_to_idx = {}
    idx_to_word = {}

    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            idx = int(idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word

    print(f"Vocabulary loaded, contains {len(word_to_idx)} words")
    return word_to_idx, idx_to_word


def save_embeddings(embeddings, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    embeddings_path = output_dir / "embeddings.npy"
    print(f"Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved")


def load_embeddings(output_dir):
    output_dir = Path(output_dir)
    embeddings_path = output_dir / "embeddings.npy"

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Embeddings loaded, shape: {embeddings.shape}")
    return embeddings


def get_nearest_neighbors(word, word_to_idx, idx_to_word, embeddings, n=5):
    if word not in word_to_idx:
        return f"Word '{word}' not in vocabulary."

    word_idx = word_to_idx[word]
    word_vec = embeddings[word_idx].reshape(1, -1)

    # Normalizace vektoru slova pro lepší výpočet podobnosti
    word_vec_norm = word_vec / np.linalg.norm(word_vec)

    # Získání všech ostatních vektorů kromě dotazovaného slova
    other_indices = [i for i in range(embeddings.shape[0]) if i != word_idx]
    other_vecs = embeddings[other_indices]

    # Normalizace všech vektorů pro lepší výpočet kosinové podobnosti
    other_vecs_norm = other_vecs / np.linalg.norm(other_vecs, axis=1, keepdims=True)

    # Výpočet kosinové podobnosti
    similarities = cosine_similarity(word_vec_norm, other_vecs_norm)[0]

    sorted_indices = np.argsort(similarities)[::-1]
    neighbors = [(idx_to_word[other_indices[i]], similarities[i]) for i in sorted_indices[:n]]

    return neighbors


def find_nearest_neighbors_batch(test_words, word_to_idx, idx_to_word, embeddings, n=5):
    results = {}
    for word in test_words:
        neighbors = get_nearest_neighbors(word, word_to_idx, idx_to_word, embeddings, n)
        results[word] = neighbors
    return results


def visualize_embeddings(embeddings, idx_to_word, output_dir=None, sample_count=750):
    sns.set_theme(style="whitegrid")
    print("Generating t-SNE visualization of embeddings...")

    vocab_size = embeddings.shape[0]
    # Použijeme prvních sample_count indexů místo náhodného vzorkování
    selected_indices = list(range(min(sample_count, vocab_size)))
    selected_embeddings = embeddings[selected_indices]

    # Normalizace embeddingů pro lepší stabilitu t-SNE
    selected_embeddings = selected_embeddings / np.linalg.norm(selected_embeddings, axis=1, keepdims=True)

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, sample_count // 4),
                learning_rate=200, max_iter=1000, init='pca')
    embeddings_2d = tsne.fit_transform(selected_embeddings)

    embeddings_2d = (embeddings_2d - embeddings_2d.min(axis=0)) / (
            embeddings_2d.max(axis=0) - embeddings_2d.min(axis=0))
    embeddings_2d = (embeddings_2d * 2) - 1  # Scaling to [-1, 1]

    plt.figure(figsize=(20, 20))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=15)

    texts = []
    for i, idx in enumerate(selected_indices):
        word = idx_to_word[idx]
        texts.append(plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=10))

    adjust_text(texts,
                arrowprops=dict(arrowstyle='-', color='black', lw=0.7),
                force_text=0.7,
                force_points=0.7,
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2))

    plt.title("t-SNE visualization of first " + str(len(selected_indices)) + " embeddings", fontsize=14)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        viz_path = output_dir / "word_embeddings.png"
        plt.savefig(viz_path, dpi=300)
        print(f"Visualization saved to {viz_path}")

    plt.show()


def main():
    output_dir = Path("../output/task8")
    output_dir.mkdir(exist_ok=True, parents=True)
    pytorch_dir = output_dir / "pytorch"
    pytorch_dir.mkdir(exist_ok=True, parents=True)
    scratch_dir = output_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True, parents=True)

    # Model parameters
    vocab_size = 10_000
    window_size = 2
    embedding_dim = 100
    learning_rate = 0.01
    epochs = 10
    batch_size = 256
    visualize_sample_count = 750

    # 1. Kontrola/vytvoření dat a slovníku
    all_tokens = []
    vocab_path = output_dir / "vocab.txt"
    if vocab_path.exists():
        print(f"Vocabulary file found at {vocab_path}")
        word_to_idx, idx_to_word = load_vocabulary(output_dir)
    else:
        print("Vocabulary file not found, creating new vocabulary...")
        # Načtení a předzpracování dat
        tokenized_texts, all_tokens = load_and_preprocess_data(
            dataset_name="wikimedia/wikipedia",
            dataset_config="20231101.cs",
            split="train[:1%]",
            use_nltk=True
        )
        # Vytvoření slovníku
        word_to_idx, idx_to_word, counts = build_vocabulary(all_tokens, vocab_size)
        # Uložení slovníku
        save_vocabulary(word_to_idx, output_dir)

    # 2. Always create CBOW training data from scratch
    print("Creating CBOW training data from scratch...")
    if not all_tokens:
        print("Loading data for CBOW creation...")
        tokenized_texts, all_tokens = load_and_preprocess_data(
            dataset_name="wikimedia/wikipedia",
            dataset_config="20231101.cs",
            split="train[:1%]",
            use_nltk=True
        )

    # Create CBOW training data
    cbow_data = create_cbow_training_data(all_tokens, word_to_idx, window_size)

    if not cbow_data or len(cbow_data) == 0:
        print("ERROR: Failed to create CBOW data. Exiting.")
        return

    # 3. Příprava PyTorch data loaderu
    dataloader = prepare_training_tensors(cbow_data, batch_size)

    # 4. PyTorch model - Varianta A
    print("\n=== PyTorch Implementation (Variant A) ===")

    # Kontrola existence PyTorch modelu
    pytorch_model_path = pytorch_dir / "cbow_model.pth"
    pytorch_embeddings_path = pytorch_dir / "embeddings.npy"
    pytorch_losses_path = pytorch_dir / "losses.npy"

    if pytorch_model_path.exists() and pytorch_embeddings_path.exists() and pytorch_losses_path.exists():
        print(f"PyTorch model components found, loading from {pytorch_dir}")
        # Model existuje, načteme ho
        model_pytorch = load_model(pytorch_dir, len(word_to_idx), embedding_dim)
        embeddings_pytorch = load_embeddings(pytorch_dir)
        pytorch_losses = np.load(pytorch_losses_path)
        print(f"PyTorch model loaded, embedding shape: {embeddings_pytorch.shape}")
    else:
        print(f"PyTorch model components missing, training new model")
        # Trénování nového PyTorch modelu
        model_pytorch, embeddings_pytorch, pytorch_losses = train_cbow_model(
            dataloader=dataloader,
            vocab_size=len(word_to_idx),
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            output_dir=pytorch_dir
        )

    # 5. Model od základů - Varianta B
    print("\n=== From-Scratch Implementation (Variant B) ===")

    # Kontrola existence modelu od základů
    scratch_embeddings_path = scratch_dir / "embeddings.npy"
    scratch_losses_path = scratch_dir / "losses.npy"

    if scratch_embeddings_path.exists() and scratch_losses_path.exists():
        print(f"From-scratch model components found, loading from {scratch_dir}")
        # Model existuje, načteme embeddingy a ztráty
        embeddings_scratch = load_embeddings(scratch_dir)
        scratch_losses = np.load(scratch_losses_path)
        print(f"From-scratch embeddings loaded, shape: {embeddings_scratch.shape}")
        # Model jen z embeddingů nemůžeme plně načíst, ale pro evaluaci stačí embeddingy
        model_scratch = None
    else:
        print(f"From-scratch model components missing, training new model")
        # Trénování nového modelu od základů
        model_scratch, embeddings_scratch, scratch_losses = train_cbow_model_from_scratch(
            training_data=cbow_data,
            vocab_size=len(word_to_idx),
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        # Uložení embeddingů a ztrát
        save_embeddings(embeddings_scratch, scratch_dir)
        np.save(scratch_dir / "losses.npy", scratch_losses)
        print(f"From-scratch model saved to {scratch_dir}")

    # 6. Evaluace a analýza modelů
    test_words = ["muž", "žena", "král", "královna", "praha", "řeka", "pes", "kočka", "škola", "auto"]

    # 6.1 Evaluace PyTorch modelu
    print("\n=== PyTorch Implementation Results ===")
    neighbors_pytorch = find_nearest_neighbors_batch(test_words, word_to_idx, idx_to_word, embeddings_pytorch)
    for word, neighbors in neighbors_pytorch.items():
        if isinstance(neighbors, str):
            print(neighbors)
        else:
            neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
            print(f"Nearest to '{word}': {neighbor_str}")

    # 6.2 Evaluace modelu od základů
    print("\n=== From-Scratch Implementation Results ===")
    neighbors_scratch = find_nearest_neighbors_batch(test_words, word_to_idx, idx_to_word, embeddings_scratch)
    for word, neighbors in neighbors_scratch.items():
        if isinstance(neighbors, str):
            print(neighbors)
        else:
            neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
            print(f"Nearest to '{word}': {neighbor_str}")

    # 7. Vizualizace embeddingů
    # 7.1 Vizualizace PyTorch embeddingů
    print("\nVisualizing PyTorch embeddings...")
    visualize_embeddings(embeddings_pytorch, idx_to_word, pytorch_dir, visualize_sample_count)

    # 7.2 Vizualizace embeddingů z modelu od základů
    print("\nVisualizing From-Scratch embeddings...")
    visualize_embeddings(embeddings_scratch, idx_to_word, scratch_dir, visualize_sample_count)

    # 8. Porovnání ztrát během tréninku
    print("\n=== Training Loss Comparison ===")
    # Poznámka: Implementace od základů používá negativní vzorkování, zatímco PyTorch implementace 
    # používá cross-entropy loss. Tento zásadní rozdíl ve výpočtu ztrát 
    # vysvětluje, proč jsou hodnoty a křivky ztrát mezi implementacemi tak odlišné.
    # 
    # Klíčové rozdíly mezi implementacemi:
    # - Scratch: Přímo používá negativní vzorkování, vybírá náhodná necílová slova jako negativní příklady
    # - PyTorch: Používá cross-entropy ztrátu, která implicitně bere v úvahu všechna slova ve slovníku
    # - Scratch: Aktualizuje váhy pomocí vlastních výpočtů gradientů a ručně optimalizovaného učení
    # - PyTorch: Používá optimalizátor Adam s automatickou diferenciací
    # - Scratch: Přímo manipuluje s embedding maticemi pomocí numpy operací
    # - PyTorch: Používá vrstvy neuronové sítě s GPU akcelerací, když je k dispozici

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), pytorch_losses, 'b-', marker='o', label='PyTorch')
    plt.plot(range(1, epochs + 1), scratch_losses, 'r-', marker='s', label='From Scratch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)

    loss_plot_path = output_dir / "loss_comparison.png"
    plt.savefig(loss_plot_path)
    print(f"Loss comparison plot saved to {loss_plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
