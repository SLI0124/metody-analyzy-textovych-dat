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
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Initialize weights with small random values
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.output_weights = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.output_bias = np.zeros(vocab_size)

    def forward(self, context_indices):
        # Get embeddings for context words
        context_embeddings = self.embedding_matrix[context_indices]

        # Average the context embeddings
        mean_embedding = np.mean(context_embeddings, axis=0)

        # Linear transformation
        z = np.dot(self.output_weights, mean_embedding) + self.output_bias

        # Softmax
        exp_z = np.exp(z - np.max(z))  # For numerical stability
        probs = exp_z / np.sum(exp_z)

        return probs, mean_embedding

    def backward(self, context_indices, target_index, probs, mean_embedding, learning_rate):
        # Calculate gradient of loss with respect to z
        dz = probs.copy()
        dz[target_index] -= 1

        # Update output weights and bias
        dw_output = np.outer(dz, mean_embedding)
        db_output = dz.copy()

        # Gradient of loss with respect to mean embedding
        d_mean_embedding = np.dot(self.output_weights.T, dz)

        # Distribute gradient equally to all context words
        d_context = d_mean_embedding / len(context_indices)

        # Update embedding matrix
        for idx in context_indices:
            self.embedding_matrix[idx] -= learning_rate * d_context

        self.output_weights -= learning_rate * dw_output
        self.output_bias -= learning_rate * db_output

    def train(self, training_data, epochs=5, learning_rate=0.01, batch_size=128):
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)

            # Process in batches
            # Vylepšený tqdm s výpisem ztráty během tréninku
            progress_bar = tqdm(range(0, len(training_data), batch_size),
                                desc=f"Epoch {epoch + 1}/{epochs}")

            for i in progress_bar:
                batch = training_data[i:i + batch_size]

                # Vectorized forward pass for whole batch
                contexts = np.array([x[0] for x in batch])
                targets = np.array([x[1] for x in batch])

                # Get embeddings for all contexts
                context_embeddings = self.embedding_matrix[contexts]  # [batch_size, window_size*2, emb_dim]

                # Average embeddings
                mean_embeddings = np.mean(context_embeddings, axis=1)  # [batch_size, emb_dim]

                # Calculate scores
                scores = np.dot(mean_embeddings, self.output_weights.T) + self.output_bias  # [batch_size, vocab_size]

                # Softmax
                max_scores = np.max(scores, axis=1, keepdims=True)
                exp_scores = np.exp(scores - max_scores)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Calculate loss
                batch_loss = -np.sum(np.log(probs[np.arange(len(batch)), targets] + 1e-10)) / len(batch)
                total_loss += batch_loss * len(batch)

                # Aktualizace tqdm s aktuální hodnotou ztráty
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

                # Backward pass
                dz = probs.copy()
                dz[np.arange(len(batch)), targets] -= 1

                # Gradients for output weights and bias
                dw_output = np.dot(dz.T, mean_embeddings) / len(batch)
                db_output = np.mean(dz, axis=0)

                # Gradient for embeddings
                d_mean_embedding = np.dot(dz, self.output_weights) / len(batch)

                # Update parameters
                self.output_weights -= learning_rate * dw_output
                self.output_bias -= learning_rate * db_output

                # Update embeddings (this part is trickier to vectorize completely)
                for j, context in enumerate(contexts):
                    d_context = d_mean_embedding[j] / len(context)
                    self.embedding_matrix[context] -= learning_rate * d_context

            avg_epoch_loss = total_loss / len(training_data)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Average loss: {avg_epoch_loss:.4f}")

        return losses

    def get_embeddings(self):
        return self.embedding_matrix


def train_cbow_model_from_scratch(training_data, vocab_size, embedding_dim=100,
                                  learning_rate=0.01, epochs=5, batch_size=128):
    print("Training CBOW model from scratch...")
    model = CBOW_Scratch(vocab_size, embedding_dim)
    losses = model.train(training_data, epochs, learning_rate, batch_size)
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
                if token and token.isalpha():
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
    context_tensor = torch.tensor([item[0] for item in cbow_data], dtype=torch.long)
    target_tensor = torch.tensor([item[1] for item in cbow_data], dtype=torch.long)

    dataset = TensorDataset(context_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class CBOW_PyTorch(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_PyTorch, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_indices):
        # Získání embedingů pro kontextová slova
        context_embeddings = self.embeddings(context_indices)
        # Průměrování embeddingů kontextu
        mean_embeddings = torch.mean(context_embeddings, dim=1)
        # Lineární transformace a predikce
        out = self.linear(mean_embeddings)
        return out


def train_cbow_model(dataloader, vocab_size, embedding_dim=100, learning_rate=0.01,
                     epochs=5, output_dir=None):
    print(f"Using device: {DEVICE}")
    model = CBOW_PyTorch(vocab_size, embedding_dim).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
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
        print(f"Epoch {epoch + 1}/{epochs}, Average loss: {avg_loss:.4f}")

    print("Training completed.")

    embeddings = model.embeddings.weight.data.cpu().numpy()

    if output_dir:
        save_model(model, output_dir)
        save_embeddings(embeddings, output_dir)

    return model, embeddings


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
        model.load_state_dict(torch.load(model_path))
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

    other_indices = [i for i in range(embeddings.shape[0]) if i != word_idx]
    other_vecs = embeddings[other_indices]

    similarities = cosine_similarity(word_vec, other_vecs)[0]

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
    random_indices = random.sample(range(vocab_size), min(sample_count, vocab_size))
    selected_embeddings = embeddings[random_indices]

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
    for i, idx in enumerate(random_indices):
        word = idx_to_word[idx]
        texts.append(plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=10))

    adjust_text(texts,
                arrowprops=dict(arrowstyle='-', color='black', lw=0.7),
                force_text=0.7,
                force_points=0.7,
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2))

    plt.title("t-SNE visualization of randomly selected embeddings", fontsize=14)
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

    # Model parameters
    vocab_size = 10_000
    window_size = 2
    embedding_dim = 100
    learning_rate = 0.01
    epochs = 5
    batch_size = 256
    visualize_sample_count = 750

    # Load and preprocess data from Hugging Face dataset
    tokenized_texts, all_tokens = load_and_preprocess_data(
        dataset_name="wikimedia/wikipedia",
        dataset_config="20231101.cs",
        split="train[:1%]",
        use_nltk=True
    )

    # Check if vocabulary exists
    vocab_path = output_dir / "vocab.txt"
    if vocab_path.exists():
        print(f"Vocabulary file found at {vocab_path}")
        word_to_idx, idx_to_word = load_vocabulary(output_dir)
    else:
        print("Vocabulary file not found, creating new vocabulary...")
        # Create vocabulary
        word_to_idx, idx_to_word, counts = build_vocabulary(all_tokens, vocab_size)
        # Save vocabulary
        save_vocabulary(word_to_idx, output_dir)

    # Create CBOW training data
    cbow_data = create_cbow_training_data(all_tokens, word_to_idx, window_size)

    # Prepare dataloader for PyTorch implementation
    dataloader = prepare_training_tensors(cbow_data, batch_size)

    # Train both models
    print("\n=== Training PyTorch Implementation (Variant A) ===")
    model_pytorch, embeddings_pytorch = train_cbow_model(
        dataloader=dataloader,
        vocab_size=len(word_to_idx),
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        output_dir=output_dir / "pytorch"
    )

    print("\n=== Training From-Scratch Implementation (Variant B) ===")
    model_scratch, embeddings_scratch, losses = train_cbow_model_from_scratch(
        training_data=cbow_data,
        vocab_size=len(word_to_idx),
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    save_embeddings(embeddings_scratch, output_dir / "scratch")

    # Evaluate both models
    test_words = ["muž", "žena", "král", "královna", "praha", "řeka", "pes", "kočka", "škola", "auto"]

    print("\n=== PyTorch Implementation Results ===")
    neighbors_pytorch = find_nearest_neighbors_batch(test_words, word_to_idx, idx_to_word, embeddings_pytorch)
    for word, neighbors in neighbors_pytorch.items():
        if isinstance(neighbors, str):
            print(neighbors)
        else:
            neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
            print(f"Nearest to '{word}': {neighbor_str}")

    print("\n=== From-Scratch Implementation Results ===")
    neighbors_scratch = find_nearest_neighbors_batch(test_words, word_to_idx, idx_to_word, embeddings_scratch)
    for word, neighbors in neighbors_scratch.items():
        if isinstance(neighbors, str):
            print(neighbors)
        else:
            neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
            print(f"Nearest to '{word}': {neighbor_str}")

    # Visualize embeddings from both models
    print("\nVisualizing PyTorch embeddings...")
    visualize_embeddings(embeddings_pytorch, idx_to_word, output_dir / "pytorch", visualize_sample_count)

    print("\nVisualizing From-Scratch embeddings...")
    visualize_embeddings(embeddings_scratch, idx_to_word, output_dir / "scratch", visualize_sample_count)

    # Compare training losses
    print("\n=== Training Loss Comparison ===")
    # For PyTorch, you would need to modify train_cbow_model to return losses
    # Here we just show the from-scratch losses
    print("From-Scratch Implementation Losses:")
    for epoch, loss in enumerate(losses):
        print(f"Epoch {epoch + 1}: {loss:.4f}")


if __name__ == "__main__":
    main()
