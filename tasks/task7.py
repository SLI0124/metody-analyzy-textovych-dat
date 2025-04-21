import numpy as np
from tqdm import tqdm
import os
import random
import requests
import gzip
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt


def download_file(url, file_path):
    if os.path.exists(file_path):
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192

        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Stahování {file_path}") as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def download_resources():
    input_dir = os.path.join('..', 'input', 'task7')
    os.makedirs(input_dir, exist_ok=True)

    cs_emb_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz"
    en_emb_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
    train_dict_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/cs-en.txt"
    test_dict_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/cs-en.0-5000.txt"

    cs_emb_path = os.path.join(input_dir, "cc.cs.300.vec")
    en_emb_path = os.path.join(input_dir, "cc.en.300.vec")
    train_dict_path = os.path.join(input_dir, "cs-en.train.txt")
    test_dict_path = os.path.join(input_dir, "cs-en.test.txt")

    if (os.path.exists(cs_emb_path) and os.path.exists(en_emb_path)
            and os.path.exists(train_dict_path) and os.path.exists(test_dict_path)):
        print("Všechny potřebné soubory již existují, přeskakuji stahování.")
        return cs_emb_path, en_emb_path, train_dict_path, test_dict_path

    download_file(cs_emb_url, cs_emb_path + ".gz")
    download_file(en_emb_url, en_emb_path + ".gz")
    download_file(train_dict_url, train_dict_path)
    download_file(test_dict_url, test_dict_path)

    for emb_path in [cs_emb_path, en_emb_path]:
        if not os.path.exists(emb_path) and os.path.exists(emb_path + ".gz"):
            print(f"Rozbaluji {emb_path}.gz...")
            with gzip.open(emb_path + ".gz", 'rb') as f_in:
                with open(emb_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Rozbaleno: {emb_path}")

    return cs_emb_path, en_emb_path, train_dict_path, test_dict_path


def process_chunk(chunk_lines):
    chunk_embeddings = {}
    for line in chunk_lines:
        values = line.strip().split(' ')
        word = values[0]
        vector = np.array([float(val) for val in values[1:]])
        chunk_embeddings[word] = vector

    return chunk_embeddings


def process_file_chunk(file_path, start_pos, end_pos):
    chunk_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(start_pos)
        if start_pos > 0:
            f.readline()

        while f.tell() < end_pos:
            line = f.readline()
            if not line:
                break

            values = line.strip().split(' ')
            word = values[0]
            vector = np.array([float(val) for val in values[1:]])
            chunk_embeddings[word] = vector

    return chunk_embeddings


def load_embeddings(embedding_file):
    load_start_time = time.time()
    print(f"Načítání embeddingů z {embedding_file}...")

    # Use half of available CPU cores to avoid system overload
    num_workers = max(1, multiprocessing.cpu_count() // 2)

    file_size = os.path.getsize(embedding_file)

    with open(embedding_file, 'r', encoding='utf-8') as f:
        header_end_pos = f.tell()

    chunk_size = (file_size - header_end_pos) // num_workers
    start_positions = [header_end_pos + i * chunk_size for i in range(num_workers)]
    end_positions = start_positions[1:] + [file_size]

    print(f"Zpracování embeddingů pomocí {num_workers} procesů...")
    word_to_vector = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(process_file_chunk, embedding_file, start, end): i
            for i, (start, end) in enumerate(zip(start_positions, end_positions))
        }

        with tqdm(total=num_workers, desc="Paralelní zpracování embeddingů") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_result = future.result()
                pbar.update(1)
                word_to_vector.update(chunk_result)

    print(f"Načteno {len(word_to_vector)} embeddingů.")
    load_end_time = time.time()
    print(f"Čas načítání: {load_end_time - load_start_time:.2f} sekund")
    return word_to_vector


def load_word_pairs(dict_file):
    print(f"Načítání slovníku z {dict_file}...")
    word_pairs = []
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            source, target = line.strip().split()
            word_pairs.append((source, target))
    print(f"Načteno {len(word_pairs)} párů slov.")
    return word_pairs


def create_embedding_matrices(word_pairs, source_embeddings, target_embeddings):
    source_vectors = []
    target_vectors = []
    matched_pairs = []

    for source_word, target_word in tqdm(word_pairs, desc="Vytváření matic embedů"):
        if source_word in source_embeddings and target_word in target_embeddings:
            source_vectors.append(source_embeddings[source_word])
            target_vectors.append(target_embeddings[target_word])
            matched_pairs.append((source_word, target_word))

    source_matrix = np.array(source_vectors)
    target_matrix = np.array(target_vectors)

    print(f"\nVytvořeny matice embedů s {len(matched_pairs)} páry slov.")
    return source_matrix, target_matrix, matched_pairs


def frobenius_norm_squared(X, W_t, Y):
    """ ||XW^T - Y||_F^2 """
    diff = np.matmul(X, W_t) - Y
    return np.sum(diff ** 2)


def compute_difference(X, W_t, Y):
    """ XW^T - Y """
    return np.matmul(X, W_t) - Y


def compute_gradient(X, W_t, Y):
    """
    Výpočet gradientu ztrátové funkce vůči matici W^T
    Gradient = 2 * X^T * (XW^T - Y)
    https://math.stackexchange.com/questions/2128462/gradient-of-squared-frobenius-norm-of-a-matrix
    """
    diff = compute_difference(X, W_t, Y)
    return 2 * np.matmul(X.T, diff)


def gradient_descent(X, Y, learning_rate=0.01, max_iterations=1000, tol=1e-6, patience=10):
    n_features = X.shape[1]
    W_t = np.random.randn(n_features, n_features) / np.sqrt(n_features)

    loss_history = []
    best_loss = float('inf')
    best_W_t = None
    patience_counter = 0

    print("\nZačátek trénování gradient descent...")
    pbar = tqdm(range(max_iterations), desc="Gradient Descent")
    for i in pbar:
        current_loss = frobenius_norm_squared(X, W_t, Y)
        loss_history.append(current_loss)
        gradient = compute_gradient(X, W_t, Y)

        W_t = W_t - learning_rate * gradient  # upravíme W_t na základě gradientu

        pbar.set_postfix(loss=f"{current_loss:.4f}", patience=patience_counter)

        # Ukládání nejlepšího modelu
        if current_loss < best_loss:
            best_loss = current_loss
            best_W_t = W_t.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Pokud se loss nezlepšil po 'patience' iteracích, ukončíme trénink
        if patience_counter >= patience:
            print(f"Zastaveno: Loss se nezlepšil po {patience} iteracích")
            break

        # Kontrola, zda došlo ke konvergenci
        if i > 0 and abs(loss_history[-2] - loss_history[-1]) < tol:
            print(f"Konvergence dosažena v iteraci {i}")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Průběh ztrátové funkce během trénování')
    plt.xlabel('Iterace')
    plt.ylabel('Loss (Frobenius norma^2)')
    plt.grid(True)

    save_dir = "../output/task7"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))

    plt.close()

    return best_W_t if best_W_t is not None else W_t


def translate_word(word, source_embeddings, target_word_to_vec, W_t, top_n=5):
    word_embedding = source_embeddings[word]

    # Transformujeme embedding do cílového prostoru
    transformed_embedding = np.matmul(word_embedding, W_t)

    # Normalizujeme pro výpočet kosinové podobnosti
    transformed_embedding = transformed_embedding / np.linalg.norm(transformed_embedding)

    # Vypočítáme podobnost s embeddingy slov cílového jazyka
    similarities = []
    for target_word, target_embedding in target_word_to_vec.items():
        target_embedding_norm = target_embedding / np.linalg.norm(target_embedding)
        similarity = np.dot(transformed_embedding, target_embedding_norm)
        similarities.append((target_word, similarity))

    # Vrátíme top_n nejpodobnějších slov
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]


def evaluate_translation(test_pairs, source_embeddings, target_embeddings, W_t):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for source_word, target_word in tqdm(test_pairs, desc="Vyhodnocování překladu"):
        if source_word not in source_embeddings or target_word not in target_embeddings:
            continue

        translations = translate_word(source_word, source_embeddings, target_embeddings, W_t, top_n=5)
        if not translations:
            continue

        total += 1

        # Kontrola top-1 přesnosti
        if translations[0][0] == target_word:
            correct_top1 += 1

        # Kontrola top-5 přesnosti
        if any(t[0] == target_word for t in translations):
            correct_top5 += 1

    top1_accuracy = correct_top1 / total if total > 0 else 0
    top5_accuracy = correct_top5 / total if total > 0 else 0

    return top1_accuracy, top5_accuracy


def main():
    # 1. Stažení a příprava dat
    cs_emb_path, en_emb_path, train_dict_path, test_dict_path = download_resources()

    # 2. Načtení embeddingů a slovníků - načítáme všechna dostupná slova!
    cs_embeddings = load_embeddings(cs_emb_path)
    en_embeddings = load_embeddings(en_emb_path)

    train_pairs = load_word_pairs(train_dict_path)
    test_pairs = load_word_pairs(test_dict_path)

    # 3. Vytvoření matic X (zdrojový jazyk) a Y (cílový jazyk) pro trénovací část
    X_train, Y_train, valid_train_pairs = create_embedding_matrices(
        train_pairs, cs_embeddings, en_embeddings
    )

    # 4. Vytvoření matic X_test a Y_test pro testovací část
    X_test, Y_test, valid_test_pairs = create_embedding_matrices(
        test_pairs, cs_embeddings, en_embeddings
    )

    print(f"Matice X_train tvar: {X_train.shape}")
    print(f"Matice Y_train tvar: {Y_train.shape}")
    print(f"Matice X_test tvar: {X_test.shape}")
    print(f"Matice Y_test tvar: {Y_test.shape}")
    print(f"Počet platných trénovacích párů: {len(valid_train_pairs)}")
    print(f"Počet platných testovacích párů: {len(valid_test_pairs)}")

    # 5. Trénování transformační matice pomocí gradient descent
    print("\nZačátek trénování transformační matice...")
    W_t = gradient_descent(X_train, Y_train, learning_rate=0.001, max_iterations=500, patience=20)
    print("\nTrénování dokončeno!")

    # 6. Vyhodnocování kvality překladu na malém vzorku dat
    print("Vyhodnocování kvality překladu...")
    # Použijeme pouze náhodných 15 párů pro rychlejší vyhodnocení
    random.seed(42)  # pro reprodukovatelnost
    sample_test_pairs = random.sample(valid_test_pairs, min(15, len(valid_test_pairs)))
    print(f"Vyhodnocuji na vzorku {len(sample_test_pairs)} párů z celkových {len(valid_test_pairs)}")
    top1_accuracy, top5_accuracy = evaluate_translation(sample_test_pairs, cs_embeddings, en_embeddings, W_t)
    print(f"Přesnost překladu (Top-1): {top1_accuracy:.4f}")
    print(f"Přesnost překladu (Top-5): {top5_accuracy:.4f}")

    # also translate those 15 pairs
    print("\nUkázka překladu vybraných párů:")
    for source_word, target_word in sample_test_pairs:
        translations = translate_word(source_word, cs_embeddings, en_embeddings, W_t)
        print(f"{source_word} -> {[t[0] for t in translations]} (Očekávaný překlad: {target_word})")

    # 7. Ukázka překladu několika českých slov
    print("\nUkázka překladu vybraných slov:")
    test_words = ["pes", "kočka", "auto", "počítač", "dům"]
    for word in test_words:
        if word in cs_embeddings:
            translations = translate_word(word, cs_embeddings, en_embeddings, W_t)
            print(f"{word} -> {[t[0] for t in translations]} (Očekávaný překlad: {translations[0][0]})")


if __name__ == "__main__":
    main()
