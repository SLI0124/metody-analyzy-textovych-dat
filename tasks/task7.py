import numpy as np
from tqdm import tqdm
import os
import requests
import gzip
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
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
            with gzip.open(emb_path + ".gz", 'rb') as f_in:
                with open(emb_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    return cs_emb_path, en_emb_path, train_dict_path, test_dict_path


def process_chunk(chunk_lines, dim=300):
    chunk_embeddings = {}
    for line in chunk_lines:
        values = line.strip().split(' ')
        word = values[0]
        vector = np.array([float(val) for val in values[1:]])
        chunk_embeddings[word] = vector

    return chunk_embeddings


def process_file_chunk(file_path, start_pos, end_pos, dim=300):
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
        header_line = f.readline().strip()
        header_parts = header_line.split()
        vocab_size, dim = int(header_parts[0]), int(header_parts[1])
        header_end_pos = f.tell()

    chunk_size = (file_size - header_end_pos) // num_workers
    start_positions = [header_end_pos + i * chunk_size for i in range(num_workers)]
    end_positions = start_positions[1:] + [file_size]

    print(f"Zpracování embeddingů pomocí {num_workers} procesů...")
    word_to_vector = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(process_file_chunk, embedding_file, dim=dim)
        future_to_chunk = {
            executor.submit(process_func, start, end): i
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

    print(f"Vytvořeny matice embedů s {len(matched_pairs)} páry slov.")
    return source_matrix, target_matrix, matched_pairs


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


if __name__ == "__main__":
    main()
