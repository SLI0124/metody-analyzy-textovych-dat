import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess_documents(documents, custom_stopwords=None):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stopwords)

    processed_docs = []

    for doc in documents:
        doc_lower = doc.lower()
        doc_no_punctuation = doc_lower.translate(str.maketrans('', '', string.punctuation))
        doc_no_numbers = ''.join([char for char in doc_no_punctuation if not char.isdigit()])
        doc_clean = ' '.join(doc_no_numbers.split())
        tokens = word_tokenize(doc_clean)
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha() and len(word) > 1]

        processed_docs.append(filtered_tokens)

    return processed_docs


def compute_tf(document):
    term_counts = {}
    doc_length = len(document)

    for term in document:
        term_counts[term] = term_counts.get(term, 0) + 1

    tf = {}
    for term, count in term_counts.items():
        tf[term] = count / doc_length
    return tf


def compute_idf(processed_docs):
    num_docs = len(processed_docs)
    term_doc_freq = {}

    for doc in processed_docs:
        terms_in_doc = set(doc)
        for term in terms_in_doc:
            if term in term_doc_freq:
                term_doc_freq[term] += 1
            else:
                term_doc_freq[term] = 1

    idf = {}
    for term, freq in term_doc_freq.items():
        idf[term] = math.log(num_docs / freq)
    return idf


def compute_tf_idf(processed_docs):
    idf_values = compute_idf(processed_docs)
    tf_idf_docs = []

    for doc in processed_docs:
        tf_values = compute_tf(doc)
        doc_tf_idf = {}
        for term, tf in tf_values.items():
            idf = idf_values.get(term, 0)
            doc_tf_idf[term] = tf * idf
        tf_idf_docs.append(doc_tf_idf)

    return tf_idf_docs, idf_values


def search_documents(query_tokens, tf_idf_docs, file_ids):
    scores = []

    for i, doc_tf_idf in enumerate(tf_idf_docs):
        score = 0
        for term in query_tokens:
            if term in doc_tf_idf:
                score += doc_tf_idf[term]

        scores.append((i, score, file_ids[i]))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def cosine_similarity(doc1, doc2):
    all_terms = []
    for key in doc1:
        if key not in all_terms:
            all_terms.append(key)
    for key in doc2:
        if key not in all_terms:
            all_terms.append(key)

    dot_product = 0
    for term in all_terms:
        dot_product += doc1.get(term, 0) * doc2.get(term, 0)

    norm1 = 0
    for value in doc1.values():
        norm1 += value * value
    norm1 = np.sqrt(norm1)

    norm2 = 0
    for value in doc2.values():
        norm2 += value * value
    norm2 = np.sqrt(norm2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)


def compute_document_similarities(tf_idf_docs):
    n_docs = len(tf_idf_docs)
    sim_matrix = np.zeros((n_docs, n_docs))

    most_similar = (0, 0, 0)

    for i in range(n_docs):
        for j in range(n_docs):
            sim = cosine_similarity(tf_idf_docs[i], tf_idf_docs[j])
            sim_matrix[i, j] = sim

            if i != j and sim > most_similar[2]:
                most_similar = (i, j, sim)

    return sim_matrix, most_similar


def plot_similarity_matrix(similarity_matrix, file_ids, most_similar=None):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')

    if most_similar:
        i, j, _ = most_similar
        plt.plot(j, i, 'x', markersize=10, markeredgewidth=2)

    labels = [fid.split('-')[0] for fid in file_ids]

    step = max(1, len(file_ids) // 10)
    plt.xticks(range(0, len(file_ids), step), [labels[i] for i in range(0, len(file_ids), step)], rotation=90)
    plt.yticks(range(0, len(file_ids), step), [labels[i] for i in range(0, len(file_ids), step)])

    save_path = "../output/task5/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.title('Cosine Similarity Matrix of Documents')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'document_similarity.png'), dpi=300)
    plt.close()


def main():
    try:
        nltk.data.find('corpora/inaugural')
    except LookupError:
        nltk.download('inaugural')

    data = nltk.corpus.inaugural
    file_ids = data.fileids()
    documents = [data.raw(fileid) for fileid in file_ids]

    print(f"Loaded \033[94m{len(documents)}\033[0m documents from inaugural speeches.")
    custom_stopwords = ['government', 'country', 'nation', 'people', 'would', 'america']
    processed_docs = preprocess_documents(documents, custom_stopwords)

    print("\n\033[91m=== Task one ===\033[0m")
    random_indices = random.sample(range(len(documents)), 3)
    for i in random_indices:
        print(f"\nDocument {i + 1}, \033[92mFile: {file_ids[i]}\033[0m")
        first_tokens = processed_docs[i][:20]
        print(f"Tokens: {first_tokens}")

    print("\n\033[91m=== Task two ===\033[0m")
    tf_idf_docs, idf_values = compute_tf_idf(processed_docs)

    random_doc_idx = random.choice(range(len(processed_docs)))
    print(f"\nSample tf-idf values for document \033[92m{file_ids[random_doc_idx]}\033[0m:")

    sample_terms = list(tf_idf_docs[random_doc_idx].items())[:5]
    table_data = [[term, f"{weight:.4f}"] for term, weight in sample_terms]
    print(tabulate(table_data, headers=["Term", "tf-idf"], tablefmt="simple_grid"))

    print("\n\033[94mSearch examples using tf-idf:\033[0m")
    queries = [
        ["freedom", "liberty"],
        ["war", "peace"],
        ["economy", "future"]
    ]

    for query in queries:
        print(f"\nResults for query: \033[93m{query}\033[0m")
        results = search_documents(query, tf_idf_docs, file_ids)

        for i, (doc_idx, score, file_id) in enumerate(results[:5]):
            print(f"{i + 1}. Document: \033[92m{file_id}\033[0m, Score: \033[94m{score:.4f}\033[0m")
            representative_terms = sorted(
                [(term, weight) for term, weight in tf_idf_docs[doc_idx].items()],
                key=lambda x: x[1], reverse=True
            )[:3]
            rep_terms_str = ", ".join([f"{term}" for term, _ in representative_terms])
            print(f"\tRepresentative terms: {rep_terms_str}")

    print("\n\033[91m=== Task three ===\033[0m")
    similarity_matrix, most_similar = compute_document_similarities(tf_idf_docs)
    i, j, sim = most_similar

    print(f"\nMost similar documents:")
    print(f"\t1. \033[92m{file_ids[i]}\033[0m")
    print(f"\t2. \033[92m{file_ids[j]}\033[0m")
    print(f"\tCosine similarity: \033[94m{sim:.4f}\033[0m")

    common_terms = set(tf_idf_docs[i]).intersection(tf_idf_docs[j])
    combined_scores = {}
    for term in common_terms:
        combined_scores[term] = tf_idf_docs[i][term] + tf_idf_docs[j][term]
    important_common_terms = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:5]

    print("\nSignificant common terms:")
    for term, _ in important_common_terms:
        print(f"\t• {term}")

    plot_similarity_matrix(similarity_matrix, file_ids, most_similar)


if __name__ == '__main__':
    main()
