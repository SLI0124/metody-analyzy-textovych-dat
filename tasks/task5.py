import random
import math

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


if __name__ == '__main__':
    main()
