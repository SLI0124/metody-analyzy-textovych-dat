import random

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


if __name__ == '__main__':
    main()
