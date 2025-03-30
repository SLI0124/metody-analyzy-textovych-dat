import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import random


def create_inverted_index(df):
    try:  # download stopwords if not already present
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    documents = {}
    inverted_index = {}

    for idx, row in df.iterrows():
        doc_id = idx
        content = row['content']
        title = row['title']

        documents[doc_id] = {
            'title': title,
            'content': content,
            'filename': row['filename'],
            'category': row['category']
        }

        text = content.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]

        word_frequency_dict = {}
        for token in tokens:
            if token in word_frequency_dict:
                word_frequency_dict[token] += 1
            else:
                word_frequency_dict[token] = 1

        # add the word frequency to the inverted index
        for token, freq in word_frequency_dict.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][doc_id] = freq

    return inverted_index, documents


def main():
    df = pd.read_csv("../input/task4/bbc-news-data.csv", sep='\t')
    # columns: category, filename, title, content

    print("\033[91m" + "=== Task one ===" + "\033[0m")
    inverted_index, documents = create_inverted_index(df)

    print("Total number of documents: " + "\033[91m" + "{:,}"
          .format(len(documents)).replace(",", " ") + "\033[0m")
    print("Total number of inverted index: " + "\033[91m" + "{:,}"
          .format(len(inverted_index))
          .replace(",", " ") + "\033[0m")

    avg_list_length = sum(len(postings) for postings in inverted_index.values()) / len(inverted_index)
    print("Average list length: " + "\033[91m" + "{:.2f}"
          .format(avg_list_length) + "\033[0m")

    total_entries = sum(len(postings) for postings in inverted_index.values())
    print("Celkový počet záznamů v indexu: " + "\033[91m" + "{:,}"
          .format(total_entries)
          .replace(",", " ") + "\033[0m")

    print("\033[94m" + "\nPick 5 random tokes from the inverted index:" + "\033[0m")
    sampled_tokens = random.sample(list(inverted_index.items()), 5)
    for token, postings in sampled_tokens:
        print(f"Token: '\033[92m{token}\033[0m', Numbers of documents: {len(postings)}")
        if len(postings) > 3:
            for doc_id, freq in list(postings.items())[:3]:
                print(f"\tDocument {doc_id}: {documents[doc_id]['title']} (frequency: {freq})")
            print("\t...")
        else:
            for doc_id, freq in list(postings.items()):
                print(f"\tDocument {doc_id}: {documents[doc_id]['title']} (frequency: {freq})")


if __name__ == "__main__":
    main()
