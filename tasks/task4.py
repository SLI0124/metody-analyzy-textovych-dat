import pandas as pd
import nltk
from nltk.corpus import stopwords

import random
import re
import string
from enum import Enum


class TokenType(Enum):
    WORD = 1
    AND = 2
    OR = 3
    NOT = 4
    LPAREN = 5
    RPAREN = 6


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


def tokenize_query(query):
    query = query.lower()
    query = query.replace('(', ' ( ')
    query = query.replace(')', ' ) ')

    tokens = []
    words = query.split()

    for word in words:
        if word == 'and':
            tokens.append((TokenType.AND, word))
        elif word == 'or':
            tokens.append((TokenType.OR, word))
        elif word == 'not':
            tokens.append((TokenType.NOT, word))
        elif word == '(':
            tokens.append((TokenType.LPAREN, word))
        elif word == ')':
            tokens.append((TokenType.RPAREN, word))
        else:
            word = re.sub(f'[{string.punctuation}]', '', word.lower())
            if word:
                tokens.append((TokenType.WORD, word))

    return tokens


def parse_query_into_abstract_syntax_tree(tokens):
    """ Parse the tokenized query into an abstract syntax tree. This function uses a recursive descent parser to handle
    the correct order of operators and parentheses.
    The grammar is as follows:
    query ::= or_expr
    or_expr ::= and_expr (OR and_expr)*
    and_expr ::= not_expr (AND not_expr)*
    not_expr ::= NOT term | term
    term ::= WORD | LPAREN query RPAREN
    This grammar allows for nested expressions and operator ordering. """

    def parse_or():
        left = parse_and()
        while tokens and tokens[0][0] == TokenType.OR:
            tokens.pop(0)
            right = parse_and()
            left = ('OR', left, right)
        return left

    def parse_and():
        left = parse_not()
        while tokens and tokens[0][0] == TokenType.AND:
            tokens.pop(0)
            right = parse_not()
            left = ('AND', left, right)
        return left

    def parse_not():
        if tokens and tokens[0][0] == TokenType.NOT:
            tokens.pop(0)
            return 'NOT', parse_term()
        return parse_term()

    def parse_term():
        if not tokens:
            raise ValueError("Unexpected end of input")

        token_type, value = tokens[0]

        if token_type == TokenType.WORD:
            tokens.pop(0)
            return 'WORD', value

        if token_type == TokenType.LPAREN:
            tokens.pop(0)  # remove left parenthesis
            result = parse_or()  # evaluate the expression inside the parentheses

            if not tokens or tokens[0][0] != TokenType.RPAREN:
                raise ValueError("Missing right parenthesis")

            tokens.pop(0)  # remove right parenthesis
            return result

        raise ValueError(f"Unexpected token: {value}")

    return parse_or()


def evaluate_abstract_syntax_tree(abstract_syntax_tree, inverted_index, documents):
    if not abstract_syntax_tree:
        return set()

    operator = abstract_syntax_tree[0]

    if operator == 'WORD':  # return documents containing the word
        word = abstract_syntax_tree[1]
        return set(inverted_index.get(word, {}).keys())

    if operator == 'NOT':  # complement of the documents containing the word
        result = evaluate_abstract_syntax_tree(abstract_syntax_tree[1], inverted_index, documents)
        return set(documents.keys()) - result

    if operator == 'AND':  # intersection of the documents containing both words
        left_result = evaluate_abstract_syntax_tree(abstract_syntax_tree[1], inverted_index, documents)
        right_result = evaluate_abstract_syntax_tree(abstract_syntax_tree[2], inverted_index, documents)
        return left_result & right_result

    if operator == 'OR':  # union of the documents containing either word
        left_result = evaluate_abstract_syntax_tree(abstract_syntax_tree[1], inverted_index, documents)
        right_result = evaluate_abstract_syntax_tree(abstract_syntax_tree[2], inverted_index, documents)
        return left_result | right_result

    raise ValueError(f"Unrecognized operator: {operator}")


def display_results(results, documents, max_results=10):
    if not results:
        print("\033[91m" + "Haven´t found any documents matching the query." + "\033[0m")
        return

    print(f"\033[92mFound {len(results)} documents:\033[0m")

    for i, doc_id in enumerate(list(results)[:max_results]):
        doc = documents[doc_id]
        print(f"{i + 1}. [{doc_id}] {doc['title']}")
        first_sentence = doc['content'].split('.')[0] + '.'
        print(f"\t{first_sentence}...")

    if len(results) > max_results:
        print(f"\t and {len(results) - max_results} more documents.")


def search(query, inverted_index, documents):
    tokens = tokenize_query(query)
    ast = parse_query_into_abstract_syntax_tree(tokens)
    result_docs = evaluate_abstract_syntax_tree(ast, inverted_index, documents)

    return result_docs


def main():
    df = pd.read_csv("../input/task4/bbc-news-data.csv", sep='\t')
    # columns: category, filename, title, content

    print("\033[91m" + "=== Task one & Task three ===" + "\033[0m")
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
    print("Total number of entries in the index: " + "\033[91m" + "{:,}"
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

    print("\n\033[91m=== Task two ===\033[0m")
    test_queries = [
        "economy",
        "economy AND business",
        "economy OR business",
        "economy AND NOT crisis",
        "(economy OR market) AND business",
        "economy AND (business OR trade) AND NOT crisis"
    ]

    for query in test_queries:
        print(f"\n\033[94mTesting query:\033[0m {query}")
        try:
            results = search(query, inverted_index, documents)
            display_results(results, documents, max_results=3)
        except ValueError as e:
            print(f"\033[91mError: {e}\033[0m")


if __name__ == "__main__":
    main()
