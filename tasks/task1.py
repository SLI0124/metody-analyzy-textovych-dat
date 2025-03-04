import re
import random
import math


def load_data_to_lowercase(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().lower()


def tokenize_data(data):
    data = re.sub(r"\[\d+(?:,\s?\d+)*]", "", data)  # remove references and citations
    data = re.sub(r"[.,:;!?\-()\[\]]", "", data)  # remove punctuation
    data = re.sub(r"\b(\d+)\s*(\d+)\b", r"\1 \2", data)  # remove spaces between numbers
    return [line.split() for line in data.split("\n")]


def create_n_grams(tokens, n):
    n_grams = []
    for line in tokens:
        for i in range(len(line) - n + 1):
            n_grams.append(tuple(line[i:i + n]))
    return n_grams


def create_n_grams_dict(tokens, n):
    n_grams = {}
    for n_gram in create_n_grams(tokens, n):
        if n_gram in n_grams:
            n_grams[n_gram] += 1
        else:
            n_grams[n_gram] = 1
    return n_grams


def get_most_frequent_n_grams(tokens, n, k):
    n_grams = create_n_grams_dict(tokens, n)
    return sorted(n_grams.items(), key=lambda x: x[1], reverse=True)[:k]


def calculate_next_word_probability_laplace(tokens, n, alpha=1):
    n_grams = create_n_grams_dict(tokens, n)
    n_plus_one_grams = create_n_grams_dict(tokens, n + 1)
    vocab_size = len(set(word for line in tokens for word in line))
    next_word_probabilities = {}

    for n_plus_one_gram, count in n_plus_one_grams.items():
        prev_words, next_word = n_plus_one_gram[:-1], n_plus_one_gram[-1]
        if prev_words not in next_word_probabilities:
            next_word_probabilities[prev_words] = {}
        probability = (count + alpha) / (n_grams.get(prev_words, 0) + alpha * vocab_size)
        next_word_probabilities[prev_words][next_word] = probability

    return next_word_probabilities


def calculate_possible_next_words(n_plus_one_grams, n_grams, current_word, alpha, vocab_size):
    possible_next_words = {}
    for n_plus_one_gram, count in n_plus_one_grams.items():
        prev_words, next_word = n_plus_one_gram[:-1], n_plus_one_gram[-1]
        if prev_words[-1] == current_word:
            probability = (count + alpha) / (n_grams.get(prev_words, 0) + alpha * vocab_size)
            possible_next_words[next_word] = probability
    return possible_next_words


def predict_next_word(tokens, input_word, context_size=3, alpha=1, top_k=5):
    n_grams = create_n_grams_dict(tokens, context_size)
    n_plus_one_grams = create_n_grams_dict(tokens, context_size + 1)
    vocab = set(word for line in tokens for word in line)
    vocab_size = len(vocab)

    if not input_word:
        return []

    predictions = calculate_possible_next_words(n_plus_one_grams, n_grams, input_word, alpha, vocab_size)

    found_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    # Apply fallback smoothing if no predictions found
    if not found_predictions:
        print(f"No direct {context_size}-grams found for '{input_word}', applying Laplace smoothing.")
        for word in vocab:
            predictions[word] = alpha / (alpha * vocab_size)

        found_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    # Fill in with additional predictions if there are fewer than top_k predictions
    remaining_predictions = top_k - len(found_predictions)
    if remaining_predictions > 0:
        print(f"Less than {top_k} predictions found ({len(found_predictions)}), "
              f"filling in with additional predictions.")

        # print the found predictions
        print("Found predictions:", end=" ")
        for word, prob in found_predictions:
            print(f"{word}: {prob:.4f}", end=", ")
        print()

        additional_predictions = [(word, alpha / (alpha * vocab_size)) for word in vocab]
        additional_predictions = sorted(additional_predictions, key=lambda x: x[1], reverse=True)

        # Add additional predictions to the results
        found_predictions.extend(additional_predictions[:remaining_predictions])

    return found_predictions[:top_k]


def generate_text(tokens, sentence_count=5, max_sentence_length=20, alpha=1, min_sentence_length=5):
    n_grams = create_n_grams_dict(tokens, 3)  # tri-grams
    n_plus_one_grams = create_n_grams_dict(tokens, 4)  # four-grams
    vocab = set(word for line in tokens for word in line)
    vocab_size = len(vocab)

    input_word = random.choice([word for line in tokens for word in line if len(word) > 2])
    print(f"Starting text generation with word: '{input_word}'")

    generated_text = []

    for _ in range(sentence_count):
        sentence = [input_word]
        current_word = input_word

        while len(sentence) < max_sentence_length:
            next_words = calculate_possible_next_words(n_plus_one_grams, n_grams, current_word, alpha, vocab_size)
            if not next_words:
                break

            next_word = random.choices(list(next_words.keys()), list(next_words.values()))[0]
            sentence.append(next_word)
            current_word = next_word

            if len(sentence) >= min_sentence_length and sentence[-1][-1] in ['.', '?', '!']:
                break

        generated_text.append(" ".join(sentence).capitalize() + ".")

    return "\n\n".join(generated_text)


def calculate_perplexity(tokens, n, test_data, alpha=1):
    def get_ngram_count(ngram_dict, ngram):
        return ngram_dict.get(ngram, 0)

    n_grams = create_n_grams_dict(tokens, n)
    n_plus_one_grams = create_n_grams_dict(tokens, n + 1)
    vocab_size = len(set(word for line in tokens for word in line))

    log_prob_sum = 0
    total_ngrams = 0

    for line in test_data:
        line_tokens = line.split()
        if len(line_tokens) < n:
            continue
        for i in range(len(line_tokens) - n):
            context = tuple(line_tokens[i:i + n])
            next_word = line_tokens[i + n]

            context_count = get_ngram_count(n_grams, context)
            n_plus_one_gram = context + (next_word,)
            n_plus_one_count = get_ngram_count(n_plus_one_grams, n_plus_one_gram)

            probability = (n_plus_one_count + alpha) / (context_count + alpha * vocab_size)
            log_prob_sum += math.log2(probability)
            total_ngrams += 1

    if total_ngrams == 0:
        return float('inf')  # Avoid division by zero

    perplexity = 2 ** (-log_prob_sum / total_ngrams)
    return perplexity


def main():
    data = load_data_to_lowercase("../input/ELRC-antibiotic.cs-en.cs.txt")
    tokens = tokenize_data(data)

    print("\033[1;31mTask 1:\033[0m")
    for n in range(1, 4):
        print(f"\033[1;34mMost frequent {n}-grams:\033[0m")
        for n_gram, freq in get_most_frequent_n_grams(tokens, n, 10):
            print(f"{' '.join(n_gram)}: {freq}")
        print()

    print("\033[1;31mTask 2:\033[0m")
    for n in range(2, 4):
        print(f"\033[1;34mNext word probabilities for {n}-grams:\033[0m")
        next_word_probs = calculate_next_word_probability_laplace(tokens, n)
        most_frequent_n_grams = get_most_frequent_n_grams(tokens, n, 10)

        for n_gram, _ in most_frequent_n_grams:
            print(f"\033[1;32mNext word probabilities for \"{' '.join(n_gram)}\":\033[0m")
            if n_gram in next_word_probs:
                for next_word, prob in sorted(next_word_probs[n_gram].items(), key=lambda x: x[1], reverse=True):
                    print(f"{next_word}: {prob:.4f}")
            print()

    print("\033[1;31mTask 3:\033[0m")
    while True:
        input_word = input("Enter a word to predict the next word: (or type 'exit' to quit)\n")
        if input_word == "exit":
            break
        predictions = predict_next_word(tokens, input_word)
        if predictions:
            print(f"\nNext word predictions for '{input_word}':")
            for word, prob in predictions:
                print(f"{word}: {prob:.4f}")
            print()
        else:
            print("No predictions available.")

    print("\033[1;31mTask 4:\033[0m")
    generated_text = generate_text(tokens)
    print("\033[1;32mGenerated text:\033[0m")
    print(generated_text)

    print("\033[1;31mTask 5: Perplexity Evaluation\033[0m")
    split_index = int(0.8 * len(tokens))
    train_tokens = tokens[:split_index]
    test_tokens = tokens[split_index:]

    test_data = [" ".join(line) for line in test_tokens]

    print("\033[1;34mPerplexity Comparison:\033[0m")
    for n in range(1, 4):  # one-grams, bi-grams, tri-grams
        for alpha in [0.1, 1, 5]:
            perplexity = calculate_perplexity(train_tokens, n, test_data, alpha)
            print(f"Perplexity for {n}-grams with alpha={alpha}: {perplexity:.4f}")
        print()


if __name__ == "__main__":
    main()
