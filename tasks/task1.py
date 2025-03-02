def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def tokenize_data(data):
    return [line.split() for line in data.split("\n")]


def create_n_grams(tokens, n):
    return [line[i:i + n] for line in tokens for i in range(len(line) - n + 1)]


def create_n_grams_dict(tokens, n):
    n_grams = {}
    for n_gram in create_n_grams(tokens, n):
        n_gram_str = " ".join(n_gram)
        n_grams[n_gram_str] = n_grams.get(n_gram_str, 0) + 1
    return n_grams


def get_the_most_frequent_n_grams(tokens, n, k):
    n_grams = create_n_grams_dict(tokens, n)
    return sorted(n_grams.items(), key=lambda x: x[1], reverse=True)[:k]


def calculate_probabilities(n_grams_dict):
    total = sum(n_grams_dict.values())
    return {k: v / total for k, v in n_grams_dict.items()}


def calculate_next_word_probability_laplace(tokens, n, alpha=1):
    # source: https://en.wikipedia.org/wiki/Additive_smoothing
    n_grams = create_n_grams_dict(tokens, n)
    n_plus_one_grams = create_n_grams_dict(tokens, n + 1)

    vocab_size = len(set(word for line in tokens for word in line))
    next_word_probabilities = {}

    for n_plus_one_gram, count in n_plus_one_grams.items():
        n_gram = " ".join(n_plus_one_gram.split()[:-1])
        next_word = n_plus_one_gram.split()[-1]

        if n_gram not in next_word_probabilities:
            next_word_probabilities[n_gram] = {}

        probability = (count + alpha) / (n_grams.get(n_gram, 0) + alpha * vocab_size)
        next_word_probabilities[n_gram][next_word] = probability

    return next_word_probabilities


def main():
    data = load_data("../input/ELRC-antibiotic.cs-en.cs.txt")
    tokens = tokenize_data(data)

    # task 1
    for n in range(1, 4):
        print(f"\033[92mMost frequent {n}-grams:\033[0m")
        print(get_the_most_frequent_n_grams(tokens, n, 10), end="\n\n")

    # task 2
    for n in range(2, 4):
        print(f"\033[92mMost frequent {n}-grams:\033[0m")
        next_word_probabilities = calculate_next_word_probability_laplace(tokens, n)
        most_frequent_2_grams = get_the_most_frequent_n_grams(tokens, n, 10)
        for n_gram, _ in most_frequent_2_grams:
            print(f"\033[94mNext word probabilities for \"{n_gram}\":\033[0m")
            if n_gram in next_word_probabilities:
                for next_word, probability in next_word_probabilities[n_gram].items():
                    print(f"{next_word}: {probability}")
                print()
            else:
                print("No data available", end="\n\n")


if __name__ == "__main__":
    main()
