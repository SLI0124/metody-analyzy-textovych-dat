def load_data(file_path):
    with open(file_path, "r") as file:
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


def main():
    data = load_data("../input/ELRC-antibiotic.cs-en.cs.txt")
    tokens = tokenize_data(data)

    # task 1
    for n in range(1, 4):
        print(f"Most frequent {n}-grams:")
        print(get_the_most_frequent_n_grams(tokens, n, 10), end="\n\n")


if __name__ == "__main__":
    main()
