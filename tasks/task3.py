from tabulate import tabulate
import random


def levenstein_distance_dp(word1, word2):
    n = len(word1)
    m = len(word2)
    dp = []

    for i in range(n + 1):  # Initialize dp table with zeros
        dp.append([0] * (m + 1))

    # Base cases: distance from empty string requires i deletions or j insertions
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill dp table calculating minimum edit distance at each position
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + (word1[i - 1] != word2[j - 1])  # Substitution or match
            )
    return dp[n][m], dp


def print_dp_table(word1, word2, dp):
    headers = [''] + list(word2)

    table_data = []
    for i, row in enumerate(dp):
        row_header = '' if i == 0 else word1[i - 1]
        table_data.append([row_header] + row)

    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))


def build_czech_dictionary(file_path):
    word_counts = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = text.replace('\n', ' ').lower()
    for char in '.,!?;:()[]{}"/\\@#$%^&*+=|<>':
        text = text.replace(char, ' ')

    words = text.split()
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def get_most_common_words(word_counts, top_n=50):
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_counts[:top_n]


def generate_word_variants(word, max_distance=2):
    def generate_edits(innie_word):
        alphabet = 'abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýž'
        splits = []  # Generate all possible splits of the word and from them generate edits
        for i in range(len(innie_word) + 1):
            splits.append((innie_word[:i], innie_word[i:]))

        deletions = []
        for left, right in splits:
            if right:
                deletions.append(left + right[1:])

        insertions = []
        for left, right in splits:
            for char in alphabet:
                insertions.append(left + char + right)

        replacements = []
        for left, right in splits:
            if right:
                for char in alphabet:
                    replacements.append(left + char + right[1:])

        transpositions = []
        for left, right in splits:
            if len(right) > 1:
                transpositions.append(left + right[1] + right[0] + right[2:])

        return set(deletions + insertions + replacements + transpositions)

    variants = {word}  # Include the original word

    edits_distance = generate_edits(word)
    variants.update(edits_distance)

    if max_distance > 1:  # generate more levels of edits recursively
        for edit in edits_distance:
            variants.update(generate_edits(edit))

    return variants


def main():
    print(f"\033[91mFirst task\033[0m")
    desired_word = "kitchen"
    words = ["kitchen", "kitten", "sitten", "sittin", "sitting"]
    for word in words:
        distance, dp_table = levenstein_distance_dp(desired_word, word)
        print(f"Levenstein distance between '{desired_word}' and '{word}': {distance}")
        print_dp_table(desired_word, word, dp_table)
        print()

    print(f"\033[91mSecond task\033[0m")
    filepath = "../input/task3/hp_1.txt"
    word_counts = build_czech_dictionary(filepath)
    most_common_words = get_most_common_words(word_counts)
    print(tabulate(most_common_words, headers=["Word", "Frequency"], tablefmt='fancy_outline'))

    print(f"\033[91mThird task - Word Variants\033[0m")
    random_number_word_count = 3
    random_words = random.sample(list(word_counts.keys()), random_number_word_count)
    print(f"Random words from the dataset: {random_words}")
    max_distance = 2
    for test_word in random_words:
        variants = generate_word_variants(test_word, max_distance)
        variant_count = len(variants)
        print(f"Word: '{test_word}'")
        print(f"Number of variants (edit distance ≤ {max_distance}): {variant_count}")
        number_of_examples = 5
        sample_variants = random.sample(list(variants), number_of_examples)
        print(f"Sample of variants: {sample_variants}...")
        print()


if __name__ == "__main__":
    main()
