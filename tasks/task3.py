from tabulate import tabulate


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


if __name__ == "__main__":
    main()
