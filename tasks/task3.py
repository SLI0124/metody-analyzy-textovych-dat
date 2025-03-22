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


def main():
    desired_word = "kitchen"
    words = ["kitchen", "kitten", "sitten", "sittin", "sitting"]
    for word in words:
        distance, dp_table = levenstein_distance_dp(desired_word, word)
        print(f"Levenstein distance between '{desired_word}' and '{word}': {distance}")
        print_dp_table(desired_word, word, dp_table)
        print()


if __name__ == "__main__":
    main()
