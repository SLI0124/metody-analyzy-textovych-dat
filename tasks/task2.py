import os
import csv

from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def brute_force(data, match_string):
    comparisons = 0
    positions = []
    for i in range(len(data) - len(match_string) + 1):
        comparisons += 1
        if data[i:i + len(match_string)] == match_string:
            positions.append(i)
    return comparisons, len(positions), positions


def construct_kmp_table(pattern, lps):
    length = 0
    m = len(pattern)

    lps[0] = 0

    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1


def kmp_search_algorithm(data, pattern):
    """
    https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
    """
    n = len(data)
    m = len(pattern)

    lsp = [0] * m
    result = []

    construct_kmp_table(pattern, lsp)

    i = 0
    j = 0
    comparisons = 0
    positions = []

    while i < n:
        if data[i] == pattern[j]:
            i += 1
            j += 1
            comparisons += 1

            if j == m:
                result.append(i - j)
                positions.append(i - j)
                j = lsp[j - 1]
        else:
            if j != 0:
                j = lsp[j - 1]
            else:
                i += 1

    return comparisons, len(positions), positions


def bad_character_heuristic(pattern, size):
    number_of_characters = 256
    bad_characters = [-1] * number_of_characters
    for i in range(size):
        bad_characters[ord(pattern[i])] = i

    return bad_characters


def boyer_moore_search_algorithm(data, pattern):
    """
    https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/
    """
    m = len(pattern)
    n = len(data)

    bad_characters = bad_character_heuristic(pattern, m)
    shift = 0
    comparisons = 0
    positions = []
    while shift <= n - m:
        j = m - 1

        while j >= 0 and pattern[j] == data[shift + j]:
            j -= 1
            comparisons += 1

        if j < 0:
            shift += (m - bad_characters[ord(data[shift + m])] if shift + m < n else 1)
            positions.append(shift - len(pattern) - 1)
        else:
            shift += max(1, j - bad_characters[ord(data[shift + j])])

    return comparisons, len(positions), positions


def main():
    data_files = {
        "Short data": '../input/task2/short_sample.txt',
        "Long data": '../input/task2/long_sample.txt',
        "DNA data": '../input/task2/dna_sample.txt'
    }

    patterns = {
        "Short data": ['little', 'he', 'legs'],
        "Long data": ['little', 'he', 'legs'],
        "DNA data": ['aagctt', 'cg', 'ccg']
    }

    algorithms = [brute_force, kmp_search_algorithm, boyer_moore_search_algorithm]
    table_data = []
    csv_data = []
    plot_data = {
        "labels": [],
        "comparisons": []
    }

    for data_name, file_path in data_files.items():
        data = load_data(file_path)

        for pattern in patterns[data_name]:
            for algorithm in algorithms:
                comparisons, count, positions = algorithm(data, pattern)

                truncated_positions = positions[:5] if len(positions) > 5 else positions
                if len(positions) > 5:
                    truncated_positions.append('...')

                table_data.append([
                    data_name,
                    pattern,
                    algorithm.__name__,
                    comparisons,
                    count,
                    str(truncated_positions)
                ])

                csv_data.append([
                    data_name,
                    pattern,
                    algorithm.__name__,
                    comparisons,
                    count,
                    positions
                ])

                plot_data["comparisons"].append(comparisons)
                plot_data["labels"].append(f"{data_name} - {pattern} - {algorithm.__name__}")

    headers = ["Dataset", "Pattern", "Algorithm", "Comparisons", "Count", "Positions"]

    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

    save_dir = '../output/task2/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name_csv = 'task2.csv'
    with open(save_dir + file_name_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

        colors = ['purple', 'orange', 'red', 'blue', 'green', 'cyan']

        alg_names_dict = {
            "brute_force": "Brute Force",
            "kmp_search_algorithm": "KMP",
            "boyer_moore_search_algorithm": "Boyer Moore"
        }

        for data_name in data_files.keys():
            dataset_labels = []
            dataset_comparisons = []
            dataset_patterns = []
            dataset_algorithms = []

            for i, label in enumerate(plot_data["labels"]):
                if label.startswith(data_name):
                    parts = label.split(" - ")
                    dataset_patterns.append(parts[1])  # Extract pattern
                    dataset_algorithms.append(parts[2])  # Extract algorithm
                    dataset_labels.append(label)  # Full label for ordering
                    dataset_comparisons.append(plot_data["comparisons"][i])

            if dataset_labels:
                color_map = dict(zip(set(dataset_patterns), colors))

                fig, ax = plt.subplots(figsize=(12, 5))
                y_positions = np.arange(len(dataset_labels))

                pattern_legend_shown = set()

                for i, (comparison, pattern, algorithm) in enumerate(
                        zip(dataset_comparisons, dataset_patterns, dataset_algorithms)):
                    color = color_map[pattern]
                    if pattern not in pattern_legend_shown:
                        ax.barh(y_positions[i], comparison, color=color, label=pattern)  # Add to legend
                        pattern_legend_shown.add(pattern)
                    else:
                        ax.barh(y_positions[i], comparison, color=color)  # No duplicate label

                ax.set_title(f'Comparisons for {data_name}')
                ax.set_xlabel('Comparisons')
                ax.set_yticks(y_positions)
                ax.set_yticklabels([alg_names_dict[alg] for alg in dataset_algorithms])
                ax.legend(title="Patterns")

                plt.tight_layout()
                plt.savefig(save_dir + f"{data_name}.png")
                plt.show()


if __name__ == '__main__':
    main()
