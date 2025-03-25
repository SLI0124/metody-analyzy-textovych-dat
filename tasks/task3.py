from tabulate import tabulate
import random
import time
from task1 import tokenize_data, create_n_grams_dict, load_data_to_lowercase


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
    headers = ['ε'] + list(word2)

    table_data = []
    for i, row in enumerate(dp):
        row_header = 'ε' if i == 0 else word1[i - 1]
        table_data.append([row_header] + row)

    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))


def build_czech_dictionary(file_path):
    word_counts = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = text.replace('\n', ' ').lower()
    for char in '.,!?;:()[]{}"/\\@#$%^&*+=|<>\"\'“”':
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


def autocorrect_word_variant_approach(word, word_counts, max_distance=2):
    if word in word_counts:
        return word

    variants = generate_word_variants(word, max_distance)
    valid_candidates = {}
    for variant in variants:
        if variant in word_counts:
            valid_candidates[variant] = word_counts[variant]

    return max(valid_candidates.items(), key=lambda x: x[1])[0] if valid_candidates else word


def autocorrect_word_dict_approach(word, word_counts, max_distance=2):
    if word in word_counts:
        return word

    candidates = {}
    for dict_word, freq in word_counts.items():
        distance, _ = levenstein_distance_dp(word, dict_word)
        if distance <= max_distance:
            candidates[dict_word] = freq

    return max(candidates.items(), key=lambda x: x[1])[0] if candidates else word


def tokenize(sentence):
    words, current_word = [], ""
    for char in sentence:
        if char.isalnum() or char in "áčďéěíňóřšťúůýž":
            current_word += char
        else:
            if current_word:
                words.append(current_word)
                current_word = ""
            if char.strip():
                words.append(char)
    if current_word:
        words.append(current_word)
    return words


def autocorrect_sentence(sentence, word_counts, max_distance=2, approach='variant'):
    punctuation = ".,!?;:()"

    if approach == 'variant':
        corrector = autocorrect_word_variant_approach
    elif approach == 'dict':
        corrector = autocorrect_word_dict_approach
    else:
        raise ValueError("Invalid approach. Use 'variant' or 'dict'")

    words = tokenize(sentence)
    corrected_words = []
    for word in words:
        if word in punctuation:
            corrected_words.append(word)
        else:
            corrected_words.append(corrector(word.lower(), word_counts, max_distance))

    corrected_sentence = corrected_words[0].capitalize()
    for i in range(1, len(corrected_words)):
        if corrected_words[i] not in punctuation:
            corrected_sentence += " "
        corrected_sentence += corrected_words[i]

    return corrected_sentence


def autocorrect_word_ngram_approach(word, word_counts, tokens, context_words, max_distance=2, context_size=2):
    if word in word_counts:
        return word

    candidates = set()
    for dict_word in word_counts:
        distance, _ = levenstein_distance_dp(word, dict_word)
        if distance <= max_distance:
            candidates.add(dict_word)

    if not candidates:
        return word

    n_gram_probs = {}
    for candidate in candidates:
        # Calculate probability of candidate given previous words
        prob = 1.0
        for i in range(1, context_size + 1):
            if i <= len(context_words):
                context = tuple(context_words[-i:] + [candidate])
                n_grams = create_n_grams_dict(tokens, len(context))
                context_count = n_grams.get(context, 0)
                total_context = sum(1 for ng in n_grams if ng[:-1] == context[:-1])
                prob *= (context_count + 1) / (total_context + len(word_counts))  # Laplace smoothing

        n_gram_probs[candidate] = prob

    # Return candidate with the highest n-gram probability
    return max(n_gram_probs.items(), key=lambda x: x[1])[0] if n_gram_probs else word


def autocorrect_sentence_ngram(sentence, word_counts, tokens, max_distance=2):
    words = tokenize(sentence)
    corrected_words = []
    context_words = []

    for word in words:
        if word in ".,!?;:()":
            corrected_words.append(word)
            continue

        correction = autocorrect_word_ngram_approach(
            word.lower(),
            word_counts,
            tokens,
            context_words,
            max_distance
        )

        corrected_words.append(correction)
        context_words.append(correction.lower())

    corrected_sentence = corrected_words[0].capitalize()
    for i in range(1, len(corrected_words)):
        if corrected_words[i] not in ".,!?;:()":
            corrected_sentence += " "
        corrected_sentence += corrected_words[i]

    return corrected_sentence


def compare_all_approaches(test_words, word_counts, tokens):
    results = []

    for word in test_words:
        start_variant = time.time()
        correction1 = autocorrect_word_variant_approach(word, word_counts)
        time_variant = time.time() - start_variant

        start_dict = time.time()
        correction2 = autocorrect_word_dict_approach(word, word_counts)
        time_dict = time.time() - start_dict

        start_ngram = time.time()
        correction3 = autocorrect_word_ngram_approach(word, word_counts, tokens, [])
        time_ngram = time.time() - start_ngram

        results.append([
            word,
            correction1,
            correction2,
            correction3,
            f"{time_variant:.6f}s",
            f"{time_dict:.6f}s",
            f"{time_ngram:.6f}s",
            "All same" if correction1 == correction2 == correction3 else "Different"
        ])

    print(tabulate(
        results,
        headers=["Misspelled", "Variant", "Dict Scan", "N-gram",
                 "Time (V)", "Time (D)", "Time (N)", "Agreement"],
        tablefmt='fancy_grid'
    ))


def main():
    print(f"\033[91m\n=== Task 1: Levenshtein Distance Calculation ===\033[0m")
    desired_word = "kitchen"
    words = ["kitchen", "kitten", "sitten", "sittin", "sitting"]
    for word in words:
        distance, dp_table = levenstein_distance_dp(desired_word, word)
        print(f"\033[94mSubtask:\033[0m Distance between '{desired_word}' and '{word}': {distance}")
        print_dp_table(desired_word, word, dp_table)
        print()

    print(f"\033[91m\n=== Task 2: Czech Dictionary Building ===\033[0m")
    filepath = "../input/task3/hp_1.txt"
    word_counts = build_czech_dictionary(filepath)
    print("\033[94mSubtask:\033[0m Most common words:")
    most_common_words = get_most_common_words(word_counts)
    print(tabulate(most_common_words, headers=["Word", "Frequency"], tablefmt='fancy_outline'))

    print(f"\033[91m\n=== Task 3: Word Variants Generation ===\033[0m")
    random_number_word_count = 3
    random_words = random.sample(list(word_counts.keys()), random_number_word_count)
    print(f"\033[94mSubtask:\033[0m Random words from the dataset: {random_words}")
    max_distance = 2
    for test_word in random_words:
        variants = generate_word_variants(test_word, max_distance)
        variant_count = len(variants)
        print(f"\n\033[94mSubtask:\033[0m Word: '{test_word}'")
        print(f"Number of variants (edit distance ≤ {max_distance}): {variant_count}")
        number_of_examples = 5
        sample_variants = random.sample(list(variants), number_of_examples)
        print(f"Sample of variants: {sample_variants}...")

    print(f"\033[91m\n=== Task 4: Autocorrection (Variant Approach) ===\033[0m")
    test_sentence = "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí."
    print(f"\n\033[94mSubtask:\033[0m Original sentence: {test_sentence}\n")

    corrected_sentence = autocorrect_sentence(test_sentence, word_counts, approach='variant')
    print(f"\033[94mSubtask:\033[0m Corrected sentence (Variant Approach): {corrected_sentence}\n")

    misspelled_words = ["oběť", "restauarci", "zpěť", "televezí"]
    print("\033[94mSubtask:\033[0m Individual word corrections:")
    for word in misspelled_words:
        correction = autocorrect_word_variant_approach(word, word_counts)
        print(f"'{word}' → '{correction}'")

    print(f"\033[91m\n=== Task 5: Alternative Approach (Dictionary Scan) ===\033[0m")
    print(f"\n\033[94mSubtask:\033[0m Original sentence: {test_sentence}\n")

    corrected_sentence = autocorrect_sentence(test_sentence, word_counts, approach='dict')
    print(f"\033[94mSubtask:\033[0m Corrected sentence (Dictionary Approach): {corrected_sentence}\n")

    print("\033[94mSubtask:\033[0m Individual word corrections:")
    for word in misspelled_words:
        correction = autocorrect_word_dict_approach(word, word_counts)
        print(f"'{word}' → '{correction}'")

    print(f"\033[91m\n=== Task 6: N-gram Enhanced Autocorrection ===\033[0m")
    data = load_data_to_lowercase("../input/task3/hp_1.txt")
    tokens = tokenize_data(data)

    print(f"\n\033[94mSubtask:\033[0m Original sentence: {test_sentence}\n")
    corrected_sentence = autocorrect_sentence_ngram(test_sentence, word_counts, tokens)
    print(f"\033[94mSubtask:\033[0m Corrected sentence (N-gram Enhanced): {corrected_sentence}\n")

    test_comparison_words = ["restauarci", "oběť", "oběd", "zpěť", "televezí",
                             "kavarna", "knjha", "kufrr", "fletn", "dračč",
                             "lod", "kost", "košť"]
    print("\033[94mSubtask:\033[0m Approach comparison:")
    compare_all_approaches(test_comparison_words, word_counts, tokens)


if __name__ == "__main__":
    main()
