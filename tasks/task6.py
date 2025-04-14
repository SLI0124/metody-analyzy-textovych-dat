import random
import string
import os
import matplotlib.pyplot as plt

SAVE_DIR = '../output/task6'


def unary_encode(n):
    """Encodes n in unary: 0→'0', n>0→(n-1)'1's followed by '0'. Examples: 1→'0', 2→'10', 3→'110'"""
    if n == 0:
        return '0'
    return '1' * (n - 1) + '0'


def unary_decode(code):
    """Decodes a unary-encoded string back to an integer. Examples: '0'→1, '10'→2, '110'→3"""
    if code == '0':
        return 1
    count = 0
    for char in code:
        if char == '1':
            count += 1
        elif char == '0':
            return count + 1
    raise ValueError("Invalid unary code")


def elias_gamma_encode(n):
    """
    Encodes n in Elias gamma encoding. Examples: 1→'0', 2→'10', 3→'110', 4→'1100'
    1. Find the largest k such that 2^k <= n
    2. Write k zeros followed by a 1
    3. Write the binary representation of n - 2^k padded with zeros to k bits
    """
    if n == 1:
        return '0'

    # Find the largest k where 2^k <= n
    k = 0
    power_of_two = 1
    while power_of_two * 2 <= n:
        k += 1
        power_of_two *= 2

    # Make the prefix - k zeros followed by one 1
    prefix = '0' * k + '1'

    # Calculate the remainder and convert to binary
    remainder = n - power_of_two

    # Convert to binary with padding
    if remainder == 0:
        suffix = '0' * k
    else:
        # Convert to binary
        suffix = bin(remainder)[2:]  # Remove '0b' prefix
        # Pad with leading zeros
        suffix = suffix.zfill(k)

    return prefix + suffix


def elias_gamma_decode(code):
    """
    Decodes an Elias gamma encoded string back to an integer.
    1. Find the position of the first '1' in the code
    2. The number of leading zeros gives k
    3. The next k bits give the binary representation of n - 2^k
    """
    if code == '0':
        return 1

    # Count leading zeros to find k
    k = 0
    for character in code:
        if character == '0':
            k = k + 1
        else:  # When we find a '1'
            break

    # Special case: if the first character is '1', then k=0
    if k == 0:
        return 1

    # Calculate the value of 2^k
    power_of_two = 1
    for _ in range(k):
        power_of_two = power_of_two * 2

    # Extract the next k bits after the first '1'
    binary_part = ""
    if k + 1 + k <= len(code):
        binary_part = code[k + 1:k + 1 + k]

    # Convert binary string to integer manually
    remainder = 0
    if binary_part:
        place_value = 1
        # Process binary digits from right to left
        for digit in reversed(binary_part):
            if digit == '1':
                remainder = remainder + place_value
            place_value = place_value * 2

    # Calculate final number
    n = power_of_two + remainder

    return n


def fibonacci_encode(n):
    """
    1. Find the largest Fibonacci number <= n.
    2. Subtract it from n and mark its position in the codeword.
    3. Repeat until n becomes 0.
    4. Append an additional '1' to the codeword.
    """
    # Generate Fibonacci sequence up to n
    fib = [1, 2]
    while fib[-1] <= n:
        fib.append(fib[-1] + fib[-2])

    fib.pop()  # Remove the last Fibonacci number if it exceeds n

    # Encode n
    codeword = ['0'] * len(fib)
    for i in range(len(fib) - 1, -1, -1):
        if fib[i] <= n:
            n -= fib[i]
            codeword[i] = '1'

    # Append the additional '1'
    return ''.join(codeword) + '1'


def fibonacci_decode(code):
    """
    1. Use the Fibonacci sequence to calculate the value of the codeword.
    2. Ignore the last '1' in the codeword (terminating bit).
    """
    # Generate Fibonacci sequence up to the length of the code
    fib = [1, 2]
    while len(fib) < len(code) - 1:  # Exclude the terminating '1'
        fib.append(fib[-1] + fib[-2])

    # Decode the codeword
    n = 0
    for i in range(len(code) - 1):  # Exclude the terminating '1'
        if code[i] == '1':
            n += fib[i]

    return n


def generate_random_words(num_words):
    words = set()

    while len(words) < num_words:
        word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        words.add(word)
    return list(words)


def generate_random_pairs(words, num_docs, num_pairs):
    pairs = set()

    while len(pairs) < num_pairs:
        word = random.choice(words)
        doc_id = random.randint(1, num_docs)
        pairs.add((word, doc_id))
    return list(pairs)


def build_inverted_index(pairs):
    inverted_index = {}

    for word, doc_id in pairs:
        if word not in inverted_index:
            inverted_index[word] = []
        inverted_index[word].append(doc_id)

    for word in inverted_index:
        inverted_index[word].sort()

    return inverted_index


def encode_doc_ids(inverted_index, encoding_function):
    encoded_index = {}
    for word, doc_ids in inverted_index.items():
        differences = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            differences.append(doc_ids[i] - doc_ids[i - 1])

        encoded_index[word] = [encoding_function(diff) for diff in differences]

    return encoded_index


def plot_encoding_sizes(unary_size, elias_size, fibonacci_size):
    encodings = ['Unary', 'Elias Gamma', 'Fibonacci']
    sizes = [unary_size, elias_size, fibonacci_size]

    plt.figure(figsize=(8, 6))
    plt.bar(encodings, sizes, color=['blue', 'green', 'orange'])
    plt.title('Comparison of Encoded Index Sizes')
    plt.xlabel('Encoding Type')
    plt.ylabel('Total Size (bits)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_path = os.path.join(SAVE_DIR, 'encoding_sizes.png')

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    plt.savefig(plot_path)


def main():
    random.seed(42)
    test_numero = 7
    print(f"Chosen number: \033[94m{test_numero}\033[0m")

    print("\033[91mUnary Encoding and Decoding\033[0m")
    unary_encoded = unary_encode(test_numero)
    print(f"Unary encoding: \033[93m{unary_encoded}\033[0m")
    decoded_number = unary_decode(unary_encoded)
    print(f"Decoded number from unary: \033[93m{decoded_number}\033[0m")

    print("\033[91mElias Gamma Encoding\033[0m")
    elias_encoded = elias_gamma_encode(test_numero)
    print(f"Elias gamma encoding: \033[93m{elias_encoded}\033[0m")
    decoded_number = elias_gamma_decode(elias_encoded)
    print(f"Decoded number from Elias gamma: \033[93m{decoded_number}\033[0m")

    print("\033[91mFibonacci Encoding\033[0m")
    fibonacci_encoded = fibonacci_encode(test_numero)
    print(f"Fibonacci encoding: \033[93m{fibonacci_encoded}\033[0m")
    decoded_number = fibonacci_decode(fibonacci_encoded)
    print(f"Decoded number from Fibonacci: \033[93m{decoded_number}\033[0m")

    print("\033[91mSimulating Data and Encoding\033[0m")
    num_words = 1000
    num_docs = 10000
    num_pairs = 1000000

    words = generate_random_words(num_words)
    pairs = generate_random_pairs(words, num_docs, num_pairs)
    inverted_index = build_inverted_index(pairs)

    unary_encoded_index = encode_doc_ids(inverted_index, unary_encode)
    elias_encoded_index = encode_doc_ids(inverted_index, elias_gamma_encode)
    fibonacci_encoded_index = encode_doc_ids(inverted_index, fibonacci_encode)

    unary_size = sum(len(''.join(encoded)) for encoded in unary_encoded_index.values())
    elias_size = sum(len(''.join(encoded)) for encoded in elias_encoded_index.values())
    fibonacci_size = sum(len(''.join(encoded)) for encoded in fibonacci_encoded_index.values())

    print(f"Unary encoding total size: \033[93m{unary_size} bits\033[0m")
    print(f"Elias gamma encoding total size: \033[93m{elias_size} bits\033[0m")
    print(f"Fibonacci encoding total size: \033[93m{fibonacci_size} bits\033[0m")

    plot_encoding_sizes(unary_size, elias_size, fibonacci_size)


if __name__ == "__main__":
    main()
