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


def main():
    test_numero = 7
    print(f"Chosen number: \033[94m{test_numero}\033[0m")

    print("\033[91mUnary Encoding and Decoding\033[0m")
    unary_encoded = unary_encode(test_numero)
    print(f"Unary encoding: \033[93m{unary_encoded}\033[0m")
    decoded_number = unary_decode(unary_encoded)
    print(f"Decoded number from unary: \033[93m{decoded_number}\033[0m")


if __name__ == "__main__":
    main()
