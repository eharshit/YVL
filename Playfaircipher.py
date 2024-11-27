def create_matrix(key):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    key = key.upper().replace("J", "I")
    matrix = []
    used = set()

    for char in key:
        if char not in used and char in alphabet:
            matrix.append(char)
            used.add(char)

    for char in alphabet:
        if char not in used:
            matrix.append(char)
            used.add(char)

    return [matrix[i:i + 5] for i in range(0, 25, 5)]

def preprocess_text(text):
    text = text.upper().replace("J", "I").replace(" ", "")
    result = []
    i = 0

    while i < len(text):
        a = text[i]
        if i + 1 < len(text):
            b = text[i + 1]
            if a == b:
                result.append(a + "X")
                i += 1
            else:
                result.append(a + b)
                i += 2
        else:
            result.append(a + "X")
            i += 1

    return result

def find_position(matrix, char):
    for row, line in enumerate(matrix):
        if char in line:
            return row, line.index(char)
    return None, None

def encrypt_digraph(matrix, digraph):
    a, b = digraph
    row_a, col_a = find_position(matrix, a)
    row_b, col_b = find_position(matrix, b)

    if row_a == row_b:
        return matrix[row_a][(col_a + 1) % 5] + matrix[row_b][(col_b + 1) % 5]
    elif col_a == col_b:
        return matrix[(row_a + 1) % 5][col_a] + matrix[(row_b + 1) % 5][col_b]
    else:
        return matrix[row_a][col_b] + matrix[row_b][col_a]

def decrypt_digraph(matrix, digraph):
    a, b = digraph
    row_a, col_a = find_position(matrix, a)
    row_b, col_b = find_position(matrix, b)

    if row_a == row_b:
        return matrix[row_a][(col_a - 1) % 5] + matrix[row_b][(col_b - 1) % 5]
    elif col_a == col_b:
        return matrix[(row_a - 1) % 5][col_a] + matrix[(row_b - 1) % 5][col_b]
    else:
        return matrix[row_a][col_b] + matrix[row_b][col_a]

def encrypt(plaintext, key):
    matrix = create_matrix(key)
    plaintext_digraphs = preprocess_text(plaintext)
    ciphertext = ""

    for digraph in plaintext_digraphs:
        ciphertext += encrypt_digraph(matrix, digraph)

    return ciphertext

def decrypt(ciphertext, key):
    matrix = create_matrix(key)
    ciphertext_digraphs = preprocess_text(ciphertext)
    plaintext = ""

    for digraph in ciphertext_digraphs:
        plaintext += decrypt_digraph(matrix, digraph)

    return plaintext


#key = "PLAYFAIR"
#plaintext = "HELLO WORLD"

key = input("Enter the key: ")
plaintext = input("Enter the plain text: ")


ciphertext = encrypt(plaintext, key)
decrypted_text = decrypt(ciphertext, key)

print(f"Key: {key}")
print(f"Plaintext: {plaintext}")
print(f"Ciphertext: {ciphertext}")
print(f"Decrypted Text: {decrypted_text}")