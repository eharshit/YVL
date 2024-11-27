def create_vigenere_matrix():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    matrix = []
    for i in range(26):
        row = alphabet[i:] + alphabet[:i]
        matrix.append(row)
    return matrix

def encrypt_vigenere(plain_text, keyword):
    plain_text = plain_text.upper().replace(" ", "")
    keyword = keyword.upper().replace(" ", "")
    repeated_keyword = []
    keyword_length = len(keyword)
    for i in range(len(plain_text)):
        repeated_keyword.append(keyword[i % keyword_length])
    matrix = create_vigenere_matrix()
    cipher_text = []
    for pt_char, kw_char in zip(plain_text, repeated_keyword):
        if pt_char.isalpha():
            row_index = ord(pt_char) - ord('A')
            col_index = ord(kw_char) - ord('A')
            cipher_char = matrix[row_index][col_index]
            cipher_text.append(cipher_char)
        else:
            cipher_text.append(pt_char)
    return "".join(cipher_text)

if __name__ == "__main__":
    plain_text = input("Enter the plaintext: ").strip()
    keyword = input("Enter the keyword: ").strip()
    encrypted_text = encrypt_vigenere(plain_text, keyword)
    print("Encrypted text:", encrypted_text)
print("DECYPHERED TEXT",plain_text)