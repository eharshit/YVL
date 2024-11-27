import numpy as np

def matrix_mod_inv(matrix, modulus):
    det = int(np.round(np.linalg.det(matrix)))
    det_inv = pow(det, -1, modulus)
    matrix_mod_inv = (
        det_inv * np.round(det * np.linalg.inv(matrix)).astype(int) % modulus
    )
    return matrix_mod_inv

def hill_cipher_encrypt(plain_text, key_matrix):
    plain_text = plain_text.replace(" ", "").upper()
    while len(plain_text) % key_matrix.shape[0] != 0:
        plain_text += "X"
    plain_nums = [ord(char) - ord("A") for char in plain_text]
    plain_matrix = np.array(plain_nums).reshape(-1, key_matrix.shape[0])
    cipher_matrix = np.dot(plain_matrix, key_matrix) % 26
    cipher_text = "".join([chr(num + ord("A")) for num in cipher_matrix.flatten()])
    return cipher_text

def hill_cipher_decrypt(cipher_text, key_matrix):
    cipher_text = cipher_text.replace(" ", "").upper()
    cipher_nums = [ord(char) - ord("A") for char in cipher_text]
    cipher_matrix = np.array(cipher_nums).reshape(-1, key_matrix.shape[0])
    key_matrix_inv = matrix_mod_inv(key_matrix, 26)
    plain_matrix = np.dot(cipher_matrix, key_matrix_inv) % 26
    plain_text = "".join([chr(int(num) + ord("A")) for num in plain_matrix.flatten()])
    return plain_text

key_matrix = np.array([[6, 24, 1], [13, 16, 10], [20, 17, 15]])

plain_text = input("Enter the plain text: ").strip()
cipher_text = hill_cipher_encrypt(plain_text, key_matrix)
print(f"Encrypted: {cipher_text}")

decrypted_text = hill_cipher_decrypt(cipher_text, key_matrix)
print(f"Decrypted: {decrypted_text}")