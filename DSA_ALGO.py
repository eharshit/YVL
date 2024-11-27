import hashlib
import random

# Helper functions
def mod_inverse(a, m):
    """Compute the modular inverse of a under modulo m."""
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        a, m = m, a % m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

# DSA Parameters
p = 23  # Small prime for simplicity
q = 11  # Small prime divisor of p-1
g = 2   # Generator

# Private and Public Key Generation
x = random.randint(1, q - 1)  # Private key
y = pow(g, x, p)              # Public key

print("Private Key (x):", x)
print("Public Key (y):", y)

# Signing
message = "Hello, DSA!".encode()
H = int(hashlib.sha256(message).hexdigest(), 16) % q  # Hash of the message
k = random.randint(1, q - 1)  # Random k
r = pow(g, k, p) % q
k_inv = mod_inverse(k, q)
s = (k_inv * (H + x * r)) % q

print("Signature (r, s):", (r, s))

# Verifying
w = mod_inverse(s, q)
u1 = (H * w) % q
u2 = (r * w) % q
v = ((pow(g, u1, p) * pow(y, u2, p)) % p) % q

print("Verification:", "Valid" if v == r else "Invalid")
