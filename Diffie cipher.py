import random

def modular_exponential(base ,exponent,modulus):
    return pow(base,exponent,modulus)

def generate_private_key(prime):
    return random.randint(1,prime - 1)

def generate_public_key(private_key,base,prime):
    return modular_exponential(base,private_key,prime)

def generate_shared_secret_key(private_key,public_key,prime):
    return modular_exponential(public_key,private_key,prime)

def main():
   prime = int(input(f"Enter largest prime number:"))
   base = int(input(f"Enter base:"))

   private_key_a= generate_private_key(prime)
   private_key_b = generate_private_key(prime)

   public_key_a = generate_public_key(private_key_a,base,prime)
   public_key_b = generate_public_key(private_key_b,base,prime)

   print(f"private key A:{private_key_a}")
   print(f"public key A:{public_key_a}")
   print(f"private key B:{private_key_b}")
   print(f"public key B:{public_key_b}")

   shared_secret_key_a = generate_shared_secret_key(private_key_a,public_key_b,prime)
   shared_secret_key_b = generate_shared_secret_key(private_key_b,public_key_a,prime)

   print(f"Shared secret key A:{shared_secret_key_a}")
   print(f"Shared secret key B:{shared_secret_key_b}")

   if shared_secret_key_a == shared_secret_key_b:
      print("shared secret key are successfully established")
   else:
      print("shared secret key does not match")

main()