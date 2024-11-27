import hashlib

def md5_hash_string(input_string):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the bytes of the string
    md5_hash.update(input_string.encode('utf-8'))

    # Return the hexadecimal digest of the hash
    return md5_hash.hexdigest()

# Example usage
input_string = "Hello, world!"
print(f"MD5 hash of '{input_string}': {md5_hash_string(input_string)}")