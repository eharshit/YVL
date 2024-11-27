import hashlib
def sha256(messsage):
    return hashlib.sha256(message.encode()).hexdigest()
message = "hello world"
print(sha256(message))