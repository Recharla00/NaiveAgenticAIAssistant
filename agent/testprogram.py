import hashlib
import mmh3  # Note: Requires pip install mmh3
import secrets


def main():
    data = "Hello World"

    # 1. SHA-256 (Cryptographic: Secure, Fixed 64-character Hexadecimal string)
    sha256_hash = hashlib.sha256(data.encode()).hexdigest()
    print(f"SHA-256: {sha256_hash}") # Output: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e

    # A random salt makes the hash different even when the input is unchanged.
    salt = secrets.token_bytes(1)  # 8 bytes of random salt
    print(f"Salt: {salt}")
    salted_hash = hashlib.sha256(salt + data.encode()).hexdigest()
    print(f"Salt (hex): {salt.hex()}")
    print(f"Salted SHA-256: {salted_hash}")

    # 2. MurmurHash3 (Non-Cryptographic: Super fast, returns a 32-bit Integer)
    murmur_hash = mmh3.hash(data)
    print(f"MurmurHash3: {murmur_hash}")

    # 1. Raw Binary Representation (Removes leading zeros)
    print(f"Binary: {bin(murmur_hash)}") # Output: -232047814 (An integer index perfectly suited for data lookup)
    print(f"Bit Count: {murmur_hash.bit_count()} ") # Returns 16 (The count of '1' bits inside the number)
    print(f"Bit length: {murmur_hash.bit_length()}") # Returns 29 (The actual number of total bits needed to represent it)

if __name__ == "__main__":
    main()