import sys
from getpass import getpass
from visualize import derive_fractal_key  # this is the function we added earlier


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 shah_kdf.py <username>", file=sys.stderr)
        sys.exit(1)

    username = sys.argv[1]
    print(username)
    password = getpass("Password: ")

    key = derive_fractal_key(username, password)  # 64 bytes = 512 bits

    # Print full 512-bit key as hex (128 hex chars)
    print(key.hex())


if __name__ == "__main__":
    main()