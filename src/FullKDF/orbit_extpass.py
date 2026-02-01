#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read()
    # Log what we actually got
    with open("/tmp/orbit_extpass_debug.txt", "w") as f:
        f.write(repr(data))
    # For now, just echo it back to gocryptfs
    print(data.strip())  # whatever came in, goes out

if __name__ == "__main__":
    main()
