import os
import random
import string
import hashlib
from collections import Counter

import numpy as np

import main  # your script with derive_fractal_key defined


def random_cred(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def hamming_distance(a: bytes, b: bytes) -> int:
    assert len(a) == len(b)
    dist = 0
    for x, y in zip(a, b):
        v = x ^ y
        dist += bin(v).count("1")
    return dist


def test_collision_rate(num_samples=10000):
    """
    Generate num_samples random creds, compute keys,
    and look for collisions.
    """
    seen = {}
    collisions = 0
    for i in range(num_samples):
        u = random_cred()
        p = random_cred()
        key = main.derive_fractal_key(u, p)
        if key in seen:
            collisions += 1
        else:
            seen[key] = (u, p)
        if (i + 1) % 1000 == 0:
            print(f"[collision test] {i+1}/{num_samples} ...")
    print(f"\nCollision test on {num_samples} samples:")
    print(f"  Collisions found: {collisions}")
    return collisions


def test_avalanche(num_pairs=2000):
    """
    Take random creds, flip 1 char in username or password,
    and measure average bit differences in the key.
    """
    total_bits = 0
    total_dist = 0

    for i in range(num_pairs):
        u = random_cred()
        p = random_cred()
        key1 = main.derive_fractal_key(u, p)

        # flip exactly one character in username
        idx = random.randrange(len(u))
        chars = string.ascii_letters + string.digits
        new_char = random.choice([c for c in chars if c != u[idx]])
        u2 = u[:idx] + new_char + u[idx+1:]

        key2 = main.derive_fractal_key(u2, p)
        dist = hamming_distance(key1, key2)
        total_dist += dist
        total_bits += len(key1) * 8

        if (i + 1) % 200 == 0:
            print(f"[avalanche test] {i+1}/{num_pairs} ...")

    avg_flip_pct = 100.0 * total_dist / total_bits
    print(f"\nAvalanche test on {num_pairs} username flips:")
    print(f"  Average bit difference: {avg_flip_pct:.2f}%")
    return avg_flip_pct


def test_bit_distribution(num_samples=10000):
    """
    Check per-bit 0/1 balance across many outputs.
    """
    key_len = len(main.derive_fractal_key("u0", "p0"))
    bit_counts = np.zeros(key_len * 8, dtype=int)

    for i in range(num_samples):
        u = random_cred()
        p = random_cred()
        key = main.derive_fractal_key(u, p)
        bits = ''.join(f"{byte:08b}" for byte in key)
        for j, b in enumerate(bits):
            if b == '1':
                bit_counts[j] += 1
        if (i + 1) % 1000 == 0:
            print(f"[bit dist] {i+1}/{num_samples} ...")

    print(f"\nBit distribution over {num_samples} samples:")
    for i in range(0, min(64, len(bit_counts)), 8):
        window = bit_counts[i:i+8]
        freqs = ", ".join(f"{c/num_samples:0.3f}" for c in window)
        print(f"  bits {i:3d}-{i+7:3d}: {freqs}")
    return bit_counts


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    test_collision_rate(num_samples=100000)
    test_avalanche(num_pairs=10000)
    test_bit_distribution(num_samples=10000)

