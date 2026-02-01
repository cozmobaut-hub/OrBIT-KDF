#!/usr/bin/env python3
import random
import numpy as np

from main import derive_fractal_key_hyper  # Hyper KDF: Chirikov+DeJong+Lorenz + Julias


def random_cred(length=8):
    import string
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def hamming_distance(a: bytes, b: bytes) -> int:
    assert len(a) == len(b)
    dist = 0
    for x, y in zip(a, b):
        v = x ^ y
        dist += bin(v).count("1")
    return dist


def run_tests(label, kdf_func,
              collision_samples=200000,
              avalanche_pairs=20000,
              bit_samples=50000):
    print(f"\n==== Testing {label} ====")

    # 1) Collisions
    seen = {}
    collisions = 0
    for i in range(collision_samples):
        u = random_cred()
        p = random_cred()
        key = kdf_func(u, p)
        if key in seen:
            collisions += 1
        else:
            seen[key] = (u, p)
        if (i + 1) % 2000 == 0:
            print(f"[{label}] collisions {i+1}/{collision_samples} ...")
    print(f"Collision test on {collision_samples}: {collisions} collisions")

    # 2) Avalanche
    total_bits = 0
    total_dist = 0
    for i in range(avalanche_pairs):
        u = random_cred()
        p = random_cred()
        k1 = kdf_func(u, p)

        idx = random.randrange(len(u))
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        new_char = random.choice([c for c in chars if c != u[idx]])
        u2 = u[:idx] + new_char + u[idx+1:]

        k2 = kdf_func(u2, p)
        d = hamming_distance(k1, k2)
        total_dist += d
        total_bits += len(k1) * 8
        if (i + 1) % 200 == 0:
            print(f"[{label}] avalanche {i+1}/{avalanche_pairs} ...")

    print(f"Avalanche over {avalanche_pairs} flips: "
          f"{100.0 * total_dist / total_bits:.2f}% bits changed")

    # 3) Bit distribution
    key_len = len(kdf_func("u0", "p0"))
    bit_counts = np.zeros(key_len * 8, dtype=int)
    for i in range(bit_samples):
        u = random_cred()
        p = random_cred()
        k = kdf_func(u, p)
        bits = ''.join(f"{byte:08b}" for byte in k)
        for j, b in enumerate(bits):
            if b == "1":
                bit_counts[j] += 1
        if (i + 1) % 1000 == 0:
            print(f"[{label}] bits {i+1}/{bit_samples} ...")

    print(f"Bit frequencies (first 64 bits) over {bit_samples} samples:")
    for i in range(0, 64, 8):
        window = bit_counts[i:i+8]
        freqs = ", ".join(f"{c/bit_samples:0.3f}" for c in window)
        print(f"  bits {i:2d}-{i+7:2d}: {freqs}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    run_tests("Hyper derive_fractal_key_hyper", derive_fractal_key_hyper)
