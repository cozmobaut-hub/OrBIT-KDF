#!/usr/bin/env python3
import hashlib
import struct
import numpy as np
import random
import string

# ---------- Old pipeline: compute_fractal_key ----------

def compute_fractal_key(username: str, password: str) -> bytes:
    h_user = hashlib.sha256(username.lower().encode()).digest()
    xu0 = struct.unpack('>d', h_user[:8])[0] % (2 * np.pi)
    pu0 = struct.unpack('>d', h_user[8:16])[0] % (2 * np.pi)
    Ku  = (struct.unpack('>d', h_user[16:24])[0] % 9.5) + 0.5

    xu, pu = xu0, pu0
    for _ in range(5000):
        pu = (pu + Ku * np.sin(xu)) % (2 * np.pi)
        xu = (xu + pu) % (2 * np.pi)

    h_pass = hashlib.sha256(password.encode()).digest()
    xp = struct.unpack('>d', h_pass[:8])[0] % (2 * np.pi)
    pp = struct.unpack('>d', h_pass[8:16])[0] % (2 * np.pi)

    c = complex(xu, pu) + complex(xp, pp)
    z = 0j
    for _ in range(5000):
        z = z**2 + c
        if abs(z) > 2:
            break

    stats = struct.pack('ddd', xu, pu, abs(z))
    return hashlib.sha512(stats).digest()

# ---------- New pipeline: import from visualize ----------

from visualize import derive_fractal_key  # your SHAh KDF


# ---------- Shared test helpers ----------

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
        chars = string.ascii_letters + string.digits
        new_char = random.choice([c for c in chars if c != u[idx]])
        u2 = u[:idx] + new_char + u[idx+1:]

        k2 = kdf_func(u2, p)
        d = hamming_distance(k1, k2)
        total_dist += d
        total_bits += len(k1) * 8
        if (i + 1) % 200 == 0:
            print(f"[{label}] avalanche {i+1}/{avalanche_pairs} ...")

    print(f"Avalanche over {avalanche_pairs} flips: {100.0 * total_dist / total_bits:.2f}% bits changed")

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

    run_tests("old compute_fractal_key", compute_fractal_key)
    run_tests("new derive_fractal_key", derive_fractal_key)
