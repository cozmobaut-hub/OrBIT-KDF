#!/usr/bin/env python3
import time
import math
import random

from main import derive_fractal_key  # your SHAh KDF


# ---------- Helpers ----------

def random_cred(length=8):
    import string
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


# ---------- 1) Chi-square on first byte ----------

def chi_square_first_byte(kdf_func, num_samples=50000):
    counts = [0] * 256  # frequency of each possible byte value 0..255

    for i in range(num_samples):
        u = random_cred(8)
        p = random_cred(8)
        h = kdf_func(u, p)
        b = h[0]  # first 8 bits
        counts[b] += 1
        if (i + 1) % 5000 == 0:
            print(f"[chi-square] {i+1}/{num_samples} ...")

    expected = num_samples / 256.0
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    df = 255

    print(f"\n=== Chi-square on first byte over {num_samples} samples ===")
    print(f"chi^2 = {chi2:.2f} (df = {df})")
    print("first 16 bucket counts:", counts[:16])


# ---------- 2) Throughput measurement ----------

def measure_throughput(kdf_func, duration=1.0):
    start = time.time()
    iters = 0
    while time.time() - start < duration:
        u = f"u{iters}"
        p = f"p{iters}"
        kdf_func(u, p)
        iters += 1
    elapsed = time.time() - start
    rate = iters / elapsed
    print(f"\n=== Throughput over {elapsed:.2f} s ===")
    print(f"Total evals: {iters}")
    print(f"Rate: {rate:,.0f} KDF/s")


# ---------- Main ----------

if __name__ == "__main__":
    random.seed(0)

    # 1) Chi-square uniformity test on first byte
    chi_square_first_byte(derive_fractal_key, num_samples=50000)

    # 2) Throughput test
    measure_throughput(derive_fractal_key, duration=1.0)
