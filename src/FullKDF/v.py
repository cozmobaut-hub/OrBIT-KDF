#!/usr/bin/env python3
import os, subprocess, tempfile
from main import derive_fractal_key_hyper as kdf

VD = "/home/haustintexas2/vault_18"
E = os.path.join(VD, "data_encrypted")
P = os.path.join(VD, "data_plain")

u = input("U: ").strip()
p = input("P: ").strip()
k = input("K: ").strip()

exp = kdf(u, p).hex()
if k != exp:
    print("NO")
    raise SystemExit(1)

os.makedirs(P, exist_ok=True)

with tempfile.NamedTemporaryFile("w", delete=False) as f:
    f.write(k + "\n")
    pf = f.name

try:
    subprocess.run(
        ["gocryptfs", f"-passfile={pf}", E, P],
        check=True,
    )
    print("OK", P)
finally:
    os.remove(pf)
