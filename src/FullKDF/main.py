#!/usr/bin/env python3
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import string
import random

# ---------- Global params ----------

K_ch = 2.5  # Chirikov standard map parameter
a_dj, b_dj, c_dj, d_dj = 2.01, -2.53, 1.61, -0.33  # De Jong
sigma_l, rho_l, beta_l = 10.0, 28.0, 8.0 / 3.0     # Lorenz


# ---------- Hash helpers ----------

def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def seed_to_float_pair(seed: bytes, scale: float) -> tuple[float, float]:
    a = int.from_bytes(seed[0:8], "big")
    b = int.from_bytes(seed[8:16], "big")
    x = (a / 2**64) * (2 * scale) - scale
    y = (b / 2**64) * (2 * scale) - scale
    return x, y


def seed_to_float_triple(seed: bytes, scale: float) -> tuple[float, float, float]:
    a = int.from_bytes(seed[0:8], "big")
    b = int.from_bytes(seed[8:16], "big")
    c = int.from_bytes(seed[16:24], "big")
    x = (a / 2**64) * (2 * scale) - scale
    y = (b / 2**64) * (2 * scale) - scale
    z = (c / 2**64) * (2 * scale) - scale
    return x, y, z


# ---------- Chirikov (standard map) ----------

def chirikov_step(x, p, K=K_ch):
    p_next = p + K * np.sin(x)
    x_next = x + p_next
    x_next = (x_next + np.pi) % (2 * np.pi) - np.pi
    p_next = (p_next + np.pi) % (2 * np.pi) - np.pi
    return x_next, p_next


def chirikov_trajectory(username: str, n_iter: int, scale: float = np.pi):
    x, p = seed_to_float_pair(sha256_bytes("CH:" + username), scale)
    xs, ps = [], []
    for _ in range(n_iter):
        x, p = chirikov_step(x, p, K=K_ch)
        xs.append(x)
        ps.append(p)
    return np.array(xs), np.array(ps)


# ---------- De Jong map ----------

def dejong_step(x, y, a=a_dj, b=b_dj, c=c_dj, d=d_dj):
    x_next = np.sin(a * y) - np.cos(b * x)
    y_next = np.sin(c * x) - np.cos(d * y)
    return x_next, y_next


def dejong_trajectory(username: str, n_iter: int, scale: float):
    x, y = seed_to_float_pair(sha256_bytes("DJ:" + username), scale)
    xs, ys = [], []
    for _ in range(n_iter):
        x, y = dejong_step(x, y)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# ---------- Lorenz system ----------

def lorenz_step(x, y, z, sigma=sigma_l, rho=rho_l, beta=beta_l, dt=0.01):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    x_next = x + dx * dt
    y_next = y + dy * dt
    z_next = z + dz * dt
    return x_next, y_next, z_next


def lorenz_trajectory(username: str, n_iter: int, scale: float, dt: float = 0.01):
    x, y, z = seed_to_float_triple(sha256_bytes("LZ:" + username), scale)
    xs, ys, zs = [], [], []
    for _ in range(n_iter):
        x, y, z = lorenz_step(x, y, z, dt=dt)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)


def lorenz_trajectory_for_stability(username: str,
                                    n_iter: int,
                                    scale: float,
                                    dt: float = 0.01):
    return lorenz_trajectory(username, n_iter=n_iter, scale=scale, dt=dt)


# ---------- Julia set + orbit ----------

def julia_set_grid(c: complex,
                   x_min=-1.5, x_max=1.5,
                   y_min=-1.5, y_max=1.5,
                   N=900, max_iter=300):
    xs = np.linspace(x_min, x_max, N)
    ys = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    img = np.zeros(Z.shape, dtype=int)

    z = Z.copy()
    mask = np.ones(Z.shape, dtype=bool)
    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c
        escaped = (np.abs(z) > 2.0) & mask
        img[escaped] = i
        mask &= ~escaped
        if not mask.any():
            break
    img[mask] = max_iter
    return xs, ys, img


def julia_trajectory(seed_str: str, c: complex, n_iter: int = 1500):
    seed = sha256_bytes(seed_str)
    a = int.from_bytes(seed[0:8], "big")
    b = int.from_bytes(seed[8:16], "big")
    z = complex((a / 2**64) * 2.0 - 1.0,
                (b / 2**64) * 2.0 - 1.0)
    zs = []
    for _ in range(n_iter):
        z = z * z + c
        zs.append(z)
        if abs(z) > 2.0 * 1e3:
            break
    return np.array(zs)


# ---------- Stability Monte Carlo (Lorenz) ----------

def is_stable_orbit(xs, ys, zs, radius=2000000.0, lyap_thresh=0.0):
    r = np.sqrt(xs**2 + ys**2 + zs**2)
    if r.max() > radius:
        return False

    dx = np.diff(xs)
    dy = np.diff(ys)
    dz = np.diff(zs)
    step = np.sqrt(dx**2 + dy**2 + dz**2)

    step[step == 0] = 1e-16
    ratios = step[1:] / step[:-1]
    lyap_est = np.mean(np.log(np.abs(ratios)))

    return lyap_est < lyap_thresh


def monte_carlo_stability(num_samples=1000,
                          uname_len=8,
                          scale=2.0,
                          n_iter=5000,
                          radius=50.0,
                          lyap_thresh=0.0,
                          dt: float = 0.01):
    chars = string.ascii_letters + string.digits
    stable = 0
    unstable = 0

    for _ in range(num_samples):
        uname = ''.join(random.choice(chars) for _ in range(uname_len))
        xs, ys, zs = lorenz_trajectory_for_stability(
            uname, n_iter=n_iter, scale=scale, dt=dt
        )
        if is_stable_orbit(xs, ys, zs, radius=radius, lyap_thresh=lyap_thresh):
            stable += 1
        else:
            unstable += 1

    total = stable + unstable
    pct_stable = 100.0 * stable / total
    pct_unstable = 100.0 * unstable / total
    print(f"\n=== Lorenz Monte Carlo stability over {total} random username orbits ===")
    print(f"Stable:   {stable} ({pct_stable:.2f}%)")
    print(f"Unstable: {unstable} ({pct_unstable:.2f}%)\n")


# ---------- Master visualization: each trajectory separate ----------

def visualize_all(username: str, password: str):
    xs_ch, ps_ch = chirikov_trajectory(username, n_iter=4000, scale=np.pi)
    xs_dj, ys_dj = dejong_trajectory(username, n_iter=4000, scale=2.0)
    xs_lz, ys_lz, zs_lz = lorenz_trajectory(username, n_iter=20000, scale=2.0, dt=0.005)

    # Chirikov Julia
    x_ch_f, p_ch_f = xs_ch[-1], ps_ch[-1]
    c1 = complex(np.clip(np.tanh(x_ch_f) * 0.8, -0.8, 0.8),
                 np.clip(np.tanh(p_ch_f) * 0.8, -0.8, 0.8))
    zs_j1 = julia_trajectory("J1:" + password, c=c1, n_iter=1500)

    # De Jong Julia
    x_dj_f, y_dj_f = xs_dj[-1], ys_dj[-1]
    c2 = complex(np.clip(np.tanh(x_dj_f) * 0.8, -0.8, 0.8),
                 np.clip(np.tanh(y_dj_f) * 0.8, -0.8, 0.8))
    zs_j2 = julia_trajectory("J2:" + password, c=c2, n_iter=1500)

    # Lorenz Julia
    x_lz_f, y_lz_f, z_lz_f = xs_lz[-1], ys_lz[-1], zs_lz[-1]
    c3 = complex(np.clip(np.tanh(x_lz_f) * 0.6, -0.6, 0.6),
                 np.clip(np.tanh(y_lz_f) * 0.6, -0.6, 0.6))
    zs_j3 = julia_trajectory("J3:" + password, c=c3, n_iter=1500)

    fig = plt.figure(figsize=(22, 10))

    # Top row: username trajectories
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(xs_ch, ps_ch, s=1.5, alpha=0.8, color="cyan")
    ax1.set_title(f"Chirikov orbit (x,p)\nusername='{username}'")
    ax1.set_xlabel("x")
    ax1.set_ylabel("p")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(xs_dj, ys_dj, color="lime", linewidth=0.4)
    ax2.scatter(xs_dj, ys_dj, s=1.0, color="lime", alpha=0.7)
    ax2.set_title("De Jong orbit (x,y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.plot(xs_lz, ys_lz, zs_lz, lw=0.5, color="orange")
    ax3.set_title("Lorenz orbit (x,y,z)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    # Bottom row: Julia trajectories, clamped to [-2,2]^2 so they look like real Julias
    ax4 = fig.add_subplot(2, 3, 4)
    if len(zs_j1) > 0:
        j1 = zs_j1[np.abs(zs_j1) <= 2.0]
        ax4.scatter(j1.real, j1.imag, s=3.0, color="magenta",
                    edgecolors="black", linewidths=0.3, alpha=0.9)
    ax4.set_title(f"Julia 1 from Chirikov\nc1={c1.real:.3f}+{c1.imag:.3f}i")
    ax4.set_xlabel("Re(z)")
    ax4.set_ylabel("Im(z)")
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)

    ax5 = fig.add_subplot(2, 3, 5)
    if len(zs_j2) > 0:
        j2 = zs_j2[np.abs(zs_j2) <= 2.0]
        ax5.scatter(j2.real, j2.imag, s=3.0, color="cyan",
                    edgecolors="black", linewidths=0.3, alpha=0.9)
    ax5.set_title(f"Julia 2 from De Jong\nc2={c2.real:.3f}+{c2.imag:.3f}i")
    ax5.set_xlabel("Re(z)")
    ax5.set_ylabel("Im(z)")
    ax5.set_xlim(-2, 2)
    ax5.set_ylim(-2, 2)

    ax6 = fig.add_subplot(2, 3, 6)
    if len(zs_j3) > 0:
        j3 = zs_j3[np.abs(zs_j3) <= 2.0]
        ax6.scatter(j3.real, j3.imag, s=3.0, color="yellow",
                    edgecolors="black", linewidths=0.3, alpha=0.9)
    ax6.set_title(f"Julia 3 from Lorenz\nc3={c3.real:.3f}+{c3.imag:.3f}i")
    ax6.set_xlabel("Re(z)")
    ax6.set_ylabel("Im(z)")
    ax6.set_xlim(-2, 2)
    ax6.set_ylim(-2, 2)

    plt.tight_layout()
    plt.show()

    global c1_g, c2_g, c3_g, zs_j1_g, zs_j2_g, zs_j3_g
    c1_g, c2_g, c3_g = c1, c2, c3
    zs_j1_g, zs_j2_g, zs_j3_g = zs_j1, zs_j2, zs_j3


# ---------- Combined 6‑stage curve embedded in 3D ----------

def visualize_combined_hyper(username: str, password: str):
    xs_ch, ps_ch = chirikov_trajectory(username, n_iter=4000, scale=np.pi)
    xs_dj, ys_dj = dejong_trajectory(username, n_iter=4000, scale=2.0)
    xs_lz, ys_lz, zs_lz = lorenz_trajectory(username, n_iter=20000, scale=2.0, dt=0.005)

    x_ch_f, p_ch_f = xs_ch[-1], ps_ch[-1]
    c1 = complex(np.tanh(x_ch_f) * 0.8, np.tanh(p_ch_f) * 0.8)
    zs_j1 = julia_trajectory("J1:" + password, c=c1, n_iter=1500)

    x_dj_f, y_dj_f = xs_dj[-1], ys_dj[-1]
    c2 = complex(np.tanh(x_dj_f) * 0.8, np.tanh(y_dj_f) * 0.8)
    zs_j2 = julia_trajectory("J2:" + password, c=c2, n_iter=1500)

    x_lz_f, y_lz_f, z_lz_f = xs_lz[-1], ys_lz[-1], zs_lz[-1]
    c3 = complex(np.tanh(x_lz_f) * 0.6, np.tanh(y_lz_f) * 0.6)
    zs_j3 = julia_trajectory("J3:" + password, c=c3, n_iter=1500)

    L = min(
        len(xs_ch), len(ps_ch),
        len(xs_dj), len(ys_dj),
        len(xs_lz), len(ys_lz), len(zs_lz),
        len(zs_j1), len(zs_j2), len(zs_j3),
    )

    xs_ch_s = xs_ch[:L]
    ps_ch_s = ps_ch[:L]
    xs_dj_s = xs_dj[:L]
    ys_dj_s = ys_dj[:L]
    xs_lz_s = xs_lz[:L]
    ys_lz_s = ys_lz[:L]
    zs_lz_s = zs_lz[:L]
    r_j1 = np.abs(zs_j1[:L])
    r_j2 = np.abs(zs_j2[:L])
    r_j3 = np.abs(zs_j3[:L])

    X = xs_ch_s + xs_dj_s + xs_lz_s
    Y = ps_ch_s + ys_dj_s + ys_lz_s
    Z = r_j1 + r_j2 + r_j3

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(X, Y, Z, lw=0.8)
    ax.set_title("Combined 6‑stage trajectory (embedded in 3D)")
    ax.set_xlabel("X = x_ch + x_dj + x_lz")
    ax.set_ylabel("Y = p_ch + y_dj + y_lz")
    ax.set_zlabel("Z = |J1| + |J2| + |J3|")
    plt.tight_layout()
    plt.show()


# ---------- Hyper KDF ----------

def derive_fractal_key_hyper(username: str, password: str) -> bytes:
    xs_ch, ps_ch = chirikov_trajectory(username, n_iter=4000, scale=np.pi)
    x_ch_f, p_ch_f = xs_ch[-1], ps_ch[-1]
    c1 = complex(np.tanh(x_ch_f) * 0.8, np.tanh(p_ch_f) * 0.8)
    zs_j1 = julia_trajectory("J1:" + password, c=c1, n_iter=1500)

    xs_dj, ys_dj = dejong_trajectory(username, n_iter=4000, scale=2.0)
    x_dj_f, y_dj_f = xs_dj[-1], ys_dj[-1]
    c2 = complex(np.tanh(x_dj_f) * 0.8, np.tanh(y_dj_f) * 0.8)
    zs_j2 = julia_trajectory("J2:" + password, c=c2, n_iter=1500)

    xs_lz, ys_lz, zs_lz = lorenz_trajectory(
        username, n_iter=20000, scale=2.0, dt=0.005
    )
    x_lz_f, y_lz_f, z_lz_f = xs_lz[-1], ys_lz[-1], zs_lz[-1]
    c3 = complex(np.tanh(x_lz_f) * 0.6, np.tanh(y_lz_f) * 0.6)
    global c
    c = c3
    zs_j3 = julia_trajectory("J3:" + password, c=c3, n_iter=1500)
    global zs_p
    zs_p = zs_j3

    L = min(
        512,
        len(xs_ch), len(ps_ch),
        len(xs_dj), len(ys_dj),
        len(xs_lz), len(ys_lz), len(zs_lz),
        len(zs_j1), len(zs_j2), len(zs_j3),
    )

    xs_ch_s = xs_ch[:L]
    ps_ch_s = ps_ch[:L]
    xs_dj_s = xs_dj[:L]
    ys_dj_s = ys_dj[:L]
    xs_lz_s = xs_lz[:L]
    ys_lz_s = ys_lz[:L]
    zs_lz_s = zs_lz[:L]
    r_j1 = np.abs(zs_j1[:L])
    r_j2 = np.abs(zs_j2[:L])
    r_j3 = np.abs(zs_j3[:L])

    buf = bytearray()
    for xc, pc, xd, yd, xl, yl, zl, r1, r2, r3 in zip(
        xs_ch_s, ps_ch_s,
        xs_dj_s, ys_dj_s,
        xs_lz_s, ys_lz_s, zs_lz_s,
        r_j1, r_j2, r_j3,
    ):
        buf += np.float64(xc).tobytes()
        buf += np.float64(pc).tobytes()
        buf += np.float64(xd).tobytes()
        buf += np.float64(yd).tobytes()
        buf += np.float64(xl).tobytes()
        buf += np.float64(yl).tobytes()
        buf += np.float64(zl).tobytes()
        buf += np.float64(r1).tobytes()
        buf += np.float64(r2).tobytes()
        buf += np.float64(r3).tobytes()

    return hashlib.sha512(bytes(buf)).digest()


# ---------- Find good demo path (all 6 trajectories) ----------

def random_cred(length=8):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def orbit_spread_score(*arrays):
    spread = sum(float(a.std()) for a in arrays)
    return spread - np.exp(-spread)


import sys

def progress_bar(iteration, total, prefix="Passwords", length=40):
    if total <= 0:
        return
    frac = iteration / total
    filled = int(length * frac)
    bar = "#" * filled + "-" * (length - filled)
    percent = int(frac * 100)
    sys.stdout.write(f"\r{prefix}: [{bar}] {percent:3d}% ({iteration}/{total})")
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")


import sys

def progress_bar(iteration, total, prefix="Passwords", length=40):
    if total <= 0:
        return
    frac = iteration / total
    filled = int(length * frac)
    bar = "#" * filled + "-" * (length - filled)
    percent = int(frac * 100)
    sys.stdout.write(f"\r{prefix}: [{bar}] {percent:3d}% ({iteration}/{total})")
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")


def find_good_demo_path_hyper(
    uname_tries=5000,   # kept for signature compatibility, not used as hard cap
    pwd_tries=5000,     # used as batch size for progress bar
    uname_len=10,
    pwd_len=12,
    n_iter_ch=4000,
    n_iter_dj=4000,
    n_iter_lz=10000,
    julia_len_thresh=1400,
    scale_ch=np.pi,
    scale_dj=2.0,
    scale_lz=2.0,
    dt_lz=0.01,
    rng_seed=0,  #
):
    random.seed(rng_seed)
    chars = string.ascii_letters + string.digits

    # -------- 1) LOOP until we find a good username --------
    print("\n[+] Searching for demo username (will not stop until found)...")
    attempts = 0
    best_uname = None
    best_uname_score = -1.0

    while True:
        attempts += 1
        uname = random_cred(uname_len)

        xs_ch, ps_ch = chirikov_trajectory(uname, n_iter=n_iter_ch, scale=scale_ch)
        xs_dj, ys_dj = dejong_trajectory(uname, n_iter=n_iter_dj, scale=scale_dj)
        xs_lz, ys_lz, zs_lz = lorenz_trajectory_for_stability(
            uname, n_iter=n_iter_lz, scale=scale_lz, dt=dt_lz
        )

        # require Lorenz stability
        if not is_stable_orbit(xs_lz, ys_lz, zs_lz, radius=50.0, lyap_thresh=0.0):
            if attempts % 100 == 0:
                print(f"  tried {attempts} usernames (still looking for stable one)...")
            continue

        score = orbit_spread_score(xs_ch, ps_ch, xs_dj, ys_dj, xs_lz, ys_lz, zs_lz)
        if score > best_uname_score:
            best_uname_score = score
            best_uname = uname

        # choose a threshold so we *stop* once we see a “good enough” orbit
        if score >= 0.5:   # tweak threshold if you want stricter/looser
            print(f"\n[+] Found demo username after {attempts} tries: '{best_uname}' "
                  f"(spread score={best_uname_score:.3f})")
            break

        if attempts % 200 == 0:
            print(f"  usernames tried: {attempts}, current best='{best_uname}' "
                  f"(score={best_uname_score:.3f})")

    uname = best_uname

    # recompute orbits we’ll use for c1,c2,c3
    xs_ch, ps_ch = chirikov_trajectory(uname, n_iter=n_iter_ch, scale=scale_ch)
    xs_dj, ys_dj = dejong_trajectory(uname, n_iter=n_iter_dj, scale=scale_dj)
    xs_lz, ys_lz, zs_lz = lorenz_trajectory(uname, n_iter=n_iter_lz, scale=scale_lz, dt=dt_lz)

    x_ch_f, p_ch_f = xs_ch[-1], ps_ch[-1]
    c1 = complex(np.tanh(x_ch_f) * 0.8, np.tanh(p_ch_f) * 0.8)

    x_dj_f, y_dj_f = xs_dj[-1], ys_dj[-1]
    c2 = complex(np.tanh(x_dj_f) * 0.8, np.tanh(y_dj_f) * 0.8)

    x_lz_f, y_lz_f, z_lz_f = xs_lz[-1], ys_lz[-1], zs_lz[-1]
    c3 = complex(np.tanh(x_lz_f) * 0.6, np.tanh(y_lz_f) * 0.6)

    # -------- 2) LOOP until we find a password with long Julias --------
    print("\n[+] Searching passwords for long Julia paths across ALL 3 Julias "
          "(will not stop until threshold hit)...")

    pwd_attempts = 0
    best_pwd = None
    best_min_jlen = -1

    batch_size = pwd_tries if pwd_tries > 0 else 500

    while True:
        pwd_attempts += 1
        pwd = random_cred(pwd_len)

        j1 = len(julia_trajectory("J1:" + pwd, c=c1, n_iter=1500))
        j2 = len(julia_trajectory("J2:" + pwd, c=c2, n_iter=1500))
        j3 = len(julia_trajectory("J3:" + pwd, c=c3, n_iter=1500))
        m = min(j1, j2, j3)

        if m > best_min_jlen:
            best_min_jlen = m
            best_pwd = pwd

        # progress bar within each batch
        progress_bar((pwd_attempts - 1) % batch_size + 1, batch_size, prefix="Passwords")

        if m >= julia_len_thresh:
            print(f"\n[+] Found demo pair after {pwd_attempts} passwords:")
            print(f"    username='{uname}'")
            print(f"    password='{pwd}'")
            print(f"    Julia lengths = (J1={j1}, J2={j2}, J3={j3})")
            return uname, pwd, (j1, j2, j3)

        if pwd_attempts % batch_size == 0:
            print(f"\n[find_pwd] tried {pwd_attempts} passwords so far; "
                  f"best min Julia len={best_min_jlen} (pwd='{best_pwd}')")

# ---------- Main ----------

if __name__ == "__main__":
    print("Using Hyper OrBIT: Chirikov → Julia → DeJong → Julia → Lorenz → Julia")
    username = input("Username: ")
    password = input("Password: ")

    key = derive_fractal_key_hyper(username, password)

    xs_lz, ys_lz, zs_lz = lorenz_trajectory_for_stability(
        username, n_iter=10000, scale=2.0, dt=0.01
    )
    stable_lz = is_stable_orbit(xs_lz, ys_lz, zs_lz, radius=50.0, lyap_thresh=0.0)
    orbit_id = hashlib.sha256(f"ID:{username}".encode("utf-8")).hexdigest()
    status = "STABLE" if stable_lz else "UNSTABLE"

    print(f"\n=== Hyper OrBIT trajectory report for username='{username}' ===")
    print(f"Global orbit ID: {orbit_id}")
    print(f"Lorenz orbit: {status}")
    print(f"  Lorenz final point: x={xs_lz[-1]:.4f}, y={ys_lz[-1]:.4f}, z={zs_lz[-1]:.4f}")
    print(f"  Julia 3 parameter c3: {c}")
    print(f"  Julia 3 length: {len(zs_p)}")

    print("\nFinal Julia 3 trajectory (from Lorenz c3):")
    print(str(zs_p)[1:(len(str(zs_p)) - 1)])

    print("\nGenerated key (hex):")
    print(key.hex())

    visualize_all(username, password)
    visualize_combined_hyper(username, password)

    monte_carlo_stability(
        num_samples=2000,
        uname_len=16,
        scale=2.0,
        n_iter=5000,
        radius=50.0,
        lyap_thresh=0.0,
        dt=0.01,
    )

    # Optional: search for a nice demo pair across all 6 trajectories
    # demo_uname, demo_pwd, demo_lengths = find_good_demo_path_hyper()
