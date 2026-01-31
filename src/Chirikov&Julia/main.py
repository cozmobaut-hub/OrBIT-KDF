import hashlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import string
import random

# ---------- Global parameter ----------

K = 1.0  # <<< set your K here once

# ---------- Hash helpers ----------

def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()

def seed_to_float_pair(seed: bytes, scale: float) -> tuple[float, float]:
    a = int.from_bytes(seed[0:8], "big")
    b = int.from_bytes(seed[8:16], "big")
    x = (a / 2**64) * (2 * scale) - scale
    p = (b / 2**64) * (2 * scale) - scale
    return x, p

# ---------- Chirikov ----------

def chirikov_step(x, p, k):
    p = p + k * np.sin(x)
    x = x + p
    return x, p

def chirikov_trajectory(username: str, k: float,
                        n_iter: int, scale: float):
    x, p = seed_to_float_pair(sha256_bytes(username), scale)
    xs, ps = [], []
    for _ in range(n_iter):
        x, p = chirikov_step(x, p, k)
        xs.append(x)
        ps.append(p)
    return np.array(xs), np.array(ps)

def chirikov_background(k: float,
                        n_seeds: int,
                        n_iter: int,
                        scale: float,
                        rng_seed: int = 0):
    xs_all, ps_all = [], []
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_seeds):
        x = rng.uniform(-scale, scale)
        p = rng.uniform(-scale, scale)
        for _ in range(n_iter):
            x, p = chirikov_step(x, p, k)
            xs_all.append(x)
            ps_all.append(p)
    return np.array(xs_all), np.array(ps_all)

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

def julia_trajectory(password: str, c: complex,
                     n_iter: int = 1500):
    seed = sha256_bytes(password)
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

# ---------- Master visualization ----------

def visualize_both_chirikov(username: str, password: str, k: float = K):
    # Chirikov orbits
    xs_small, ps_small = chirikov_trajectory(
        username, k=k, n_iter=2000, scale=2.0  # small box -> stable loop view
    )
    xs_big, ps_big = chirikov_trajectory(
        username, k=k, n_iter=2000, scale=6.0  # big box for blown-out dots
    )

    # Zoomed blown-out background
    xs_bg_big, ps_bg_big = chirikov_background(
        k=k,
        n_seeds=500,
        n_iter=500,
        scale=4.0,          # was 6.0, now zoomed a bit
        rng_seed=1
    )

    # Julia parameter from small-box final point
    x_final, p_final = xs_small[-1], ps_small[-1]
    cx = np.tanh(x_final) * 0.8
    cy = np.tanh(p_final) * 0.8
    c = complex(cx, cy)

    # High-precision Julia set + orbit
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    Xj, Yj, Jimg = julia_set_grid(
        c, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        N=900, max_iter=300
    )
    zs_p = julia_trajectory(password, c=c, n_iter=1500)

    fig = plt.figure(figsize=(22, 5))

    # Panel 1: pure Chirikov orbit (shape)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(xs_small, ps_small, color="lime", linewidth=0.6)
    ax1.scatter(xs_small, ps_small, s=2.5, color="lime", alpha=0.9)
    ax1.set_title(f"Chirikov orbit (small box)\nusername='{username}'")
    ax1.set_xlabel("x")
    ax1.set_ylabel("p")

    # Panel 2: zoomed blown-out Chirikov with orbit dots
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hexbin(xs_bg_big, ps_bg_big, gridsize=220,
               cmap="cool", mincnt=1)
    ax2.scatter(xs_big, ps_big, s=3.0, color="black", alpha=0.95)

    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_title("Chirikov blown-out (zoom)\nusername orbit as dots")
    ax2.set_xlabel("x")
    ax2.set_ylabel("p")

    # Panel 3: high-res Julia set + big orbit markers
    ax3 = fig.add_subplot(1, 4, 3)
    extent = [x_min, x_max, y_min, y_max]
    ax3.imshow(Jimg, extent=extent, origin="lower", cmap="magma")
    if len(zs_p) > 0:
        mask = ((zs_p.real >= x_min) & (zs_p.real <= x_max) &
                (zs_p.imag >= y_min) & (zs_p.imag <= y_max))
        ax3.plot(zs_p.real[mask], zs_p.imag[mask],
                 color="cyan", linewidth=1.2, alpha=0.9)
        ax3.scatter(zs_p.real[mask], zs_p.imag[mask],
                    s=14.0, color="white", alpha=0.95,
                    edgecolors="black", linewidths=0.4)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_title(
        f"High-res Julia set for c={c.real:.3f}+{c.imag:.3f}i\npassword orbit overlaid"
    )
    ax3.set_xlabel("Re(z)")
    ax3.set_ylabel("Im(z)")

    # Panel 4: 3D curve
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    L = min(len(xs_small), len(zs_p))
    xs3 = xs_small[:L]
    ps3 = ps_small[:L]
    rs3 = np.abs(zs_p[:L])
    ax4.plot(xs3, ps3, rs3, lw=0.8)
    ax4.set_title("3D curve (x, p, |z|)")
    ax4.set_xlabel("x (Chirikov)")
    ax4.set_ylabel("p (Chirikov)")
    ax4.set_zlabel("|z| (Julia)")

    plt.tight_layout()
    plt.show()

# ---------- Stability Monte Carlo ----------

def chirikov_trajectory_for_stability(username: str,
                                      k: float,
                                      n_iter: int,
                                      scale: float):
    x, p = seed_to_float_pair(sha256_bytes(username), scale)
    xs, ps = [], []
    for _ in range(n_iter):
        x, p = chirikov_step(x, p, k)
        xs.append(x)
        ps.append(p)
    return np.array(xs), np.array(ps)

def is_stable_orbit(xs, ps, radius=2000000.0, lyap_thresh=0.0):
    """Heuristic: stable if bounded and average local expansion <= threshold."""
    # 1) Boundedness
    r = np.sqrt(xs**2 + ps**2)
    if r.max() > radius:
        return False

    # 2) Crude Lyapunov estimate based on step growth
    dx = np.diff(xs)
    dp = np.diff(ps)
    step = np.sqrt(dx**2 + dp**2)

    # Avoid zeros
    step[step == 0] = 1e-16
    ratios = step[1:] / step[:-1]
    lyap_est = np.mean(np.log(np.abs(ratios)))

    # "Stable" if no net exponential growth
    return lyap_est < lyap_thresh

def monte_carlo_stability(num_samples=1000,
                          uname_len=8,
                          pwd_len=8,
                          k: float = K,
                          scale=2.0,
                          n_iter=1500,
                          radius=20.0,
                          lyap_thresh=0.0):
    chars = string.ascii_letters + string.digits
    stable = 0
    unstable = 0

    for _ in range(num_samples):
        uname = ''.join(random.choice(chars) for _ in range(uname_len))
        pwd = ''.join(random.choice(chars) for _ in range(pwd_len))  # unused
        xs, ps = chirikov_trajectory_for_stability(
            uname, k=k, n_iter=n_iter, scale=scale
        )
        if is_stable_orbit(xs, ps, radius=radius, lyap_thresh=lyap_thresh):
            stable += 1
        else:
            unstable += 1

    total = stable + unstable
    pct_stable = 100.0 * stable / total
    pct_unstable = 100.0 * unstable / total
    print(f"\n=== Monte Carlo stability over {total} random credential orbits ===")
    print(f"Stable:   {stable} ({pct_stable:.2f}%)")
    print(f"Unstable: {unstable} ({pct_unstable:.2f}%)\n")

# ---------- KDF ----------

def derive_fractal_key(username: str, password: str, k: float = K) -> bytes:
    """Full KDF: username/password -> SHA-256 -> Chirikov+Julia -> SHA-512 key."""
    xs_small, ps_small = chirikov_trajectory(
        username, k=k, n_iter=2000, scale=2.0
    )
    x_final, p_final = xs_small[-1], ps_small[-1]

    cx = np.tanh(x_final) * 0.8
    cy = np.tanh(p_final) * 0.8
    global c
    c = complex(cx, cy)

    global zs_p
    zs_p = julia_trajectory(password, c=c, n_iter=1500)

    L = min(512, len(xs_small), len(zs_p))
    xs3 = xs_small[:L]
    ps3 = ps_small[:L]
    rs3 = np.abs(zs_p[:L])

    buf = bytearray()
    for x, p, r in zip(xs3, ps3, rs3):
        buf += np.float64(x).tobytes()
        buf += np.float64(p).tobytes()
        buf += np.float64(r).tobytes()

    return hashlib.sha512(bytes(buf)).digest()

# ---------- Main ----------

if __name__ == "__main__":
    print(f"Using K = {K}")
    username = input("Username: ")
    password = input("Password: ")
    key = derive_fractal_key(username, password, k=K)

    xs, ps = chirikov_trajectory_for_stability(
        username, k=K, n_iter=2000, scale=2.0
    )

    stable = is_stable_orbit(xs, ps, radius=20.0, lyap_thresh=0.0)

    orbit_id = hashlib.sha256(f"ID:{username}".encode("utf-8")).hexdigest()

    status = "STABLE" if stable else "UNSTABLE"
    print(f"Orbit ({orbit_id}): {status} for credentials (username='{username}')")
    print(f"Trajectory intersection in complex space: {c} ")
    print(f"Chirikov/Julia space trajectory intersection: x={xs[-1]:.4f}, p={ps[-1]:.4f}")
    print(f"Final Julia Trajectory: \n {str(zs_p)[1:(len(str(zs_p))-1)]}")
    print("Generated key (hex):", key.hex())

    visualize_both_chirikov(username, password, k=K)

    monte_carlo_stability(
        num_samples=90000,
        uname_len=24,
        pwd_len=24,
        k=K,
        scale=2.0,
        n_iter=2000,
        radius=2000000.0,
        lyap_thresh=0.0,
    )
