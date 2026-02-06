#!/usr/bin/env python3
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import string
import random

# ---------- De Jong parameters ----------
# Classic chaotic set; tweak these if you want different shapes.
a_dj, b_dj, c_dj, d_dj = .026, -2.54, .146, -0.407


# ---------- Hash helpers ----------

def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def seed_to_float_pair(seed: bytes, scale: float) -> tuple[float, float]:
    a = int.from_bytes(seed[0:8], "big")
    b = int.from_bytes(seed[8:16], "big")
    x = (a / 2**64) * (2 * scale) - scale
    y = (b / 2**64) * (2 * scale) - scale
    return x, y


# ---------- De Jong map ----------

def dejong_step(x, y, a=a_dj, b=b_dj, c=c_dj, d=d_dj):
    x_next = np.sin(a * y) - np.cos(b * x)
    y_next = np.sin(c * x) - np.cos(d * y)
    return x_next, y_next


def dejong_trajectory(username: str, n_iter: int, scale: float):
    x, y = seed_to_float_pair(sha256_bytes(username), scale)
    xs, ys = [], []
    for _ in range(n_iter):
        x, y = dejong_step(x, y)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def dejong_background(n_seeds: int,
                      n_iter: int,
                      scale: float,
                      rng_seed: int = 0):
    xs_all, ys_all = [], []
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_seeds):
        x = rng.uniform(-scale, scale)
        y = rng.uniform(-scale, scale)
        for _ in range(n_iter):
            x, y = dejong_step(x, y)
            xs_all.append(x)
            ys_all.append(y)
    return np.array(xs_all), np.array(ys_all)


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


# ---------- Master visualization (De Jong + Julia) ----------

def visualize_both_dejong(username: str, password: str):
    # De Jong orbits
    xs_small, ys_small = dejong_trajectory(
        username, n_iter=2000, scale=2.0
    )
    xs_big, ys_big = dejong_trajectory(
        username, n_iter=2000, scale=6.0
    )

    # Background cloud
    xs_bg_big, ys_bg_big = dejong_background(
        n_seeds=500,
        n_iter=500,
        scale=4.0,
        rng_seed=1
    )

    # Julia parameter from small-box final point
    x_final, y_final = xs_small[-1], ys_small[-1]
    cx = np.tanh(x_final) * 0.8
    cy = np.tanh(y_final) * 0.8
    global c
    c = complex(cx, cy)

    # Julia set + orbit
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    Xj, Yj, Jimg = julia_set_grid(
        c, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        N=900, max_iter=300
    )
    global zs_p
    zs_p = julia_trajectory(password, c=c, n_iter=1500)

    fig = plt.figure(figsize=(22, 5))

    # Panel 1: pure De Jong orbit (shape)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(xs_small, ys_small, color="lime", linewidth=0.6)
    ax1.scatter(xs_small, ys_small, s=2.5, color="lime", alpha=0.9)
    ax1.set_title(f"De Jong orbit (small box)\nusername='{username}'")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Panel 2: zoomed blown-out De Jong with orbit dots
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hexbin(xs_bg_big, ys_bg_big, gridsize=220,
               cmap="cool", mincnt=1)
    ax2.scatter(xs_big, ys_big, s=3.0, color="black", alpha=0.95)

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_title("De Jong blown-out (zoom)\nusername orbit as dots")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

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
    ys3 = ys_small[:L]
    rs3 = np.abs(zs_p[:L])
    ax4.plot(xs3, ys3, rs3, lw=0.8)
    ax4.set_title("3D curve (x, y, |z|)")
    ax4.set_xlabel("x (De Jong)")
    ax4.set_ylabel("y (De Jong)")
    ax4.set_zlabel("|z| (Julia)")

    plt.tight_layout()
    plt.show()


# ---------- Stability Monte Carlo ----------

def dejong_trajectory_for_stability(username: str,
                                    n_iter: int,
                                    scale: float):
    return dejong_trajectory(username, n_iter=n_iter, scale=scale)


def is_stable_orbit(xs, ys, radius=2000000.0, lyap_thresh=0.0):
    """Heuristic: stable if bounded and average local expansion <= threshold."""
    r = np.sqrt(xs**2 + ys**2)
    if r.max() > radius:
        return False

    dx = np.diff(xs)
    dy = np.diff(ys)
    step = np.sqrt(dx**2 + dy**2)

    step[step == 0] = 1e-16
    ratios = step[1:] / step[:-1]
    lyap_est = np.mean(np.log(np.abs(ratios)))

    return lyap_est < lyap_thresh


def monte_carlo_stability(num_samples=1000,
                          uname_len=8,
                          scale=2.0,
                          n_iter=1500,
                          radius=20.0,
                          lyap_thresh=0.0):
    chars = string.ascii_letters + string.digits
    stable = 0
    unstable = 0

    for _ in range(num_samples):
        uname = ''.join(random.choice(chars) for _ in range(uname_len))
        xs, ys = dejong_trajectory_for_stability(
            uname, n_iter=n_iter, scale=scale
        )
        if is_stable_orbit(xs, ys, radius=radius, lyap_thresh=lyap_thresh):
            stable += 1
        else:
            unstable += 1

    total = stable + unstable
    pct_stable = 100.0 * stable / total
    pct_unstable = 100.0 * unstable / total
    print(f"\n=== De Jong Monte Carlo stability over {total} random username orbits ===")
    print(f"Stable:   {stable} ({pct_stable:.2f}%)")
    print(f"Unstable: {unstable} ({pct_unstable:.2f}%)\n")

def find_good_demo_path_dejong(
    uname_tries=50000,
    pwd_tries=50000,
    uname_len=8,
    pwd_len=8,
    n_iter_dejong=2000,
    julia_len_thresh=1400,
    scale=2.0,
):
    import string, random
    chars = string.ascii_letters + string.digits

    # 1) find a stable username (De Jong orbit)
    for _ in range(uname_tries):
        uname = ''.join(random.choice(chars) for _ in range(uname_len))
        xs, ys = dejong_trajectory_for_stability(
            uname, n_iter=n_iter_dejong, scale=scale
        )
        if is_stable_orbit(xs, ys, radius=20.0, lyap_thresh=0.0):
            print("Found STABLE username:", uname)

            # compute c for this username (same as in derive_fractal_key_dejong)
            x_final, y_final = xs[-1], ys[-1]
            cx = np.tanh(x_final) * 0.8
            cy = np.tanh(y_final) * 0.8
            c = complex(cx, cy)

            # 2) search passwords with long Julia paths
            best_pwd = None
            best_len = -1
            for _ in range(pwd_tries):
                pwd = ''.join(random.choice(chars) for _ in range(pwd_len))
                zs = julia_trajectory(pwd, c=c, n_iter=1500)
                L = len(zs)
                if L > best_len:
                    best_len = L
                    best_pwd = pwd
                if L >= julia_len_thresh:
                    print(
                        f"Found good DEMO pair: "
                        f"username='{uname}', password='{pwd}', "
                        f"Julia length={L}"
                    )
                    return uname, pwd, L

            print(
                f"No password reached threshold; best for username '{uname}' "
                f"is password='{best_pwd}' with length={best_len}"
            )
            return uname, best_pwd, best_len

    print("No stable username found in", uname_tries, "tries")
    return None, None, 0


# ---------- KDF (De Jong + Julia) ----------

def derive_fractal_key_dejong(username: str, password: str) -> bytes:
    """Full KDF: username/password -> SHA-256 -> De Jong + Julia -> SHA-512 key."""
    xs_small, ys_small = dejong_trajectory(
        username, n_iter=2000, scale=2.0
    )
    x_final, y_final = xs_small[-1], ys_small[-1]

    cx = np.tanh(x_final) * 0.8
    cy = np.tanh(y_final) * 0.8
    global c
    c = complex(cx, cy)

    global zs_p
    zs_p = julia_trajectory(password, c=c, n_iter=1500)

    L = min(512, len(xs_small), len(zs_p))
    xs3 = xs_small[:L]
    ys3 = ys_small[:L]
    rs3 = np.abs(zs_p[:L])

    buf = bytearray()
    for x, y, r in zip(xs3, ys3, rs3):
        buf += np.float64(x).tobytes()
        buf += np.float64(y).tobytes()
        buf += np.float64(r).tobytes()

    return hashlib.sha512(bytes(buf)).digest()


# ---------- Main ----------

if __name__ == "__main__":
    print(f"Using De Jong map with a={a_dj}, b={b_dj}, c={c_dj}, d={d_dj}")
    username = input("Username: ")
    password = input("Password: ")
    key = derive_fractal_key_dejong(username, password)

    xs, ys = dejong_trajectory_for_stability(
        username, n_iter=2000, scale=2.0
    )

    stable = is_stable_orbit(xs, ys, radius=20.0, lyap_thresh=0.0)

    orbit_id = hashlib.sha256(f"ID:{username}".encode("utf-8")).hexdigest()

    status = "STABLE" if stable else "UNSTABLE"
    print(f"Orbit ({orbit_id}): {status} for credentials (username='{username}')")
    print(f"Trajectory intersection in complex space: {c} ")
    print(f"De Jong/Julia space intersection: x={xs[-1]:.4f}, y={ys[-1]:.4f}")
    print(f"Final Julia Trajectory: \n {str(zs_p)[1:(len(str(zs_p))-1)]}")
    print("Generated key (hex):", key.hex())

    visualize_both_dejong(username, password)

    monte_carlo_stability(
        num_samples=9000,
        uname_len=24,
        scale=2.0,
        n_iter=2000,
        radius=2000000.0,
        lyap_thresh=0.0,#
    )
