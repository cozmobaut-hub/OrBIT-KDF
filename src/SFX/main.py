import hashlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import string
import random

# ---------- Global parameter ----------

K = 1.0  # kept for compatibility; used as a scale for c, not as a map parameter


# ---------- Hash helpers ----------

def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def seed_to_complex_pair(seed: bytes, scale: float) -> tuple[complex, complex]:
    """Map 32 bytes -> two complex numbers in [-scale, scale]^2."""
    a = int.from_bytes(seed[0:8], "big")
    b = int.from_bytes(seed[8:16], "big")
    c = int.from_bytes(seed[16:24], "big")
    d = int.from_bytes(seed[24:32], "big")

    x1 = (a / 2**64) * (2 * scale) - scale
    y1 = (b / 2**64) * (2 * scale) - scale
    x2 = (c / 2**64) * (2 * scale) - scale
    y2 = (d / 2**64) * (2 * scale) - scale

    return complex(x1, y1), complex(x2, y2)


# ---------- SFX fractal ----------

def clamp_complex(z: complex, limit: float = 1e6) -> complex:
    """Clamp complex value to a disk of given radius to avoid overflow."""
    r = abs(z)
    if r > limit:
        return z * (limit / r)
    return z


def sfx_step(z: complex, c: complex) -> complex:
    """
    SFX fractal step:
        z_{n+1} = z_n * |z_n|^2 - z_n * (c^2)
    Matches GLSL: z*dot(z,z) - z*(c*c)
    """
    z = z * (abs(z) ** 2) - z * (c * c)
    return clamp_complex(z, limit=1e6)


def sfx_trajectory(username: str,
                   n_iter: int,
                   scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Username -> seed -> (z0, c), then iterate SFX.
    Returns arrays of x=Re(z), y=Im(z).
    """
    seed = sha256_bytes(username)
    z0, c = seed_to_complex_pair(seed, scale)
    zs = []
    z = z0
    for _ in range(n_iter):
        z = sfx_step(z, c)
        zs.append(z)
    zs = np.array(zs, dtype=complex)
    return zs.real, zs.imag


def sfx_background(n_seeds: int,
                   n_iter: int,
                   scale: float,
                   rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random SFX orbits for a background cloud."""
    xs_all, ys_all = [], []
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_seeds):
        xr = rng.uniform(-scale, scale)
        yr = rng.uniform(-scale, scale)
        cr = rng.uniform(-scale, scale)
        ci = rng.uniform(-scale, scale)
        z = complex(xr, yr)
        c = complex(cr, ci)
        for _ in range(n_iter):
            z = sfx_step(z, c)
            xs_all.append(z.real)
            ys_all.append(z.imag)
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
                     n_iter: int = 1500) -> np.ndarray:
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


# ---------- Master visualization (SFX + Julia) ----------

def visualize_both_sfx(username: str, password: str):
    # SFX orbits
    xs_small, ys_small = sfx_trajectory(
        username, n_iter=2000, scale=1.5
    )
    xs_big, ys_big = sfx_trajectory(
        username, n_iter=2000, scale=4.0
    )

    # Background cloud
    xs_bg, ys_bg = sfx_background(
        n_seeds=500,
        n_iter=500,
        scale=3.0,
        rng_seed=1
    )

    # Julia parameter from final SFX point
    z_final = complex(xs_small[-1], ys_small[-1])
    cx = np.tanh(z_final.real) * 0.8
    cy = np.tanh(z_final.imag) * 0.8
    c = complex(cx, cy)

    # High-precision Julia set + password orbit
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    Xj, Yj, Jimg = julia_set_grid(
        c, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        N=900, max_iter=300
    )
    zs_p = julia_trajectory(password, c=c, n_iter=1500)

    fig = plt.figure(figsize=(22, 5))

    # Panel 1: pure SFX orbit
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(xs_small, ys_small, color="lime", linewidth=0.6)
    ax1.scatter(xs_small, ys_small, s=2.5, color="lime", alpha=0.9)
    ax1.set_title(f"SFX orbit (small box)\nusername='{username}'")
    ax1.set_xlabel("Re(z)")
    ax1.set_ylabel("Im(z)")

    # Panel 2: blown-out SFX with orbit dots
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hexbin(xs_bg, ys_bg, gridsize=220,
               cmap="cool", mincnt=1)
    ax2.scatter(xs_big, ys_big, s=3.0, color="black", alpha=0.95)
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_title("SFX blown-out (zoom)\nusername orbit as dots")
    ax2.set_xlabel("Re(z)")
    ax2.set_ylabel("Im(z)")

    # Panel 3: Julia set + password orbit
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

    # Panel 4: 3D curve (Re(z_sfx), Im(z_sfx), |z_julia|)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    L = min(len(xs_small), len(zs_p))
    xs3 = xs_small[:L]
    ys3 = ys_small[:L]
    rs3 = np.abs(zs_p[:L])
    ax4.plot(xs3, ys3, rs3, lw=0.8)
    ax4.set_title("3D curve (Re(z_sfx), Im(z_sfx), |z_julia|)")
    ax4.set_xlabel("Re(z_sfx)")
    ax4.set_ylabel("Im(z_sfx)")
    ax4.set_zlabel("|z_julia|")

    plt.tight_layout()
    plt.show()


# ---------- Stability Monte Carlo (SFX) ----------

def sfx_trajectory_for_stability(username: str,
                                 n_iter: int,
                                 scale: float) -> tuple[np.ndarray, np.ndarray]:
    return sfx_trajectory(username, n_iter=n_iter, scale=scale)


def is_stable_orbit(xs, ys, radius=20.0, lyap_thresh=0.0) -> bool:
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
                          pwd_len=8,
                          scale=2.0,
                          n_iter=1500,
                          radius=20.0,
                          lyap_thresh=0.0):
    chars = string.ascii_letters + string.digits
    stable = 0
    unstable = 0

    for _ in range(num_samples):
        uname = ''.join(random.choice(chars) for _ in range(uname_len))
        _pwd = ''.join(random.choice(chars) for _ in range(pwd_len))  # unused
        xs, ys = sfx_trajectory_for_stability(
            uname, n_iter=n_iter, scale=scale
        )
        if is_stable_orbit(xs, ys, radius=radius, lyap_thresh=lyap_thresh):
            stable += 1
        else:
            unstable += 1

    total = stable + unstable
    pct_stable = 100.0 * stable / total
    pct_unstable = 100.0 * unstable / total
    print(f"\n=== Monte Carlo stability over {total} random username SFX orbits ===")
    print(f"Stable:   {stable} ({pct_stable:.2f}%)")
    print(f"Unstable: {unstable} ({pct_unstable:.2f}%)\n")


# ---------- KDF (SFX + Julia) ----------

def derive_fractal_key(username: str, password: str) -> bytes:
    """
    Full KDF: username/password -> SHA-256 -> SFX + Julia -> SHA-512 key.
    """
    xs_small, ys_small = sfx_trajectory(
        username, n_iter=130000, scale=2.0
    )
    z_final = complex(xs_small[-1], ys_small[-1])

    cx = np.tanh(z_final.real) * 0.8
    cy = np.tanh(z_final.imag) * 0.8
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


def find_good_demo_path(
    uname_tries=50000,
    pwd_tries=50000,
    uname_len=8,
    pwd_len=8,
    n_iter_sfx=2000,
    julia_len_thresh=1400,
    scale=2.0,
):
    chars = string.ascii_letters + string.digits

    for _ in range(uname_tries):
        uname = ''.join(random.choice(chars) for _ in range(uname_len))
        xs, ys = sfx_trajectory_for_stability(
            uname, n_iter=n_iter_sfx, scale=scale
        )
        if is_stable_orbit(xs, ys, radius=20.0, lyap_thresh=0.0):
            print("Found STABLE username (SFX):", uname)

            z_final = complex(xs[-1], ys[-1])
            cx = np.tanh(z_final.real) * 0.8
            cy = np.tanh(z_final.imag) * 0.8
            c = complex(cx, cy)

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


# ---------- Main ----------

if __name__ == "__main__":
    print("Using SFX fractal (z -> z|z|^2 - z c^2)")
    username = input("Username: ")
    password = input("Password: ")
    key = derive_fractal_key(username, password)

    xs, ys = sfx_trajectory_for_stability(
        username, n_iter=2000, scale=2.0
    )

    stable = is_stable_orbit(xs, ys, radius=20.0, lyap_thresh=0.0)
    orbit_id = hashlib.sha256(f"ID:{username}".encode("utf-8")).hexdigest()
    status = "STABLE" if stable else "UNSTABLE"

    print(f"Orbit ({orbit_id}): {status} for credentials (username='{username}')")
    print(f"Trajectory intersection in complex space: {c} ")
    print(f"SFX space trajectory final point: x={xs[-1]:.4f}, y={ys[-1]:.4f}")
    print(f"Final Julia Trajectory:\n {str(zs_p)[1:(len(str(zs_p))-1)]}")
    print("Generated key (hex):", key.hex())

    visualize_both_sfx(username, password)

    monte_carlo_stability(
        num_samples=90000,
        uname_len=24,
        pwd_len=24,
        scale=2.0,
        n_iter=2000,
        radius=20.0,
        lyap_thresh=0.0,
    )
