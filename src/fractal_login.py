#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox

from visualize import derive_fractal_key  # import the new SHAh KDF


def on_login():
    username = entry_user.get().strip()
    password = entry_pass.get()
    key_entered = text_key.get("1.0", tk.END).strip()

    if not username or not password or not key_entered:
        messagebox.showerror("Error", "Username, password, and key are all required.")
        return

    # derive expected key using SHAh (same as registration)
    expected_key_bytes = derive_fractal_key(username, password)
    expected_key = expected_key_bytes.hex()  # 512-bit key as hex

    if key_entered == expected_key:
        messagebox.showinfo(
            "Login",
            "ACCESS GRANTED\n"
            "SHAh(Chirikov(username) + Julia(password)) key matches."
        )
    else:
        messagebox.showerror(
            "Login",
            "ACCESS DENIED\n"
            "Key does not match this username/password pair."
        )


# GUI setup
root = tk.Tk()
root.title("FractalAuth - Login (Chirikov + Julia, SHAh KDF)")

tk.Label(root, text="Username (Chirikov):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_user = tk.Entry(root, width=30)
entry_user.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Password (Julia):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_pass = tk.Entry(root, width=30, show="*")
entry_pass.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Fractal Key (paste from file/USB):").grid(row=2, column=0, columnspan=2)

text_key = tk.Text(root, width=64, height=3, wrap="none")
text_key.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

btn_login = tk.Button(root, text="Login", command=on_login)
btn_login.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
