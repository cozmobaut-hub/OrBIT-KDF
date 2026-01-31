#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox

from main import derive_fractal_key  # import the new KDF


def on_generate():
    username = entry_user.get().strip()
    password = entry_pass.get()
    if not username or not password:
        messagebox.showerror("Error", "Username and password required.")
        return

    # SHAh KDF: username + password -> 64-byte key (512-bit)
    key_bytes = derive_fractal_key(username, password)

    # Show full 512-bit key as 128 hex chars (like before),
    # or slice to [:32] if you want 256 bits instead.
    key_hex = key_bytes.hex()

    text_key.config(state="normal")
    text_key.delete("1.0", tk.END)
    text_key.insert(tk.END, key_hex)
    text_key.config(state="disabled")

    messagebox.showinfo(
        "Key Generated",
        "Key generated with SHAh.\n"
        "Copy and save it securely (e.g., on a USB file).\n"
        "You will need it (plus username+password) to log in."
    )


# GUI setup
root = tk.Tk()
root.title("FractalAuth - Registration (Chirikov + Julia, SHAh KDF)")

tk.Label(root, text="Username (Chirikov):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_user = tk.Entry(root, width=30)
entry_user.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Password (Julia):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_pass = tk.Entry(root, width=30, show="*")
entry_pass.grid(row=1, column=1, padx=5, pady=5)

btn_gen = tk.Button(root, text="Generate Fractal Key", command=on_generate)
btn_gen.grid(row=2, column=0, columnspan=2, pady=10)

tk.Label(root, text="Your Fractal Key (save this):").grid(row=3, column=0, columnspan=2)

text_key = tk.Text(root, width=64, height=3, wrap="none")
text_key.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
text_key.config(state="disabled")

root.mainloop()
