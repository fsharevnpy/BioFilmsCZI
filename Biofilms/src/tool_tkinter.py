# simple_gui.py
# Very simple tkinter GUI demo
# Comments in English only

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

def browse_file():
    filetypes = [("Zeiss CZI files", "*.czi"), ("All files", "*.*")]
    filename = filedialog.askopenfilename(title="Choose a file", filetypes=filetypes)
    if filename:
        path_var.set(filename)

def convert_file():
    path = path_var.get().strip()
    if not path:
        messagebox.showwarning("Warning", "Please choose a file first.")
        return
    in_path = Path(path)
    if not in_path.exists():
        messagebox.showerror("Error", f"File not found: {in_path}")
        return
    # Here just fake convert, write a text file instead
    out_path = in_path.with_suffix(".txt")
    out_path.write_text("This is where conversion result would go.", encoding="utf-8")
    messagebox.showinfo("Done", f"Converted!\nOutput: {out_path}")

# Main window
root = tk.Tk()
root.title("Simple GUI Demo")
root.geometry("400x150")

path_var = tk.StringVar()

frame = tk.Frame(root)
frame.pack(pady=20)

entry = tk.Entry(frame, textvariable=path_var, width=40)
entry.pack(side="left", padx=5)

btn_browse = tk.Button(frame, text="Browse", command=browse_file)
btn_browse.pack(side="left")

btn_convert = tk.Button(root, text="Convert", command=convert_file, height=2, width=20)
btn_convert.pack(pady=10)

root.mainloop()
