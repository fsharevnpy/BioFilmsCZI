# simple_gui.py
# Very simple tkinter GUI demo
# Comments in English only

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

def browse_file():
    # Ask for a CZI file
    filetypes = [("Zeiss CZI files", "*.czi"), ("All files", "*.*")]
    filename = filedialog.askopenfilename(title="Choose a file", filetypes=filetypes)
    if filename:
        path_var.set(filename)

def _to_uint8(img):
    # Convert any numeric array to uint8 [0..255] for preview saving
    import numpy as np
    if img.dtype == np.uint8:
        return img
    img = img.astype(float)
    finite_mask = np.isfinite(img)
    if not finite_mask.any():
        return np.zeros_like(img, dtype=np.uint8)
    vmin = img[finite_mask].min()
    vmax = img[finite_mask].max()
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - vmin) / (vmax - vmin) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def _ensure_hwc(img):
    # Make image either (H, W) grayscale or (H, W, C) for saving
    import numpy as np
    img = np.squeeze(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        # If likely CHW, move channels to last
        cands = img.shape
        # Heuristic: treat the smallest axis as channels if <= 4
        smallest = min(range(3), key=lambda k: cands[k])
        if cands[smallest] <= 4 and smallest != 2:
            img = np.moveaxis(img, smallest, -1)
        return img
    # If still >3D, keep slicing the leading dims until 2D/3D
    while img.ndim > 3:
        img = img[0]
    return _ensure_hwc(img)

def _safe_get_dims_shape(czi):
    """
    Return (dims:str|None, shape:tuple|None, dtype:object|None) robustly
    across aicspylibczi versions.
    """
    dims = None
    shape = None
    dtype = None
    try:
        res = czi.get_dims_shape()
        # res can be 1, 2, or 3 elements depending on version
        if isinstance(res, tuple):
            if len(res) == 3:
                dims, shape, dtype = res
            elif len(res) == 2:
                dims, shape = res
            elif len(res) == 1:
                # Some builds return a single tuple (dims, shape)
                x = res[0]
                if isinstance(x, tuple) and len(x) >= 2:
                    dims, shape = x[0], x[1]
    except Exception:
        pass

    # Try additional attributes if available
    if dims is None:
        # Some versions expose .dims
        if hasattr(czi, "dims"):
            try:
                dims = czi.dims
            except Exception:
                pass
    if shape is None:
        # Try .size or .size() returning dict mapping dims->len
        if hasattr(czi, "size"):
            try:
                sz = czi.size
                if isinstance(sz, dict) and dims:
                    shape = tuple(sz[d] for d in dims)
            except Exception:
                pass
    return dims, shape, dtype

def _unwrap_maybe_tuple(ret):
    """
    Some versions return just ndarray, others (ndarray, meta) or (ndarray, dims).
    This makes it consistent: always return the ndarray.
    """
    try:
        # If it's already an array-like, just return it
        import numpy as np
        if isinstance(ret, np.ndarray):
            return ret
        # If it's a (arr, something)
        if isinstance(ret, (list, tuple)) and len(ret) >= 1:
            first = ret[0]
            if isinstance(first, np.ndarray):
                return first
        return ret
    except Exception:
        return ret

def _read_first_plane_with_aicspylibczi(in_path):
    """
    Open a CZI with aicspylibczi and return:
    - img: numpy array (H,W) or (H,W,C) suitable for saving
    - info: human readable string with dims/shape/dtype
    """
    from aicspylibczi import CziFile
    import numpy as np

    czi = CziFile(str(in_path))

    dims, shape, dtype = _safe_get_dims_shape(czi)
    info = f"dims={dims}, shape={shape}, dtype={dtype}"

    # Build a selection dict: first index for each non-spatial axis
    sel = {}
    for ax in ("S", "R", "I", "H", "T", "Z", "C", "B"):  # be generous with axes
        if dims and ax in dims:
            sel[ax] = 0

    # Detect mosaic if 'M' in dims across versions
    is_mosaic = (dims and "M" in dims)

    if is_mosaic and hasattr(czi, "read_mosaic"):
        # Pick first channel if exists
        c_idx = sel.get("C", 0)
        ret = czi.read_mosaic(C=c_idx, scale_factor=1.0)
        data = _unwrap_maybe_tuple(ret)
    else:
        # Normal read
        ret = czi.read_image(**sel) if sel else czi.read_image()
        data = _unwrap_maybe_tuple(ret)

    data = np.squeeze(data)
    data = _ensure_hwc(data)
    return data, info

def convert_file():
    path = path_var.get().strip()
    if not path:
        messagebox.showwarning("Warning", "Please choose a file first.")
        return
    in_path = Path(path)
    if not in_path.exists():
        messagebox.showerror("Error", f"File not found: {in_path}")
        return

    # Try to import aicspylibczi right here to show a friendly error
    try:
        import aicspylibczi  # noqa: F401
    except Exception as e:
        messagebox.showerror("aicspylibczi missing", f"Import failed: {e}")
        return

    # Attempt to read first plane and save a PNG preview
    try:
        img, info = _read_first_plane_with_aicspylibczi(in_path)
    except Exception as e:
        messagebox.showerror("CZI read error", f"Failed to read image:\n{e}")
        return

    # Convert and save preview PNG
    try:
        from PIL import Image
        img_u8 = _to_uint8(img)
        if img_u8.ndim == 2:
            pil = Image.fromarray(img_u8, mode="L")
        else:
            if img_u8.shape[2] > 3:
                img_u8 = img_u8[:, :, :3]
            pil = Image.fromarray(img_u8)
        out_path = in_path.with_suffix(".png")
        pil.save(out_path)
    except Exception as e:
        messagebox.showerror("Save error", f"Failed to save PNG:\n{e}")
        return

    messagebox.showinfo("Done", f"Loaded CZI successfully.\n{info}\n\nPreview saved:\n{out_path}")

# Main window
root = tk.Tk()
root.title("Simple GUI Demo")
root.geometry("420x170")

path_var = tk.StringVar()

frame = tk.Frame(root)
frame.pack(pady=20)

entry = tk.Entry(frame, textvariable=path_var, width=44)
entry.pack(side="left", padx=5)

btn_browse = tk.Button(frame, text="Browse", command=browse_file)
btn_browse.pack(side="left")

btn_convert = tk.Button(root, text="Convert (load & save preview)", command=convert_file, height=2, width=28)
btn_convert.pack(pady=10)

root.mainloop()
