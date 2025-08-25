import sys
from pathlib import Path

# --- add ../inc to sys.path so we can import local modules like analyze ---
# Resolve this file's directory; fallback to CWD if __file__ is not available
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

INC_DIR = ("Biofilms/src").resolve()
if str(INC_DIR) not in sys.path:
    sys.path.insert(0, str(INC_DIR))

# Now we can import from analyze.py inside ../inc
from analyze import process_channel, compose_rgb, intensity_rgb

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from aicspylibczi import CziFile
import cv2  # Needed for BGR->RGB conversion for overlay

# ---- Globals ----
file_list = []
czi_obj = None
image_array = None
axes_labels = ""
display_imgtk = None

# Base image (current plane) + view state
base_pil = None
zoom = 1.0
min_zoom = 0.05
max_zoom = 10.0
offset_x = 0
offset_y = 0
drag_start = None

# App state
last_wm_state = None
current_view = "image"  # "image" or "info"
meta_dict = {}

# Simple info vars (basic summary shown in Info view too)
file_var  = None
shape_var = None
axes_var  = None
cnum_var  = None
znum_var  = None

# Caches
overlay_cache = {}  # {(c_idx, z_idx): PIL.Image} for overlay view
merge_cache   = {}  # {z_idx: PIL.Image} for merged RGB view (dead/live)

# --- Added: stats caches & current indices for legend ---
merge_stats_cache = {}    # {z_idx: {...}}
overlay_stats_cache = {}  # {(c_idx, z_idx): {...} or None}
_current_c_idx = 0
_current_z_idx = 0

# ---- Helpers ----
def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    # Percentile-free normalization: scale [min, max] to [0, 255]
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32, copy=False)
    vmin, vmax = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - vmin) / (vmax - vmin) * 255.0
    return a.astype(np.uint8)

def infer_axes(shape, info):
    # Try to read dimension order from info; fallback to heuristic
    if isinstance(info, dict):
        ax = info.get("axes") or info.get("dimension_order") or info.get("dims")
        if isinstance(ax, str) and len(ax) == len(shape):
            return ax
    n = len(shape)
    if n == 2: return "YX"
    if n == 3: return "CYX" if any(s in (3, 4) for s in shape) else "ZYX"
    if n == 4: return "CZYX"
    if n == 5: return "TCZYX"
    if n == 6: return "STCZYX"
    if n == 7: return "STRCZYX"
    return "YX".rjust(n, "S")

def get_count(axis, shape, axes):
    return int(shape[axes.index(axis)]) if axis in axes else 0

def extract_plane(arr, axes, c_idx=0, z_idx=0):
    # Extract a single 2D plane with optional channel and z selection
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Missing Y/X in axes: {axes}")
    idx = [slice(None)] * len(axes)
    for ax in ("S", "T", "R", "M"):
        if ax in axes:
            idx[axes.index(ax)] = 0
    if "C" in axes:
        nC = int(arr.shape[axes.index("C")])
        if nC > 0:
            c_idx = max(0, min(int(c_idx), nC - 1))
            idx[axes.index("C")] = c_idx
    if "Z" in axes:
        nZ = int(arr.shape[axes.index("Z")])
        if nZ > 0:
            z_idx = max(0, min(int(z_idx), nZ - 1))
            idx[axes.index("Z")] = z_idx
    plane = np.squeeze(arr[tuple(idx)])
    if plane.ndim == 2:
        return normalize_to_uint8(plane)
    if plane.ndim == 3 and plane.shape[-1] in (3, 4):
        if plane.dtype != np.uint8:
            plane = np.stack([normalize_to_uint8(plane[..., k]) for k in range(plane.shape[-1])], axis=-1)
        return plane[..., :3]
    return normalize_to_uint8(np.squeeze(plane))

# ---------- View (cover fit, clamp, zoom & pan) ----------
def fit_cover_view_to_canvas():
    # Auto fit the image to cover the canvas area while preserving aspect ratio
    global zoom, min_zoom, offset_x, offset_y
    if base_pil is None:
        return
    iw, ih = base_pil.size
    cw = max(1, canvas.winfo_width() or 800)
    ch = max(1, canvas.winfo_height() or 420)
    if iw == 0 or ih == 0:
        return
    cover = max(cw / iw, ch / ih)
    min_zoom = cover
    zoom = max(zoom, cover)
    sw, sh = int(iw * zoom), int(ih * zoom)
    offset_x = (cw - sw) // 2
    offset_y = (ch - sh) // 2

def clamp_offsets():
    # Prevent panning beyond the image edges
    global offset_x, offset_y
    if base_pil is None:
        return
    iw, ih = base_pil.size
    sw = max(1, int(iw * zoom))
    sh = max(1, int(ih * zoom))
    cw = max(1, canvas.winfo_width() or 800)
    ch = max(1, canvas.winfo_height() or 420)
    offset_x = min(0, max(cw - sw, offset_x))
    offset_y = min(0, max(ch - sh, offset_y))

def redraw_canvas():
    # Draw the current PIL image onto the canvas
    global display_imgtk
    canvas.delete("all")
    if base_pil is None:
        draw_placeholder()
        return
    clamp_offsets()
    iw, ih = base_pil.size
    sw = max(1, int(iw * zoom))
    sh = max(1, int(ih * zoom))
    disp = base_pil.resize((sw, sh), Image.Resampling.LANCZOS)
    display_imgtk = ImageTk.PhotoImage(disp)
    canvas.create_image(offset_x, offset_y, image=display_imgtk, anchor="nw")

    # Draw legend after image is drawn
    draw_legend_on_canvas()

def canvas_reset_view(event=None):
    # Reset zoom and recenter
    if current_view != "image":
        return
    fit_cover_view_to_canvas()
    redraw_canvas()

def canvas_start_drag(event):
    # Begin panning
    global drag_start
    if current_view != "image":
        return
    drag_start = (event.x, event.y)

def canvas_drag(event):
    # Continue panning
    global offset_x, offset_y, drag_start
    if current_view != "image":
        return
    if drag_start is None or base_pil is None:
        return
    dx = event.x - drag_start[0]
    dy = event.y - drag_start[1]
    offset_x += dx
    offset_y += dy
    drag_start = (event.x, event.y)
    redraw_canvas()

def canvas_end_drag(event):
    # End panning
    global drag_start
    drag_start = None

def canvas_zoom(event, factor=None):
    # Zoom with mouse wheel or programmatic factor
    global zoom, offset_x, offset_y
    if current_view != "image" or base_pil is None:
        return
    if factor is None:
        if getattr(event, "num", None) == 4:
            factor = 1.1
        elif getattr(event, "num", None) == 5:
            factor = 1/1.1
        else:
            factor = 1.1 if getattr(event, "delta", 0) > 0 else 1/1.1
    new_zoom = max(min(zoom * factor, max_zoom), min_zoom)
    if abs(new_zoom - zoom) < 1e-6:
        return
    mx, my = event.x, event.y
    img_x = (mx - offset_x) / zoom
    img_y = (my - offset_y) / zoom
    zoom = new_zoom
    offset_x = mx - img_x * zoom
    offset_y = my - img_y * zoom
    redraw_canvas()

# ---------- Placeholder ----------
def draw_placeholder():
    # Draw a neutral placeholder when no image is loaded
    cw = max(1, canvas.winfo_width() or 800)
    ch = max(1, canvas.winfo_height() or 420)
    pad = 12
    side = min(cw, ch) - 2 * pad
    side = max(side, 60)
    x0 = (cw - side) // 2
    y0 = (ch - side) // 2
    x1 = x0 + side
    y1 = y0 + side
    canvas.create_rectangle(x0, y0, x1, y1, outline="#444", width=2, fill="#1b1b1b")
    canvas.create_text((cw // 2), (ch // 2) - 12,
                       text="No image loaded",
                       fill="#bbb", font=("Helvetica", 14, "bold"))
    canvas.create_text((cw // 2), (ch // 2) + 12,
                       text="Click 'Browse files' to open .czi",
                       fill="#888", font=("Helvetica", 11))

# ---------- Canvas sizing to image aspect ----------
def size_canvas_to_aspect(aspect: float):
    # Adjust canvas widget size to match image aspect ratio
    root.update_idletasks()
    left_w = left.winfo_width() or 0
    win_w = max(1, root.winfo_width())
    win_h = max(1, root.winfo_height())
    avail_w = max(1, win_w - left_w)
    avail_h = max(1, win_h)
    if avail_w / avail_h > aspect:
        new_h = avail_h
        new_w = int(new_h * aspect)
    else:
        new_w = avail_w
        new_h = int(new_w / aspect)
    canvas.config(width=new_w, height=new_h)
    canvas_container.config(width=new_w, height=new_h)

def refresh_layout_and_view():
    # Recompute layout and redraw the image
    global zoom
    if current_view != "image":
        return
    if base_pil is None:
        size_canvas_to_aspect(1.0)
        redraw_canvas()
        return
    iw, ih = base_pil.size
    aspect = iw / ih if ih != 0 else 1.0
    size_canvas_to_aspect(aspect)
    prev_zoom = zoom
    fit_cover_view_to_canvas()
    zoom = prev_zoom if prev_zoom > min_zoom else min_zoom
    redraw_canvas()

# ---------- Metadata parsing ----------
def parse_czi_metadata_from_czi(czi: CziFile) -> dict:
    # Extract a small, robust subset of CZI metadata
    root = czi.meta
    try:
        _find = root.find; _findall = root.findall
    except AttributeError:
        root = root.getroot()
        _find = root.find; _findall = root.findall

    def find_text(elem, path):
        if elem is None: return None
        tag = elem.find(path)
        return tag.text.strip() if tag is not None and tag.text else None

    result = {}
    image = _find("Metadata/Information/Image")
    result['SizeX'] = find_text(image, "SizeX")
    result['SizeY'] = find_text(image, "SizeY")
    result['SizeZ'] = find_text(image, "SizeZ")
    result['SizeC'] = find_text(image, "SizeC")
    result['PixelType'] = find_text(image, "PixelType")

    result['PixelSize'] = {}
    for dist in _findall("Metadata/Scaling/Items/Distance"):
        axis = dist.attrib.get("Id") if dist is not None else None
        val = find_text(dist, "Value")
        if axis and val:
            try: result['PixelSize'][axis] = float(val)
            except ValueError: result['PixelSize'][axis] = val

    result['Channels'] = []
    for ch in _findall("Metadata/Information/Image/Dimensions/Channels/Channel"):
        det_node = ch.find("DetectionWavelength")
        detection = None
        if det_node is not None:
            detection = find_text(det_node, "Ranges")
        result['Channels'].append({
            "Name": ch.attrib.get("Name"),
            "Fluor": find_text(ch, "Fluor"),
            "Excitation": find_text(ch, "ExcitationWavelength"),
            "Emission": find_text(ch, "EmissionWavelength"),
            "DetectionRange": detection
        })

    obj = _find("Metadata/Information/Instrument/Objectives/Objective")
    manuf = obj.find("Manufacturer") if obj is not None else None
    result['Objective'] = {
        "Model": find_text(manuf, "Model") if manuf is not None else None,
        "NA": find_text(obj, "LensNA") if obj is not None else None,
        "Magnification": find_text(obj, "NominalMagnification") if obj is not None else None,
        "Immersion": find_text(obj, "Immersion") if obj is not None else None
    } if obj is not None else {}

    scope = _find("Metadata/Information/Instrument/Microscopes/Microscope")
    result['Microscope'] = find_text(scope, "System") if scope is not None else None
    return result

# ---------- Overlay builder ----------
def get_overlay_pil(c_idx: int, z_idx: int) -> Image.Image:
    """
    Build overlay PIL image (skeleton + branch points) for the current image_array.
    Uses process_channel(image_array, ch, z) from analyze.py and caches by (C, Z).
    Also computes overlay legend stats for C=0 (dead) and C=1 (live).
    """
    key = (int(c_idx), int(z_idx))
    if key in overlay_cache:
        return overlay_cache[key]

    if image_array is None:
        raise RuntimeError("No image loaded.")

    # Run the processing pipeline for the selected (C, Z)
    result = process_channel(image_array, int(c_idx), int(z_idx))
    overlay_bgr = result["overlay_bgr"]

    # Convert BGR from OpenCV to RGB for PIL
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_rgb)

    # ---- collect overlay stats for legend when in analyze mode ----
    try:
        which = "dead" if int(c_idx) == 0 else ("live" if int(c_idx) == 1 else "other")
        cur_count = int(result.get("num_skeletons", 0))

        if which == "dead":
            dead_count = cur_count
            live_count = 0
        elif which == "live":
            dead_count = 0
            live_count = cur_count
        else:
            overlay_stats_cache[(int(c_idx), int(z_idx))] = None
            overlay_cache[key] = pil_img
            return pil_img

        total = dead_count + live_count
        dead_prop = (dead_count / total) if total > 0 else 0.0
        live_prop = (live_count / total) if total > 0 else 0.0

        # Build a minimal rgb_image so we can reuse intensity_rgb(...)
        cur_u8 = result.get("brightest_u8")
        if cur_u8 is None:
            hldr = np.zeros_like(extract_plane(image_array, axes_labels, int(c_idx), int(z_idx)))
            cur_u8 = normalize_to_uint8(hldr)

        zero = np.zeros_like(cur_u8)
        if which == "dead":
            rgb_image = compose_rgb(
                dead_brightest=cur_u8,
                live_brightest=zero,
                dead_count=dead_count,
                live_count=live_count,
            )
        else:
            rgb_image = compose_rgb(
                dead_brightest=zero,
                live_brightest=cur_u8,
                dead_count=dead_count,
                live_count=live_count,
            )

        percentage, _, _ = intensity_rgb(rgb_image)

        overlay_stats_cache[(int(c_idx), int(z_idx))] = {
            "which": which,
            "dead_count": dead_count,
            "live_count": live_count,
            "dead_prop": dead_prop,
            "live_prop": live_prop,
            "percentage": percentage,
        }
    except Exception:
        overlay_stats_cache[(int(c_idx), int(z_idx))] = None

    overlay_cache[key] = pil_img
    return pil_img

# ---------- Merge builder (Option 2) ----------
def get_merge_pil(z_idx: int) -> Image.Image:
    """
    Compose RGB image (G = live, R = dead) using channels [0, 1] as [dead, live].
    C is hidden while Option 2 is active; we cache per Z slice for speed.
    Also compute stats for the legend (dead/live proportions, ratio, intensity).
    """
    key = int(z_idx)
    if image_array is None:
        raise RuntimeError("No image loaded.")
    if "C" not in axes_labels:
        raise RuntimeError("Image has no channel axis (C).")
    nC = get_count("C", image_array.shape, axes_labels)
    if nC < 2:
        raise RuntimeError("Need at least 2 channels for merging (expects [dead, live] as [0,1]).")

    # Fast path: if image cached and stats already computed, return cached image
    if key in merge_cache and key in merge_stats_cache:
        return merge_cache[key]

    # If image cached but stats missing, recompute stats without replacing cached PIL
    if key in merge_cache and key not in merge_stats_cache:
        dead = process_channel(image_array, 0, int(z_idx))
        live = process_channel(image_array, 1, int(z_idx))
        rgb_image = compose_rgb(
            dead_brightest=dead["brightest_u8"],
            live_brightest=live["brightest_u8"],
            dead_count=dead["num_skeletons"],
            live_count=live["num_skeletons"],
        )
        _fill_merge_stats_cache_from_results(key, dead, live, rgb_image)
        return merge_cache[key]

    # Normal path: compute both image and stats
    dead = process_channel(image_array, 0, int(z_idx))
    live = process_channel(image_array, 1, int(z_idx))

    rgb_image = compose_rgb(
        dead_brightest=dead["brightest_u8"],
        live_brightest=live["brightest_u8"],
        dead_count=dead["num_skeletons"],
        live_count=live["num_skeletons"],
    )
    pil_img = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))  # keep safe to RGB

    # Cache image
    merge_cache[key] = pil_img

    # Compute & cache stats for legend
    _fill_merge_stats_cache_from_results(key, dead, live, rgb_image)

    return pil_img

def _fill_merge_stats_cache_from_results(key: int, dead: dict, live: dict, rgb_image: np.ndarray):
    """Compute dead/live proportions, ratio and intensity percentage; store to merge_stats_cache."""
    dead_count = int(dead.get("num_skeletons", 0))
    live_count = int(live.get("num_skeletons", 0))
    total = dead_count + live_count

    if total > 0:
        dead_prop = dead_count / total
        live_prop = live_count / total
        ratio = (dead_count / live_count) if live_count > 0 else float("inf")
    else:
        dead_prop = 0.0
        live_prop = 0.0
        ratio = None  # N/A

    percentage, _, _ = intensity_rgb(rgb_image)

    merge_stats_cache[key] = {
        "dead_count": dead_count,
        "live_count": live_count,
        "dead_prop": dead_prop,
        "live_prop": live_prop,
        "ratio": ratio,
        "percentage": percentage,
    }

# ---------- Legend drawer ----------
def draw_legend_on_canvas():
    """Draw a small legend at bottom-right.
       - Merge: Dead%, Live%, Dead/Live, Intensity
       - Mark (overlay): if C==0 show Dead only; if C==1 show Live only; else hidden
    """
    if base_pil is None or current_view != "image":
        return

    lines = None

    if tick_merge.get():
        key = int(_current_z_idx)
        st = merge_stats_cache.get(key)
        if not st:
            return
        ratio = st["ratio"]
        ratio_str = f"{ratio:.3f}" if (ratio is not None and ratio != float('inf')) else ("∞" if ratio == float('inf') else "N/A")
        lines = [
            f"Dead: {st['dead_prop']:.2%} ({st['dead_count']})",
            f"Live: {st['live_prop']:.2%} ({st['live_count']})",
            f"Dead/Live: {ratio_str}",
            f"Intensity: {st['percentage']}",
        ]

    elif tick_analyze.get():
        key = (int(_current_c_idx), int(_current_z_idx))
        st = overlay_stats_cache.get(key)
        if not st:
            return
        if st.get("which") == "dead":
            lines = [
                f"Dead: {st['dead_count']} cell(s)"
            ]
        elif st.get("which") == "live":
            lines = [
                f"Live: {st['live_count']} cell(s)"
            ]
        else:
            return  # hide for channels other than 0/1
    else:
        return

    # Compose text and draw (nudged up from bottom)
    text = "\n".join(lines)
    pad_x = 10
    pad_y = 30
    x = (canvas.winfo_width() or 0) - pad_x
    y = (canvas.winfo_height() or 0) - pad_y

    text_id = canvas.create_text(
        x, y,
        text=text,
        fill="#eeeeee",
        font=("Helvetica", 10, "bold"),
        anchor="se",
        tags=("legend_text",),
        justify="right"
    )
    bbox = canvas.bbox(text_id)
    if bbox:
        x0, y0, x1, y1 = bbox
        bg_pad = 6
        rect_id = canvas.create_rectangle(
            x0 - bg_pad, y0 - bg_pad, x1 + bg_pad, y1 + bg_pad,
            fill="#000000", outline="#666666", width=1, tags=("legend_bg",)
        )
        canvas.tag_lower(rect_id, text_id)

# ---------- Preview update ----------
def update_preview():
    """Update preview image with priority: Merge > Mark (overlay) > raw plane."""
    global base_pil, zoom, _current_c_idx, _current_z_idx
    if image_array is None:
        if current_view == "image":
            redraw_canvas()
        return
    try:
        # Read indices (C only relevant if Merge is OFF)
        c_idx = int(spin_c.get()) if (spin_c['state'] == "normal") else 0
        z_idx = int(spin_z.get()) if spin_z['state'] == "normal" else 0

        # Track current indices for legend
        _current_c_idx, _current_z_idx = c_idx, z_idx

        if tick_merge.get():
            base_pil = get_merge_pil(z_idx)
        elif tick_analyze.get():
            base_pil = get_overlay_pil(c_idx, z_idx)
        else:
            plane = extract_plane(image_array, axes_labels, c_idx, z_idx)
            base_pil = Image.fromarray(plane)

        zoom = 1.0
        if current_view == "image":
            refresh_layout_and_view()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ---- Load CZI file ----
def load_czi(path: Path):
    # Open .czi, load image array and metadata; reset caches
    global czi_obj, image_array, axes_labels, meta_dict
    if czi_obj is not None:
        czi_obj = None
    try:
        czi_obj = CziFile(str(path))
        arr, info = czi_obj.read_image()
        image_array = arr
        axes_labels = infer_axes(arr.shape, info)

        # Reset caches when switching to a new file
        overlay_cache.clear()
        merge_cache.clear()
        merge_stats_cache.clear()
        overlay_stats_cache.clear()

        file_var.set(path.name)
        shape_var.set(str(tuple(arr.shape)))
        axes_var.set(axes_labels)

        nC = get_count("C", arr.shape, axes_labels)
        nZ = get_count("Z", arr.shape, axes_labels)
        cnum_var.set(str(nC)); znum_var.set(str(nZ))

        spin_c.config(from_=0, to=max(0, nC - 1), state="normal" if (nC > 1 and not tick_merge.get()) else "disabled", wrap=True, increment=1)
        spin_c.delete(0, "end"); spin_c.insert(0, "0")
        spin_z.config(from_=0, to=max(0, nZ - 1), state="normal" if nZ > 1 else "disabled", wrap=True, increment=1)
        spin_z.delete(0, "end"); spin_z.insert(0, "0")

        # Show/hide C control according to Merge state
        _apply_c_visibility()

        meta_dict = parse_czi_metadata_from_czi(czi_obj)

        update_preview()
        if current_view == "info":
            render_info_view()
    except Exception as e:
        image_array = None; axes_labels = ""; meta_dict = {}
        czi_obj = None
        messagebox.showerror("Error", f"Failed to open CZI:\n{e}")
        file_var.set("-"); shape_var.set("-"); axes_var.set("-")
        cnum_var.set("-"); znum_var.set("-")
        if current_view == "image": refresh_layout_and_view()
        else: render_info_view()

# ---- Multi-file + GUI plumbing ----
def on_select_file(event=None):
    sel = listbox.curselection()
    if not sel: return
    load_czi(Path(file_list[sel[0]]))

def browse_files():
    global file_list
    fns = filedialog.askopenfilenames(
        title="Choose CZI files",
        filetypes=[("Zeiss CZI files", "*.czi"), ("All files", "*.*")]
    )
    if not fns: return
    file_list = list(fns)
    listbox.delete(0, tk.END)
    for f in file_list: listbox.insert(tk.END, Path(f).name)
    listbox.selection_set(0)
    load_czi(Path(file_list[0]))

def on_close():
    try:
        pass
    finally:
        root.destroy()

# ---------- Toggle between Image and Info ----------
def set_view(view_name: str):
    global current_view, zoom
    if view_name == current_view:
        return
    current_view = view_name
    if current_view == "image":
        btn_image.config(relief="sunken"); btn_info.config(relief="raised")
        info_frame.pack_forget()
        canvas_stack.pack(fill="both", expand=True)
        canvas_container.pack(fill="both", expand=True)
        zoom = 1.0
        refresh_layout_and_view()
    else:
        btn_image.config(relief="raised"); btn_info.config(relief="sunken")
        canvas_container.pack_forget()
        canvas_stack.pack(fill="both", expand=True)
        info_frame.pack(fill="both", expand=True)
        render_info_view()

def render_info_view():
    # Render a scrollable metadata view
    for w in info_inner.winfo_children():
        w.destroy()

    tk.Label(info_inner, text="Image Information", fg="#ddd", bg="#1a1a1a",
             font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 8))

    def kv_row(parent, key, val):
        rowf = tk.Frame(parent, bg="#1a1a1a"); rowf.pack(fill="x", pady=1)
        tk.Label(rowf, text=f"{key}", fg="#bbb", bg="#1a1a1a",
                 width=14, anchor="w").pack(side="left")
        tk.Label(rowf, text=val, fg="#eee", bg="#1a1a1a",
                 anchor="w", wraplength=900, justify="left").pack(side="left", fill="x", expand=True)

    imf = tk.Frame(info_inner, bg="#1a1a1a"); imf.pack(fill="x", pady=(0,4))
    tk.Label(imf, text="Image:", fg="#ddd", bg="#1a1a1a",
             font=("Helvetica", 12, "bold")).pack(anchor="w")
    def add_if(k, v):
        if v is not None and v != "":
            kv_row(imf, f"  {k}", str(v))
    add_if("SizeX", meta_dict.get("SizeX"))
    add_if("SizeY", meta_dict.get("SizeY"))
    add_if("SizeZ", meta_dict.get("SizeZ"))
    add_if("SizeC", meta_dict.get("SizeC"))
    add_if("PixelType", meta_dict.get("PixelType"))

    ps = meta_dict.get("PixelSize") or {}
    if ps:
        psf = tk.Frame(info_inner, bg="#1a1a1a"); psf.pack(fill="x", pady=(4,4))
        tk.Label(psf, text="PixelSize:", fg="#ddd", bg="#1a1a1a",
                 font=("Helvetica", 12, "bold")).pack(anchor="w")
        for axis, val in ps.items():
            kv_row(psf, f"  {axis}", str(val))

    chs = meta_dict.get("Channels") or []
    chf = tk.Frame(info_inner, bg="#1a1a1a"); chf.pack(fill="x", pady=(4,4))
    tk.Label(chf, text=f"Channels ({len(chs)}):", fg="#ddd", bg="#1a1a1a",
             font=("Helvetica", 12, "bold")).pack(anchor="w")
    if chs:
        for i, ch in enumerate(chs, 1):
            blk = tk.Frame(chf, bg="#1a1a1a", highlightthickness=0)
            blk.pack(fill="x", pady=(2,2))
            kv_row(blk, f"  [{i}] Name", str(ch.get("Name") or ""))
            def chrow(k, v):
                kv_row(blk, f"    {k}", str(v) if v is not None else "")
            chrow("Fluor", ch.get("Fluor"))
            chrow("Excitation", ch.get("Excitation"))
            chrow("Emission", ch.get("Emission"))
            chrow("DetectionRange", ch.get("DetectionRange"))
    else:
        kv_row(chf, "  (none)", "")

    obj = meta_dict.get("Objective") or {}
    of = tk.Frame(info_inner, bg="#1a1a1a"); of.pack(fill="x", pady=(4,4))
    tk.Label(of, text="Objective:", fg="#ddd", bg="#1a1a1a",
             font=("Helvetica", 12, "bold")).pack(anchor="w")
    if obj:
        for k in ["Model", "NA", "Magnification", "Immersion"]:
            kv_row(of, f"  {k}", str(obj.get(k)) if obj.get(k) is not None else "")
    else:
        kv_row(of, "  (none)", "")

    ms = meta_dict.get("Microscope")
    mf = tk.Frame(info_inner, bg="#1a1a1a"); mf.pack(fill="x", pady=(4,0))
    tk.Label(mf, text="Microscope:", fg="#ddd", bg="#1a1a1a",
             font=("Helvetica", 12, "bold")).pack(anchor="w")
    kv_row(mf, "  System", str(ms) if ms is not None else "")

    info_inner.update_idletasks()
    info_canvas.configure(scrollregion=info_canvas.bbox("all"))

# ---- Show/Hide C selector (when Merge toggled) ----
def _apply_c_visibility():
    # Keep C selector visible but disabled when Merge is ON
    if tick_merge.get():
        spin_c.config(state="disabled")
        try:
            lbl_c.config(fg="#777")
        except Exception:
            pass
    else:
        nC = get_count("C", image_array.shape, axes_labels) if image_array is not None else 0
        spin_c.config(state="normal" if nC > 1 else "disabled")
        try:
            lbl_c.config(fg="black")
        except Exception:
            pass

# ---- Tick callbacks ----
def on_tick_analyze():
    # Make analyze & merge mutually exclusive
    if tick_analyze.get():
        if tick_merge.get():
            tick_merge.set(False)
            _apply_c_visibility()
    update_preview()

def on_tick_merge():
    # Make merge & analyze mutually exclusive
    if tick_merge.get():
        if tick_analyze.get():
            tick_analyze.set(False)
    _apply_c_visibility()
    update_preview()

# ---- GUI ----
root = tk.Tk()
root.title("Multi-CZI Analyzer")
root.geometry("1100x720")
last_wm_state = root.state()

# Left panel
left = tk.Frame(root, padx=6, pady=6)
left.pack(side="left", fill="y")

controls = tk.Frame(left)
controls.pack(fill="x")
btn_browse = tk.Button(controls, text="Browse files", command=browse_files)
btn_browse.pack(side="left", fill="x", expand=True)

listbox = tk.Listbox(left, height=15, exportselection=False)
listbox.pack(fill="both", expand=True, pady=6)
listbox.bind("<<ListboxSelect>>", on_select_file)

# Container frame right under the listbox
frame_bottom = tk.Frame(left)
frame_bottom.pack(fill="x", pady=(0,6))

# First checkbox → overlay (mark)
tick_analyze = tk.BooleanVar(value=False)
chk1 = tk.Checkbutton(frame_bottom, text="Mark",
                      variable=tick_analyze, command=on_tick_analyze)
chk1.pack(side="left")

# Second checkbox → merge dead/live and hide C
tick_merge = tk.BooleanVar(value=False)
chk2 = tk.Checkbutton(frame_bottom, text="Merge",
                      variable=tick_merge, command=on_tick_merge)
chk2.pack(side="left", padx=(10,0))

sel = tk.Frame(left, pady=6)
sel.pack(fill="x")
lbl_c = tk.Label(sel, text="C:")
lbl_c.pack(side="left")
spin_c = tk.Spinbox(sel, from_=0, to=0, width=5, state="disabled",
                    wrap=True, increment=1, command=update_preview)
spin_c.pack(side="left")
tk.Label(sel, text="Z:").pack(side="left", padx=(8, 0))
spin_z = tk.Spinbox(sel, from_=0, to=0, width=5, state="disabled",
                    wrap=True, increment=1, command=update_preview)
spin_z.pack(side="left")

# Right side (toolbar + stack area)
right = tk.Frame(root)
right.pack(side="left", fill="both", expand=True)

toolbar = tk.Frame(right, pady=4)
toolbar.pack(fill="x")
btn_image = tk.Button(toolbar, text="Image", width=8, relief="sunken",
                      command=lambda: set_view("image"))
btn_info  = tk.Button(toolbar, text="Info",  width=8, relief="raised",
                      command=lambda: set_view("info"))
btn_image.pack(side="left", padx=(0,4))
btn_info.pack(side="left")

# Stack area
canvas_stack = tk.Frame(right)
canvas_stack.pack(fill="both", expand=True)

# --- Image view ---
canvas_container = tk.Frame(canvas_stack, bg="#111")
canvas_container.pack(fill="both", expand=True)
canvas = tk.Canvas(canvas_container, bg="#222", cursor="tcross", highlightthickness=0)
canvas.place(relx=0.5, rely=0.5, anchor="center")

# --- Info view (scrollable) ---
info_frame = tk.Frame(canvas_stack, bg="#1a1a1a")
info_canvas = tk.Canvas(info_frame, bg="#1a1a1a", highlightthickness=0)
info_scrollbar = tk.Scrollbar(info_frame, orient="vertical", command=info_canvas.yview)
info_canvas.configure(yscrollcommand=info_scrollbar.set)
info_scrollbar.pack(side="right", fill="y")
info_canvas.pack(side="left", fill="both", expand=True)
info_inner = tk.Frame(info_canvas, bg="#1a1a1a")
info_window = info_canvas.create_window((0, 0), window=info_inner, anchor="nw")

def _on_info_configure(event):
    # Keep inner frame width equal to visible width for better wrapping
    info_canvas.itemconfig(info_window, width=info_canvas.winfo_width())
    info_canvas.configure(scrollregion=info_canvas.bbox("all"))
info_canvas.bind("<Configure>", _on_info_configure)

# Mouse wheel bindings for INFO view (scroll)
def info_on_mousewheel(event):
    if current_view != "info": return
    if getattr(event, "num", None) == 4:
        info_canvas.yview_scroll(-3, "units")
    elif getattr(event, "num", None) == 5:
        info_canvas.yview_scroll(3, "units")
    else:
        direction = -1 if event.delta > 0 else 1
        info_canvas.yview_scroll(direction * 3, "units")

# Bind wheel events to info_canvas
info_canvas.bind("<MouseWheel>", info_on_mousewheel)   # Windows/Mac
info_canvas.bind("<Button-4>", info_on_mousewheel)     # Linux up
info_canvas.bind("<Button-5>", info_on_mousewheel)     # Linux down

# Simple info vars init
file_var  = tk.StringVar(value="-")
shape_var = tk.StringVar(value="-")
axes_var  = tk.StringVar(value="-")
cnum_var  = tk.StringVar(value="-")
znum_var  = tk.StringVar(value="-")

# Bindings (image view)
canvas.bind("<MouseWheel>", canvas_zoom)
canvas.bind("<Button-4>", canvas_zoom)
canvas.bind("<Button-5>", canvas_zoom)
canvas.bind("<Button-1>", canvas_start_drag)
canvas.bind("<B1-Motion>", canvas_drag)
canvas.bind("<ButtonRelease-1>", canvas_end_drag)
canvas.bind("<Double-Button-1>", canvas_reset_view)

spin_c.bind("<Return>", lambda e: update_preview())
spin_c.bind("<FocusOut>", lambda e: update_preview())
spin_z.bind("<Return>", lambda e: update_preview())
spin_z.bind("<FocusOut>", lambda e: update_preview())

def kb_zoom_in(event):
    if current_view != "image": return
    class E: pass
    e = E(); e.x = canvas.winfo_width()//2; e.y = canvas.winfo_height()//2; e.delta = 120
    canvas_zoom(e)

def kb_zoom_out(event):
    if current_view != "image": return
    class E: pass
    e = E(); e.x = canvas.winfo_width()//2; e.y = canvas.winfo_height()//2; e.delta = -120
    canvas_zoom(e)

def kb_reset(event):
    canvas_reset_view()

root.bind("+", kb_zoom_in)
root.bind("-", kb_zoom_out)
root.bind("<space>", kb_reset)

# Maximize/restore → auto cover-fit (image view)
def on_root_resize(event):
    global last_wm_state
    cur = root.state()
    if cur != last_wm_state:
        last_wm_state = cur
        canvas_reset_view()
        return
    refresh_layout_and_view()

root.bind("<Configure>", on_root_resize)
root.protocol("WM_DELETE_WINDOW", on_close)

# Initial placeholder layout/draw
refresh_layout_and_view()
root.mainloop()
