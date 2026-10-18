"""Compositor for the edited Terrain Solver demo video.

Renders the storyboard from edit_storyboard.py into frame sequences
(JPEG for video, PNG for QC stills). Encoding/muxing is done by the
calling sbatch script via ffmpeg.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import h5py

import edit_storyboard as SB

FONT_FILES = {
    "bold": "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "reg": "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
}
STYLE_FILE = {"title": "bold", "head": "bold", "body": "reg", "small": "reg"}

_FONTS = {}
_META = None
_CACHE = {}
_CACHE_ORDER = []
_CACHE_MAX = 10


def font_for(style, size):
    key = (style, size)
    f = _FONTS.get(key)
    if f is None:
        f = ImageFont.truetype(FONT_FILES[STYLE_FILE[style]], size)
        _FONTS[key] = f
    return f


def load_meta(path):
    with h5py.File(path, "r") as f:
        y, x = SB.AACHEN_CELL
        return {
            "step": np.asarray(f["step_index"][:]),
            "max_water": np.asarray(f["max_water"][:]),
            "moved": np.asarray(f["moved_this_step"][:]),
            "aachen": np.asarray(f["water"][:, y, x]),
        }


_SRC_MAP = {}


def build_src_maps():
    for view, d in (("top", SB.TOP_DIR), ("slice", SB.SLICE_DIR)):
        m = {}
        for name in os.listdir(d):
            if name.startswith("frame_") and name.endswith(".png"):
                try:
                    idx = int(name.split("_")[1])
                except ValueError:
                    continue
                m[idx] = os.path.join(d, name)
        _SRC_MAP[view] = m


def _find_source(view, idx):
    if not _SRC_MAP:
        build_src_maps()
    try:
        return _SRC_MAP[view][idx]
    except KeyError:
        raise FileNotFoundError("no source frame %s/%d" % (view, idx))


def load_src(view, idx):
    key = (view, idx)
    img = _CACHE.get(key)
    if img is not None:
        return img
    path = _find_source(view, idx)
    img = Image.open(path).convert("RGB")
    patch = SB.TOP_PANEL_PATCH if view == "top" else SB.SLICE_PANEL_PATCH
    fill = (7, 9, 13) if view == "top" else (0, 0, 0)
    d = ImageDraw.Draw(img)
    d.rectangle(patch, fill=fill)
    _CACHE[key] = img
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _CACHE.pop(old, None)
    return img


def push_in(img, zoom):
    if zoom is None or abs(zoom - 1.0) < 1e-4:
        return img
    cx, cy = SB.CIRCLE_CENTER
    cw, ch = img.width / zoom, img.height / zoom
    x0 = int(round(cx - cw / 2))
    y0 = int(round(cy - ch / 2))
    x0 = max(0, min(img.width - int(round(cw)), x0))
    y0 = max(0, min(img.height - int(round(ch)), y0))
    box = (x0, y0, x0 + int(round(cw)), y0 + int(round(ch)))
    return img.crop(box).resize((img.width, img.height), Image.LANCZOS)


_BLUR_CACHE = {}


def blurred_base(img, key):
    out = _BLUR_CACHE.get(key)
    if out is None:
        small = img.resize((540, 270), Image.BILINEAR).filter(
            ImageFilter.GaussianBlur(26))
        out = small.resize((W, H), Image.LANCZOS)
        _BLUR_CACHE[key] = out
    return out


def paste_pip(base, entry):
    rect = entry["pip"]
    if rect is None:
        return
    a = entry.get("pip_alpha", 1.0)
    if a <= 0.003:
        return
    x, y, w, h = rect
    w = max(2, w)
    h = max(2, h)
    crop_top = entry.get("pip_ctop")
    if crop_top is None:
        crop_top = SB.PIP_CROP_TOP
    crop_top = max(0, min(1079 - 2, crop_top))
    src = load_src("slice", entry["pip_src"])
    crop = src.crop((0, crop_top, src.width, src.height))
    pip = crop.resize((w, h), Image.LANCZOS)
    if a >= 0.997:
        base.paste(pip, (x, y))
        return
    mask = Image.new("L", (w, h), int(round(255 * a)))
    base.paste(pip, (x, y), mask)


def draw_pip_frame(overlay, entry):
    rect = entry["pip"]
    if rect is None:
        return
    x, y, w, h = rect
    tag = entry.get("pip_tag", 0.0)
    a = entry.get("pip_alpha", 1.0)
    if tag <= 0.02 or a <= 0.05:
        return
    d = ImageDraw.Draw(overlay, "RGBA")
    alpha = int(200 * a)
    d.rectangle((x - 1, y - 1, x + w + 1, y + h + 1),
                outline=(230, 238, 250, alpha), width=2)
    chip_a = a * min(1.0, max(0.0, (tag - 0.1) / 0.25))
    if chip_a <= 0.02:
        return
    label = "EQUATORIAL CROSS-SECTION (z = 540)"
    f = font_for("small", 24)
    tw = min(d.textlength(label, font=f), max(60, w - 70))
    chip_h = 40
    chip_x0 = x + 10
    chip_y0 = y + 10
    d.rounded_rectangle((chip_x0, chip_y0, chip_x0 + tw + 28,
                         chip_y0 + chip_h),
                        radius=8, fill=(9, 12, 19, int(215 * chip_a)))
    d.text((chip_x0 + 14, chip_y0 + (chip_h - 24) / 2 - 3), label,
           font=f, fill=(225, 233, 246, int(235 * chip_a)))


def draw_scrim(overlay, item):
    a = item["alpha"]
    if a <= 0.001:
        return
    color = item.get("color", (0, 0, 0))
    layer = Image.new("RGBA", (W, H),
                      (color[0], color[1], color[2], int(round(255 * a))))
    overlay.alpha_composite(layer, (0, 0))


def draw_card(overlay, item):
    a = item["alpha"]
    if a <= 0.004:
        return
    x, y, w = item["x"], item["y"], item["w"]
    pad = item["pad"]
    lines = item["lines"]
    d = ImageDraw.Draw(overlay, "RGBA")

    layout = []
    for entry in lines:
        text, style, size, color = entry
        if text is None:
            layout.append((None, None, None, size))
            continue
        f = font_for(style, size)
        asc, desc = f.getmetrics()
        layout.append((text, f, color, asc + desc))

    box_h = 2 * pad + sum(e[3] for e in layout)
    bg = item["bg"]
    if item.get("shadow", True):
        d.rounded_rectangle((x + 6, y + 8, x + w + 6, y + box_h + 8),
                            radius=item["radius"],
                            fill=(0, 0, 0, int(90 * a)))
    d.rounded_rectangle((x, y, x + w, y + box_h), radius=item["radius"],
                        fill=(bg[0], bg[1], bg[2], int(round(bg[3] * a))))
    accent = item.get("accent")
    if accent:
        d.rounded_rectangle((x, y + 16, x + 7, y + box_h - 16), radius=4,
                            fill=(accent[0], accent[1], accent[2],
                                  int(round(220 * a))))
    ty = y + pad
    align = item.get("align", "left")
    for entry in layout:
        text, f, color, lh = entry
        if text is None:
            ty += lh
            continue
        tx = x + pad
        if align == "center":
            tw = d.textlength(text, font=f)
            tx = x + (w - tw) / 2
        d.text((tx, ty), text, font=f,
               fill=(color[0], color[1], color[2], int(round(255 * a))))
        ty += lh


def draw_arrow(overlay, item):
    a = item["alpha"]
    if a <= 0.004:
        return
    p1, p2 = item["p1"], item["p2"]
    color = item["color"]
    w = item["width"]
    d = ImageDraw.Draw(overlay, "RGBA")
    col = (color[0], color[1], color[2], int(round(255 * a)))
    d.line((p1, p2), fill=col, width=w)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / ln, dy / ln
    px, py = -uy, ux
    hl, hw = 30.0, 15.0
    tip = (p2[0], p2[1])
    b1 = (p2[0] - ux * hl + px * hw, p2[1] - uy * hl + py * hw)
    b2 = (p2[0] - ux * hl - px * hw, p2[1] - uy * hl - py * hw)
    d.polygon([tip, b1, b2], fill=col)


def draw_ring(overlay, item):
    a = item["alpha"]
    if a <= 0.004:
        return
    cx, cy = item["center"]
    r = item["r"]
    color = item["color"]
    ImageDraw.Draw(overlay, "RGBA").ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        outline=(color[0], color[1], color[2], int(round(255 * a))),
        width=item["width"])


def draw_dot(overlay, item):
    a = item["alpha"]
    if a <= 0.004:
        return
    x, y = item["p"]
    r = item["r"]
    color = item["color"]
    d = ImageDraw.Draw(overlay, "RGBA")
    d.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3),
              fill=(0, 0, 0, int(120 * a)))
    d.ellipse((x - r, y - r, x + r, y + r),
              fill=(color[0], color[1], color[2], int(round(245 * a))))


def draw_hud(overlay, entry):
    hud = entry["hud"]
    if not hud or not hud.get("show"):
        return
    i = entry["hud_src"]
    d = ImageDraw.Draw(overlay, "RGBA")
    x0, y0, x1, y1 = SB.TOP_PANEL_PATCH
    d.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(7, 9, 13, 232))

    f_word = font_for("head", 30)
    d.text((x0 + 28, y0 + 24), "TERRAIN SOLVER", font=f_word,
           fill=(SB.ACCENT[0], SB.ACCENT[1], SB.ACCENT[2], 255))
    f_small = font_for("small", 22)
    d.text((x0 + 28, y0 + 66), "test case for the CPP-ML-Interface",
           font=f_small, fill=(150, 160, 178, 255))
    d.line((x0 + 28, y0 + 104, x1 - 28, y0 + 104),
           fill=(255, 255, 255, 60), width=2)

    label = hud.get("view_label", "")
    words = label.split(" — ")
    ty = y0 + 124
    for part in words:
        d.text((x0 + 28, ty), part, font=f_small,
               fill=(196, 206, 222, 255))
        ty += 30

    step = int(_META["step"][i])
    mw = float(_META["max_water"][i])
    mv = float(_META["moved"][i])
    f_big = font_for("body", 30)
    ty = max(ty + 14, y0 + 210)
    rows = [
        ("step", "%s / %s" % (format(step, ","),
                              format(SB.MAX_FRAME_STEP, ","))),
        ("max depth", "%.1f" % mw),
        ("redistributed / step", format(int(mv), ",")),
    ]
    for name, val in rows:
        d.text((x0 + 28, ty), name, font=f_small, fill=(150, 160, 178, 255))
        d.text((x0 + 28, ty + 26), val, font=f_big, fill=(240, 245, 252, 255))
        ty += 78

    prog = hud.get("progress")
    if prog is None:
        prog = i / float(SB.N_SRC - 1)
    bar_y = H - 7
    d.rectangle((0, bar_y, W, H), fill=(0, 0, 0, 150))
    d.rectangle((0, bar_y, int(W * prog), H),
                fill=(SB.ACCENT[0], SB.ACCENT[1], SB.ACCENT[2], 230))
    for ch in SB.CHAPTERS:
        cx = int(W * ch / float(SB.N_SRC - 1))
        d.rectangle((cx - 1, bar_y - 7, cx + 1, bar_y + 1),
                    fill=(255, 255, 255, 130))


_SPARK = {"key": None, "pts": None}


def spark_points(sx0, sy0, sx1, sy1, ymax):
    key = (sx0, sy0, sx1, sy1, ymax)
    if _SPARK["key"] != key:
        ser = _META["aachen"]
        n = len(ser)
        pts = []
        for j in range(n):
            v = min(max(float(ser[j]), 0.0), ymax)
            pts.append((sx0 + (sx1 - sx0) * j / float(n - 1),
                        sy1 - (sy1 - sy0) * v / ymax))
        _SPARK["key"] = key
        _SPARK["pts"] = pts
    return _SPARK["pts"]


def draw_ticker(overlay, entry):
    if not entry.get("ticker"):
        return
    i = entry["src"]
    d = ImageDraw.Draw(overlay, "RGBA")
    x0, y0, x1, y1 = SB.TICKER_RECT
    d.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(7, 9, 13, 232))

    f_lab = font_for("small", 20)
    d.text((x0 + 24, y0 + 16), "WATER COLUMN AT AACHEN", font=f_lab,
           fill=(150, 160, 178, 255))
    d.text((x0 + 24, y0 + 42), "50.8° N, 6.1° E — tracked from the initial burst",
           font=font_for("small", 18), fill=(120, 130, 148, 255))

    f_big = font_for("head", 38)
    d.text((x0 + 24, y0 + 66), "%.1f" % float(_META["aachen"][i]),
           font=f_big, fill=(SB.ACCENT_WARM[0], SB.ACCENT_WARM[1],
                             SB.ACCENT_WARM[2], 255))

    sx0, sy0, sx1, sy1 = x0 + 24, y0 + 128, x1 - 24, y1 - 42
    ymax = 130.0
    d.line((sx0, sy1, sx1, sy1), fill=(255, 255, 255, 55), width=2)
    pts = spark_points(sx0, sy0, sx1, sy1, ymax)
    sel = pts[:i + 1]
    if len(sel) >= 2:
        d.polygon(sel + [(sel[-1][0], sy1), (sx0, sy1)],
                  fill=(122, 192, 255, 42))
        d.line(sel, fill=(122, 192, 255, 255), width=3, joint="curve")
    cx, cy = pts[i]
    d.line((cx, sy1, cx, cy), fill=(255, 178, 102, 90), width=2)
    d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 178, 102, 255))

    f_ax = font_for("small", 16)
    d.text((sx0, sy1 + 8), "step 0", font=f_ax, fill=(120, 130, 148, 255))
    stepmax = format(SB.MAX_FRAME_STEP, ",")
    tw = d.textlength(stepmax, font=f_ax)
    d.text((sx1 - tw, sy1 + 8), stepmax, font=f_ax,
           fill=(120, 130, 148, 255))
    lab = "depth (0 – %.0f)" % ymax
    tw = d.textlength(lab, font=f_ax)
    d.text((sx1 - tw, sy0 - 22), lab, font=f_ax,
           fill=(120, 130, 148, 255))


W, H = SB.W, SB.H


def compose(f):
    entry = SB.frames()[f]
    base = load_src(entry["view"], entry["src"]).copy()
    if entry.get("cmp_src") is not None and entry.get("cmp_x") is not None:
        x = max(0, min(W, int(entry["cmp_x"])))
        beg = load_src(entry["view"], entry["cmp_src"])
        if x > 0:
            base.paste(beg.crop((0, 0, x, H)), (0, 0))
        dl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dl)
        dd.line((x, 0, x, H), fill=(122, 192, 255, 225), width=4)
        dd.ellipse((x - 18, 520, x + 18, 560), fill=(9, 12, 19, 240),
                   outline=(122, 192, 255, 255), width=4)
        base = Image.alpha_composite(base.convert("RGBA"), dl).convert("RGB")
    if entry["zoom"]:
        base = push_in(base, entry["zoom"])
    if entry.get("bg_blur"):
        base = blurred_base(base, (entry["view"], entry["src"]))
    if entry["pip"] is not None:
        paste_pip(base, entry)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_pip_frame(overlay, entry)
    hud_entry = dict(entry)
    hud_entry["hud_src"] = entry["src"]
    draw_hud(overlay, hud_entry)
    draw_ticker(overlay, entry)
    for item in entry["drawables"]:
        kind = item["kind"]
        if kind == "scrim":
            draw_scrim(overlay, item)
        elif kind == "card":
            draw_card(overlay, item)
        elif kind == "arrow":
            draw_arrow(overlay, item)
        elif kind == "ring":
            draw_ring(overlay, item)
        elif kind == "dot":
            draw_dot(overlay, item)

    out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return out


def render_one(task):
    f, out_dir, jpeg = task
    img = compose(f)
    if jpeg:
        path = os.path.join(out_dir, "%05d.jpg" % f)
        img.save(path, quality=95, subsampling=0)
    else:
        path = os.path.join(out_dir, "%05d.png" % f)
        img.save(path)
    return f


def render_stills(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pts = SB.still_points()
    for name, f in sorted(pts.items()):
        f = max(0, min(SB.total_frames() - 1, int(f)))
        img = compose(f)
        path = os.path.join(out_dir, "%s_f%04d.jpg" % (name, f))
        img.save(path, quality=92, subsampling=0)
        print("still %s -> %s" % (name, path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stills", "preview", "full"],
                    default="full")
    ap.add_argument("--preview-seconds", type=float, default=48.0)
    ap.add_argument("--out-dir", default=SB.OUT_DIR)
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    global _META
    _META = load_meta(SB.TRAJ_H5)
    SB.frames()

    if args.mode == "stills":
        render_stills(os.path.join(args.out_dir, "stills"))
        return

    n = SB.total_frames()
    if args.mode == "preview":
        n = min(n, int(round(args.preview_seconds * SB.FPS)))
    frames_dir = os.path.join(
        args.out_dir, "frames_preview" if args.mode == "preview" else "frames")
    os.makedirs(frames_dir, exist_ok=True)

    t0 = time.time()
    tasks = [(f, frames_dir, True) for f in range(n)]
    done = 0
    with Pool(args.jobs) as pool:
        for _ in pool.imap_unordered(render_one, tasks, chunksize=4):
            done += 1
            if done % 200 == 0 or done == n:
                el = time.time() - t0
                print("frames %d/%d (%.1f s, %.1f fps)" %
                      (done, n, el, done / max(1e-6, el)), flush=True)
    print("rendered %d frames in %.1f s" % (n, time.time() - t0), flush=True)
    with open(os.path.join(args.out_dir,
                           "rendered_%s.txt" % args.mode), "w") as fh:
        fh.write("%d\n" % n)


if __name__ == "__main__":
    main()
