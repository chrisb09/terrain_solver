"""Storyboard for the edited Terrain Solver demo video.

Edit this file to change wording, timing, camera moves and annotation beats.
render_edit.py and make_audio.py consume the timeline built here.
"""

import os

FPS = 30
W, H = 2160, 1080
N_SRC = 3163

RUN_DIR = "/rwthfs/rz/cluster/hpcwork/thes2181/mini_app/circle_r300_d300_s10000000_triangular_2160x1080_c23ms_1n_24t_1c__rank0_gather_none_ts2"
OUT_DIR = os.path.join(RUN_DIR, "edited")
TOP_DIR = os.path.join(RUN_DIR, "rendered_frames")
SLICE_DIR = os.path.join(RUN_DIR, "rendered_slices")
TRAJ_H5 = os.path.join(RUN_DIR, "world_trajectory.h5")

TOP_PANEL_PATCH = (1598, 6, 2160, 548)
SLICE_PANEL_PATCH = (1598, 6, 2160, 548)

PIP_DST = (1488, 812, 664, 258)
PIP_CROP_TOP = 240
CIRCLE_CENTER = (1080, 540)
CIRCLE_RADIUS = 300
TICKER_RECT = (1608, 564, 2152, 788)
AACHEN_CELL = (235, 1116)

ACCENT = (122, 192, 255)
ACCENT_WARM = (255, 178, 102)
WHITE = (245, 248, 252)
DIM = (168, 178, 195)

FONT_TITLE = "title"
FONT_HEAD = "head"
FONT_BODY = "body"
FONT_SMALL = "small"

MAX_FRAME_STEP = 10000000


def step_of(i):
    return min(i * (i + 1), MAX_FRAME_STEP)


def smooth(t):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def win(f, start, end, fi=15, fo=15):
    """Alpha of an overlay active on local frames [start, end)."""
    if f < start or f >= end:
        return 0.0
    a = 1.0
    if f < start + fi:
        a = min(a, (f - start) / float(fi))
    if f > end - fo:
        a = min(a, (end - f) / float(fo))
    return max(0.0, min(1.0, a))


def card(x, y, w, lines, alpha, bg=(10, 14, 22, 215), pad=26, radius=18,
         accent=None, shadow=True, align="left"):
    return {
        "kind": "card", "x": x, "y": y, "w": w, "lines": lines,
        "alpha": alpha, "bg": bg, "pad": pad, "radius": radius,
        "accent": accent, "shadow": shadow, "align": align,
    }


def title_card(x, y, w, lines, alpha):
    return card(x, y, w, lines, alpha, bg=(8, 11, 18, 190), pad=40,
                radius=26, accent=ACCENT)


def beat_card(x, y, w, title, body_lines, alpha):
    lines = [(title, FONT_HEAD, 38, ACCENT), (None, None, 12, None)]
    lines += [(t, FONT_BODY, 30, WHITE) for t in body_lines]
    return card(x, y, w, lines, alpha, bg=(9, 12, 19, 225), accent=ACCENT)


def arrow(p1, p2, alpha, color=ACCENT, width=7):
    return {"kind": "arrow", "p1": p1, "p2": p2, "alpha": alpha,
            "color": color, "width": width}


def ring(center, r, alpha, width=9, color=ACCENT):
    return {"kind": "ring", "center": center, "r": r, "alpha": alpha,
            "width": width, "color": color}


def dot(p, alpha, r=11, color=ACCENT):
    return {"kind": "dot", "p": p, "alpha": alpha, "r": r, "color": color}


def scrim(alpha, color=(0, 0, 0)):
    return {"kind": "scrim", "alpha": alpha, "color": color}


class Timeline:
    def __init__(self):
        self.frames = []
        self.events = {}
        self.cues = []

    def _append(self, view, src, n, zoom=None, pip=None, pip_src=None,
                pip_ctop=None, pip_alpha=None, hud=None, drawables=None,
                bg_blur=False, ticker=False, cmp=None):
        for k in range(n):
            s = int(src(k)) if callable(src) else int(src)
            entry = {
                "view": view,
                "src": s,
                "zoom": None,
                "pip": pip(k) if callable(pip) else None,
                "pip_src": None,
                "pip_ctop": None,
                "pip_alpha": 1.0,
                "pip_tag": 0.0,
                "hud": hud(k) if callable(hud) else hud,
                "bg_blur": bg_blur,
                "ticker": ticker,
                "drawables": [],
                "cmp_src": None,
                "cmp_x": None,
            }
            if cmp is not None:
                csrc, cx = cmp
                entry["cmp_src"] = int(csrc)
                entry["cmp_x"] = int(cx(k)) if callable(cx) else int(cx)
            if zoom is not None:
                entry["zoom"] = zoom(k) if callable(zoom) else zoom
            if pip is not None:
                rect = entry["pip"]
                if rect is not None:
                    entry["pip_src"] = pip_src(k) if callable(pip_src) else (pip_src if pip_src is not None else s)
                    entry["pip_ctop"] = pip_ctop(k) if callable(pip_ctop) else (pip_ctop if pip_ctop is not None else PIP_CROP_TOP)
                    entry["pip_alpha"] = pip_alpha(k) if callable(pip_alpha) else (pip_alpha if pip_alpha is not None else 1.0)
                    dw = rect[2]
                    entry["pip_tag"] = max(0.0, (W - dw) / float(W - PIP_DST[2]))
            if drawables:
                for d in drawables:
                    a = d["alpha"](k) if callable(d["alpha"]) else d["alpha"]
                    if a > 0.0:
                        nd = dict(d)
                        nd["alpha"] = a
                        entry["drawables"].append(nd)
            self.frames.append(entry)

    def fade_from_black(self, n, view, src):
        self._append(view, src, n, drawables=[
            {"kind": "scrim",
             "alpha": (lambda k: 1.0 - smooth(k / float(max(1, n - 1)))),
             "color": (0, 0, 0)}])

    def still(self, view, src, n, zoom=None, drawables=None, hud=None,
              pip=None, pip_src=None, pip_ctop=None, pip_alpha=None,
              ticker=False):
        self._append(view, src, n, zoom=zoom, drawables=drawables, hud=hud,
                     pip=pip, pip_src=pip_src, pip_ctop=pip_ctop,
                     pip_alpha=pip_alpha, ticker=ticker)

    def play(self, view, a, b, n, drawables=None, hud=None, pip=None,
             pip_src=None, pip_ctop=None, pip_alpha=None, zoom=None,
             ticker=False):
        def src(k):
            if n <= 1:
                return a
            return int(round(a + (b - a) * (k / float(n - 1))))
        self._append(view, src, n, drawables=drawables, hud=hud, pip=pip,
                     pip_src=pip_src, pip_ctop=pip_ctop, pip_alpha=pip_alpha,
                     zoom=zoom, ticker=ticker)

    def freeze(self, view, src, n, drawables=None, hud=None, pip=None,
               pip_src=None, pip_ctop=None, pip_alpha=None, ticker=False):
        self._append(view, src, n, drawables=drawables, hud=hud, pip=pip,
                     pip_src=pip_src, pip_ctop=pip_ctop, pip_alpha=pip_alpha,
                     ticker=ticker)

    def zoom_pip(self, src, n, rect_a, rect_b, ctop_a, ctop_b, view="top",
                 drawables=None, hud=None, ticker=False):
        def pip(k):
            t = smooth(k / float(max(1, n - 1)))
            return tuple(int(round(a + (b - a) * t)) for a, b in
                         zip(rect_a, rect_b))

        def ctop(k):
            t = smooth(k / float(max(1, n - 1)))
            return int(round(ctop_a + (ctop_b - ctop_a) * t))
        self._append(view, src, n, pip=pip, pip_src=src, pip_ctop=ctop,
                     drawables=drawables, hud=hud, ticker=ticker)


def hud_top(k):
    return {"show": True, "view_label": "TOP VIEW — WATER DEPTH OVER TOPOGRAPHY"}


def hud_slice(k):
    return {"show": True, "view_label": "EQUATORIAL CROSS-SECTION (z = 540)"}


PIP_KEEP = dict(
    pip=lambda k: PIP_DST,
    pip_ctop=PIP_CROP_TOP,
    pip_alpha=None,
)

PIP_ON_19 = dict(
    pip=lambda k: PIP_DST,
    pip_src=19,
    pip_ctop=PIP_CROP_TOP,
    pip_alpha=1.0,
)


def build():
    T = Timeline()

    title_lines = [
        ("Terrain Solver", FONT_TITLE, 92, WHITE),
        (None, None, 22, None),
        ("Example execution of releasing a water torrent",
         FONT_BODY, 33, (225, 232, 244)),
        ("on a dry, equirectangular map of Earth",
         FONT_BODY, 33, (225, 232, 244)),
        (None, None, 14, None),
        ("A cellular-automaton shallow-water solver — test case for the CPP-ML-Interface",
         FONT_SMALL, 26, DIM),
        ("M.Sc. Thesis · RWTH Aachen · RWTH CLAIX-2023 · 24 ranks on c23ms",
         FONT_SMALL, 26, DIM),
    ]
    param_lines = [
        ("Initial-run parameters", FONT_HEAD, 34, ACCENT_WARM),
        (None, None, 10, None),
        ("Grid: 2160 × 1080 cells — dry equirectangular map of Earth topography", FONT_BODY, 28, WHITE),
        ("Initial condition: cylindrical water column — radius r = 300, depth d = 300", FONT_BODY, 28, WHITE),
        ("Evolution: 10,000,000 steps — snapshots on a triangular schedule (3,163 frames)", FONT_BODY, 28, WHITE),
        ("Views: top-down water depth + equatorial cross-section (z = 540)", FONT_BODY, 28, WHITE),
        ("Wall time: 3 h 40 m (≈ 1.3 ms per step) on 24 CPU ranks", FONT_BODY, 28, WHITE),
    ]

    # -------- Act 0: intro on a frozen first frame (static, no push-in) --------
    n_fade = 18
    T.fade_from_black(n_fade, "top", 0)

    intro_n = 280
    T._append(
        "top", 0, intro_n,
        drawables=[
            title_card(120, 118, 1430, title_lines,
                       lambda k: win(k, 16, 250, 18, 20)),
            card(790, 736, 1290, param_lines,
                 lambda k: win(k, 96, 278, 16, 16), accent=ACCENT_WARM),
        ],
        hud=hud_top,
    )
    T.events["intro"] = len(T.frames)

    # -------- Act 1: collapse and early spread (single pause) --------
    pip_in = dict(
        pip=lambda k: PIP_DST,
        pip_src=lambda k: min(6, k),
        pip_ctop=PIP_CROP_TOP,
        pip_alpha=lambda k: min(1.0, (k + 1) / 16.0),
    )

    T.play("top", 0, 19, 20, hud=hud_top, ticker=True, **pip_in)

    collapse_card = beat_card(
        90, 340, 900, "Collapse",
        ["The rim gives way first: the column spreads radially,",
         "while the front piles up to depth 360 — a fifth above 300."],
        lambda k: win(k, 6, 154, 14, 16))
    ringA = ring(CIRCLE_CENTER, 322, lambda k: win(k, 6, 154, 14, 16))
    arrowsA = [arrow((CIRCLE_CENTER[0] + 330 * dx, CIRCLE_CENTER[1] + 330 * dy),
                     (CIRCLE_CENTER[0] + 470 * dx, CIRCLE_CENTER[1] + 470 * dy),
                     lambda k: win(k, 6, 154, 14, 16))
               for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0),
                              (0.72, -0.72), (-0.72, 0.72),
                              (0.72, 0.72), (-0.72, -0.72)]]
    ringB = ring(CIRCLE_CENTER, 360, lambda k: win(k, 6, 154, 14, 16),
                 width=7, color=ACCENT_WARM)
    T.events["collapse"] = len(T.frames)
    T.freeze("top", 19, 160, hud=hud_top, ticker=True, **PIP_ON_19,
             drawables=[collapse_card, ringA, ringB] + arrowsA)

    T.play("top", 19, 110, 91, hud=hud_top, ticker=True, **pip_in)

    # -------- Zoom 1: the cross-section --------
    zoom_n = 44
    T.events["zoom1_in"] = len(T.frames)
    T.zoom_pip(110, zoom_n, PIP_DST, (0, 0, W, H), PIP_CROP_TOP, 0,
               view="top", hud=hud_slice, ticker=True)

    dome_card = beat_card(
        90, 84, 1020, "From column to dome",
        ["The column has spread into a broad dome over the basin;",
         "water ponds deepest in the trench between the plateaus."],
        lambda k: win(k, 14, 164, 14, 16))
    dotL = dot((700, 938), lambda k: win(k, 14, 164, 14, 16), r=12,
               color=ACCENT_WARM)
    dotR = dot((1220, 930), lambda k: win(k, 14, 164, 14, 16), r=12,
               color=ACCENT_WARM)
    dotT = dot((1060, 1015), lambda k: win(k, 14, 164, 14, 16), r=12,
               color=ACCENT_WARM)
    arrL = arrow((700, 840), (700, 916), lambda k: win(k, 14, 164, 14, 16),
                 color=ACCENT_WARM)
    arrR = arrow((1220, 830), (1220, 904), lambda k: win(k, 14, 164, 14, 16),
                 color=ACCENT_WARM)
    arrT = arrow((1060, 860), (1060, 990), lambda k: win(k, 14, 164, 14, 16),
                 color=ACCENT_WARM)
    T.events["beatC"] = len(T.frames)
    T.play("slice", 110, 240, 166, hud=hud_slice, ticker=True,
           drawables=[dome_card, dotL, dotR, dotT, arrL, arrR, arrT])

    T.events["zoom1_out"] = len(T.frames)
    T.zoom_pip(240, zoom_n, (0, 0, W, H), PIP_DST, 0, PIP_CROP_TOP,
               view="top", hud=hud_top, ticker=True)

    # -------- Act 2: Panama and the two Atlantic paths --------
    panama_card = beat_card(
        90, 84, 1020, "Around Panama",
        ["The flood crosses the eastern Pacific, rounds",
         "the Isthmus of Panama and fills the Caribbean."],
        lambda k: win(k, 12, 168, 14, 16))
    dotD = dot((602, 485), lambda k: win(k, 12, 168, 14, 16), r=13,
               color=ACCENT_WARM)
    T.events["beatD"] = len(T.frames)
    T.play("top", 240, 380, 170, hud=hud_top, ticker=True, **PIP_KEEP,
           drawables=[panama_card, dotD])

    capehorn_card = beat_card(
        90, 84, 1020, "Atlantic breached",
        ["Two paths — east through the Caribbean, and",
         "south around Cape Horn into the Atlantic."],
        lambda k: win(k, 12, 168, 14, 16))
    arrE_carib = arrow((612, 456), (700, 450),
                       lambda k: win(k, 12, 168, 14, 16), color=ACCENT_WARM)
    dotE_horn = dot((676, 875), lambda k: win(k, 12, 168, 14, 16), r=13,
                    color=ACCENT_WARM)
    arrE_horn = arrow((608, 928), (710, 886),
                      lambda k: win(k, 12, 168, 14, 16), color=ACCENT_WARM)
    T.events["beatE"] = len(T.frames)
    T.play("top", 380, 470, 170, hud=hud_top, ticker=True, **PIP_KEEP,
           drawables=[capehorn_card, arrE_carib, dotE_horn, arrE_horn])

    # -------- Zoom 2: reflections in the cross-section --------
    T.events["zoom2_in"] = len(T.frames)
    T.zoom_pip(470, zoom_n, PIP_DST, (0, 0, W, H), PIP_CROP_TOP, 0,
               view="top", hud=hud_slice, ticker=True)

    refl_card = beat_card(
        90, 84, 1020, "Reflections decay",
        ["Repeated reflections off basins and ridges fade away",
         "as the worldwide film levels out (peak depth < 200)."],
        lambda k: win(k, 14, 164, 14, 16))
    T.events["beatF"] = len(T.frames)
    T.play("slice", 470, 610, 170, hud=hud_slice, ticker=True,
           drawables=[refl_card])

    T.events["zoom2_out"] = len(T.frames)
    T.zoom_pip(610, zoom_n, (0, 0, W, H), PIP_DST, 0, PIP_CROP_TOP,
               view="top", hud=hud_top, ticker=True)

    # -------- Act 3: levelling and the long drain --------
    levelling_card = beat_card(
        90, 84, 1020, "Levelling",
        ["Shorelines stop reverberating —",
         "peak depth falls below 180."],
        lambda k: win(k, 14, 180, 14, 16))
    quasi_card = beat_card(
        90, 84, 1020, "Quasi-steady state",
        ["The flow relaxes into a thin, near-uniform film;",
         "at Aachen the column is down to ≈ 30 and still draining."],
        lambda k: win(k, 190, 332, 14, 16))
    T.events["beatG"] = len(T.frames)
    T.play("top", 610, 860, 340, hud=hud_top, ticker=True, **PIP_KEEP,
           drawables=[levelling_card, quasi_card])
    T.events["beatH"] = T.events["beatG"] + 190

    T.play("top", 860, 1000, 240, hud=hud_top, ticker=True, **PIP_KEEP)

    # -------- Act 4: the long drain to ten million steps --------
    drain_card = beat_card(
        90, 84, 1020, "Slow drain",
        ["The film keeps thinning as water settles into the lows —",
         "Aachen drops from ≈ 20 to ≈ 5 by step 2.5 million."],
        lambda k: win(k, 14, 285, 14, 24))
    T.events["ext1"] = len(T.frames)
    T.play("top", 1000, 1600, 300, hud=hud_top, ticker=True, **PIP_KEEP,
           drawables=[drain_card])

    dry_card = beat_card(
        90, 84, 1020, "Aachen runs dry",
        ["Near step 4.1 million the tracked column at Aachen",
         "reaches zero — the film drains off the last pools."],
        lambda k: win(k, 12, 230, 14, 24))
    T.events["ext2"] = len(T.frames)
    T.play("top", 1600, 2400, 300, hud=hud_top, ticker=True, **PIP_KEEP,
           drawables=[dry_card])

    def pip_out(k):
        return max(0.0, min(1.0, (239 - k) / 30.0))

    T.play("top", 2400, 3162, 240, hud=hud_top, ticker=True,
           pip=lambda k: PIP_DST, pip_ctop=PIP_CROP_TOP,
           pip_alpha=pip_out)

    for key, off in [("collapse", 0.4), ("beatC", 0.5), ("beatD", 0.4),
                     ("beatE", 0.4), ("beatF", 0.5), ("beatG", 0.5),
                     ("beatH", 0.4), ("ext1", 0.5), ("ext2", 0.4)]:
        T.cues.append(((T.events[key] + int(round(off * FPS))) / FPS, "beat"))

    # -------- Finale: freeze, then step 0 → step 10M comparison wipe --------
    hold_n = 40
    cmp_n = 230
    sweep_n = 150
    out_n = 58
    T.freeze("top", 3162, hold_n, hud=hud_top, ticker=True, drawables=[])
    T.events["compare"] = len(T.frames)
    T.cues.append((T.events["compare"] / FPS, "finale"))

    def cmp_x(k):
        return int(round(W - 3 - (W - 6) * smooth(k / float(sweep_n))))

    chip0 = card(90, 84, 560,
                 [("STEP 0 — INITIAL WATER COLUMN", FONT_SMALL, 27, WHITE)],
                 lambda k: win(k, 6, 110, 10, 30), pad=16, radius=12)
    chip1 = card(990, 84, 560,
                 [("STEP 10,000,000 — EQUILIBRIUM FILM", FONT_SMALL, 27, WHITE)],
                 lambda k: win(k, 60, cmp_n, 24, 26), pad=16, radius=12)
    caption = card(405, 968, 1350, [
        ("Peak depth 360 → 137 (equilibrium film) · mass drift +2.5 × 10⁻³",
         FONT_SMALL, 27, (225, 233, 246)),
        ("10,000,000 steps · 3 h 40 m on 24 CPU ranks · 3,163 snapshots",
         FONT_SMALL, 27, (225, 233, 246)),
    ], lambda k: win(k, 148, cmp_n, 16, 0), pad=18, radius=12, align="center")
    T._append("top", 3162, cmp_n, hud=hud_top, ticker=True, cmp=(0, cmp_x),
              drawables=[chip0, chip1, caption])
    cap_out = dict(caption)
    cap_out["alpha"] = 1.0
    T._append("top", 3162, out_n, hud=hud_top, ticker=True, drawables=[
        cap_out,
        scrim(lambda k: smooth(k / float(out_n - 3))),
    ])

    return T


_CACHE = {}


def build_once():
    if "timeline" not in _CACHE:
        _CACHE["timeline"] = build()
    return _CACHE["timeline"]


def frames():
    return build_once().frames


def total_frames():
    return len(frames())


def total_seconds():
    return total_frames() / float(FPS)


def events():
    return build_once().events


def cues():
    return build_once().cues


CHAPTERS = [19, 110, 240, 380, 470, 610, 860, 1000, 1600, 2400]


def still_points():
    ev = events()
    return {
        "00_intro_title": 120,
        "01_intro_params": 218,
        "02_play_start": ev["intro"] + 10,
        "03_collapse": ev["collapse"] + 40,
        "04_zoom1_mid": ev["zoom1_in"] + 22,
        "05_dome": ev["beatC"] + 140,
        "06_panama": ev["beatD"] + 70,
        "07_capehorn": ev["beatE"] + 70,
        "08_zoom2_mid": ev["zoom2_in"] + 22,
        "09_reflections": ev["beatF"] + 70,
        "10_levelling": ev["beatG"] + 70,
        "11_quasi": ev["beatH"] + 70,
        "12_final_run": ev["beatH"] + 270,
        "12b_slowdrain": ev["ext1"] + 120,
        "12c_dry": ev["ext2"] + 190,
        "13_compare": ev["compare"] + 78,
        "14_final_hold": total_frames() - 46,
    }
