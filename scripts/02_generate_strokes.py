"""Stroke Planner for Admiral Yi Sun-sin Portrait (v4 — 9-Stage Artist Pipeline).

================================================================================
STUDY OF REFERENCE.PNG (DRIVES ALL STAGE ALLOCATION, FLOW & VERIFICATION)
================================================================================
1. Subject and Composition
   - Traditional frontal three-quarter-length Joseon official portrait of Admiral
     Yi Sun-sin (충무공 이순신 표준영정, 1280x1932, aspect ratio 0.662526) centered
     symmetrically against an aged golden-ochre silk scroll background.
   - Major mass placement (normalized x, y):
     * Samo official hat (사모): tall rounded two-tier dome (x 0.37–0.62, y 0.03–0.21)
       with symmetrical rounded-rectangular side wings (양각, x 0.26–0.73, y 0.12–0.19).
     * Head, face & ears: centered (x 0.37–0.63, y 0.18–0.36); long flowing beard
       descending over neck and collar (x 0.37–0.62, y 0.29–0.46).
     * Crimson official robe (홍단령): fills lower two-thirds (x 0.04–0.99, y 0.34–1.00)
       with broad crossed sleeves resting horizontally across the waist (y 0.70–0.86).
     * Twin-tiger rank badge (쌍호흉배): centered chest square (x 0.33–0.68, y 0.48–0.68),
       anchored by curved official belt (각대, x 0.30–0.71, y 0.65–0.73) and pale ivory
       hanging sash emerging below (x 0.45–0.55, y 0.70–1.00).
     * Leopard pelt drape (표피): feathery spotted fur wings behind left (x 0.00–0.23)
       and right (x 0.80–1.00) shoulders (y 0.40–0.82).
   - Lighting: soft, even upper-frontal traditional portrait illumination; gentle warmth
     on top silk scroll, upper forehead, nose bridge, cheekbones, and top hat crown,
     deepening softly toward lower edges and under crossed sleeve folds.

2. Recognisable Features (7 Key Landmarks — Priority for Extra Passes & Finest Brushes)
   - [F1] Piercing calm eyes & arched brows: almond horizontal eyes (left x~0.445,
     right x~0.542, y~0.236) with crisp dark ink upper eyelid contours flickering
     cleanly outward, delicate double-lid creases, dark espresso irises with subtle
     warm amber catchlights, warm-ivory sclera, and softly feathered arched brows.
   - [F2] Sculpted nose, solemn lips & drooping mustache: luminous vertical nose bridge,
     delicate alar shadows, calm muted terracotta-rose lips with firm closed line, full
     dark charcoal-umber mustache curving downward and small chin soul-patch tuft.
   - [F3] Long flowing vertical se-pil beard & wispy sideburns: hundreds of crisp 1px
     vertical dark hairs layered over pale neck skin and deep crimson collar, allowing
     bright warm skin/crimson gaps to show through between strands.
   - [F4] Two-tier Samo hat & swirling wood-grain gauze wings (양각): deep matte umber
     crown with subtle tier step and dark forehead rim; translucent ochre wings with
     crisp dark borders and swirling horizontal lacquer-grain curves.
   - [F5] Twin-tiger embroidered rank badge (쌍호흉배): deep olive-charcoal ground with
     two symmetrical facing pale-ivory tigers/leopards covered in tiny dark ringed spots,
     whiskers, alert eyes, curved tails, central crimson lotus blossom, curling white sea
     waves, stylized central peak, and layered multicolor lower wave bands.
   - [F6] Official belt (각대) & ivory sash: curved dark woven-lattice belt with gleaming
     pale-gold framed black lacquer plaques (center rectangle, twin bars, outer circles)
     above the pale warm-ivory descending cloth sash.
   - [F7] Crimson robe ink drapery lines (의습선), subtle cloud brocade (운보문) & leopard
     pelt (표피): sweeping unbroken dark-burgundy/black fold creases across the collar,
     shoulder knot tab, and crossed sleeves; subtle tone-on-tone brocade clouds; outward-
     radiating feathery leopard fur and bold dark crescent rosettes behind the shoulders.

3. Color & Value Character
   - Warm, unified mineral-pigment and aged-silk temperature throughout (no cold blues or
     stark pure whites).
   - Dominant palette: warm golden-ochre aged silk (#EBD09B top -> #C48F5C mid-lower),
     luminous warm golden-peach skin (#DBA973 with warm sienna shading), deep carmine-
     burgundy silk (#802634; highlights #A03744, creases #370E18), rich warm umber/espresso
     darks (#24140D–#5D2F18 — deep but never flat dead black), olive-charcoal embroidery
     (#3C442D), and clean antique-ivory highlights (#E6DEC6 — clean, never blown out).
   - Surface textures: serene flat aged-silk background (kept loose & smooth, never noisy),
     soft barim facial transitions, crisp fine se-pil ink lines and individual hair/fur
     strokes following exact anatomical/drapery/growth directions.
================================================================================
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "reference.png"
VIS_DIR = ROOT / "visualizations"
INT_DIR = ROOT / "intermediate_results"
VIS_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = [
    "Underpainting",
    "Block-in",
    "Big shapes",
    "Forms",
    "Modeling",
    "Refining",
    "Details",
    "Fine details",
    "Final touches",
]


# ---------------------------------------------------------------------------
# Numba-accelerated exact brush mask rasterizers (matching app.js 1:1)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _det_hash(seed: int, k: int) -> float:
    """Deterministic pseudo-random float in [-1.0, 1.0] matching JS Math.imul."""
    x = ((seed * 1664525) + (k * 1013904223) + 374761393) & 0x7FFFFFFF
    x = (x ^ (x >> 13)) & 0x7FFFFFFF
    x = ((x * 48271) + 12345) & 0x7FFFFFFF
    x = (x ^ (x >> 16)) & 0x7FFFFFFF
    return (float(x & 0xFFFF) / 32767.5) - 1.0


@njit(cache=True)
def _stamp_radial_dab(
    trans: np.ndarray,
    bx0: int,
    by0: int,
    px: float,
    py: float,
    r: float,
    dab_alpha: float,
) -> None:
    """Multiply remaining transparency map by (1 - radial_alpha(d/r))."""
    bh, bw = trans.shape
    r_ceil = int(math.ceil(r)) + 1
    ix0 = max(0, int(math.floor(px)) - r_ceil - bx0)
    ix1 = min(bw - 1, int(math.ceil(px)) + r_ceil - bx0)
    iy0 = max(0, int(math.floor(py)) - r_ceil - by0)
    iy1 = min(bh - 1, int(math.ceil(py)) + r_ceil - by0)
    inv_r = 1.0 / max(r, 1e-4)
    r2 = r * r

    for iy in range(iy0, iy1 + 1):
        dy = (iy + by0 + 0.5) - py
        dy2 = dy * dy
        for ix in range(ix0, ix1 + 1):
            dx = (ix + bx0 + 0.5) - px
            d2 = dx * dx + dy2
            if d2 < r2:
                u = math.sqrt(d2) * inv_r
                if u <= 0.55:
                    f = 1.0 - 0.28 * (u / 0.55)
                else:
                    f = 0.72 * (1.0 - (u - 0.55) / 0.45)
                a = dab_alpha * f
                trans[iy, ix] *= (1.0 - a)


@njit(cache=True)
def _rasterize_bezier_stroke(
    trans: np.ndarray,
    bx0: int,
    by0: int,
    x0: float,
    y0: float,
    cx: float,
    cy: float,
    x1: float,
    y1: float,
    half_w: float,
    alpha: float,
) -> None:
    """Rasterize a round-capped quadratic Bezier stroke into transparency buffer."""
    bh, bw = trans.shape
    n_seg = 6
    pts_x = np.empty(n_seg + 1, dtype=np.float32)
    pts_y = np.empty(n_seg + 1, dtype=np.float32)
    for i in range(n_seg + 1):
        t = i / float(n_seg)
        mt = 1.0 - t
        pts_x[i] = mt * mt * x0 + 2.0 * mt * t * cx + t * t * x1
        pts_y[i] = mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1

    pad = half_w + 1.2
    min_x = pts_x[0]
    max_x = pts_x[0]
    min_y = pts_y[0]
    max_y = pts_y[0]
    for i in range(1, n_seg + 1):
        if pts_x[i] < min_x:
            min_x = pts_x[i]
        if pts_x[i] > max_x:
            max_x = pts_x[i]
        if pts_y[i] < min_y:
            min_y = pts_y[i]
        if pts_y[i] > max_y:
            max_y = pts_y[i]

    ix0 = max(0, int(math.floor(min_x - pad)) - bx0)
    ix1 = min(bw - 1, int(math.ceil(max_x + pad)) - bx0)
    iy0 = max(0, int(math.floor(min_y - pad)) - by0)
    iy1 = min(bh - 1, int(math.ceil(max_y + pad)) - by0)

    vx = np.empty(n_seg, dtype=np.float32)
    vy = np.empty(n_seg, dtype=np.float32)
    inv_l2 = np.empty(n_seg, dtype=np.float32)
    for s in range(n_seg):
        dx = pts_x[s + 1] - pts_x[s]
        dy = pts_y[s + 1] - pts_y[s]
        vx[s] = dx
        vy[s] = dy
        l2 = dx * dx + dy * dy
        inv_l2[s] = 1.0 / l2 if l2 > 1e-8 else 0.0

    outer = half_w + 0.65
    inner = max(0.0, half_w - 0.55)
    inv_aa = 1.0 / max(outer - inner, 0.5)
    outer2 = outer * outer

    for iy in range(iy0, iy1 + 1):
        py = iy + by0 + 0.5
        for ix in range(ix0, ix1 + 1):
            px = ix + bx0 + 0.5
            min_d2 = 1e12
            for s in range(n_seg):
                wx = px - pts_x[s]
                wy = py - pts_y[s]
                il2 = inv_l2[s]
                if il2 > 0.0:
                    proj = (wx * vx[s] + wy * vy[s]) * il2
                    if proj < 0.0:
                        proj = 0.0
                    elif proj > 1.0:
                        proj = 1.0
                    dx = wx - proj * vx[s]
                    dy = wy - proj * vy[s]
                else:
                    dx = wx
                    dy = wy
                d2 = dx * dx + dy * dy
                if d2 < min_d2:
                    min_d2 = d2
            if min_d2 < outer2:
                d = math.sqrt(min_d2)
                if d <= inner:
                    cov = 1.0
                else:
                    cov = (outer - d) * inv_aa
                    if cov < 0.0:
                        cov = 0.0
                    elif cov > 1.0:
                        cov = 1.0
                if half_w < 0.5:
                    cov *= (half_w * 2.0)
                trans[iy, ix] *= (1.0 - alpha * cov)


@njit(cache=True)
def build_stroke_mask(
    btype: int,
    x0: float,
    y0: float,
    cx: float,
    cy: float,
    x1: float,
    y1: float,
    r: float,
    opacity: float,
    seed: int,
    h: int,
    w: int,
):
    """Build exact effective alpha mask M_eff and its bounding box (bx0, by0)."""
    pad = r * 1.45 + 3.0
    min_x = min(x0, min(cx, x1)) - pad
    max_x = max(x0, max(cx, x1)) + pad
    min_y = min(y0, min(cy, y1)) - pad
    max_y = max(y0, max(cy, y1)) + pad

    bx0 = max(0, int(math.floor(min_x)))
    bx1 = min(w - 1, int(math.ceil(max_x)))
    by0 = max(0, int(math.floor(min_y)))
    by1 = min(h - 1, int(math.ceil(max_y)))
    bw = bx1 - bx0 + 1
    bh = by1 - by0 + 1
    if bw <= 0 or bh <= 0:
        return np.zeros((1, 1), dtype=np.float32), 0, 0, False

    trans = np.ones((bh, bw), dtype=np.float32)

    if btype == 0:
        approx_len = math.hypot(cx - x0, cy - y0) + math.hypot(x1 - cx, y1 - cy)
        step = max(2.5, r * 0.55)
        n_dabs = int(round(approx_len / step)) + 1
        if n_dabs < 1:
            n_dabs = 1
        elif n_dabs > 10:
            n_dabs = 10
        if n_dabs == 1:
            dab_alpha = opacity
        else:
            dab_alpha = 1.0 - math.pow(max(1e-4, 1.0 - opacity), 0.52)

        for i in range(n_dabs):
            t = 0.5 if n_dabs == 1 else i / float(n_dabs - 1)
            mt = 1.0 - t
            px = mt * mt * x0 + 2.0 * mt * t * cx + t * t * x1
            py = mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1
            _stamp_radial_dab(trans, bx0, by0, px, py, r, dab_alpha)

    elif btype == 1:
        dx = x1 - x0
        dy = y1 - y0
        dlen = math.hypot(dx, dy)
        if dlen < 1e-4:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -dy / dlen, dx / dlen

        _rasterize_bezier_stroke(trans, bx0, by0, x0, y0, cx, cy, x1, y1, r * 0.92, opacity * 0.34)

        offsets = (-0.68, -0.34, 0.0, 0.34, 0.68)
        weights = (0.38, 0.56, 0.65, 0.54, 0.36)
        fil_hw = max(0.5, r * 0.22)
        for i in range(5):
            j_off = _det_hash(seed, i * 3 + 1) * 0.11
            j_end = _det_hash(seed, i * 3 + 2) * 0.08 * r
            off = (offsets[i] + j_off) * r
            end_off = off * 0.78
            inv_d = 1.0 / dlen if dlen > 1.0 else 1.0
            ux = dx * inv_d * j_end
            uy = dy * inv_d * j_end
            _rasterize_bezier_stroke(
                trans,
                bx0,
                by0,
                x0 + nx * end_off + ux,
                y0 + ny * end_off + uy,
                cx + nx * off,
                cy + ny * off,
                x1 + nx * end_off - ux,
                y1 + ny * end_off - uy,
                fil_hw,
                opacity * weights[i],
            )
    elif btype == 2:
        _rasterize_bezier_stroke(trans, bx0, by0, x0, y0, cx, cy, x1, y1, r * 1.05, opacity * 0.38)
        _rasterize_bezier_stroke(trans, bx0, by0, x0, y0, cx, cy, x1, y1, r * 0.62, opacity * 0.78)
    else:
        _rasterize_bezier_stroke(trans, bx0, by0, x0, y0, cx, cy, x1, y1, max(0.375, r * 0.74), opacity)

    mask = np.float32(1.0) - trans
    return mask, bx0, by0, True


@njit(cache=True)
def evaluate_and_maybe_commit(
    canvas: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    btype: int,
    x0: float,
    y0: float,
    cx: float,
    cy: float,
    x1: float,
    y1: float,
    r: float,
    opacity: float,
    seed: int,
    max_color_dev: float,
    min_gain: float,
    force_commit: bool,
):
    """Compute exact optimal brush color S*, evaluate weighted L2 error gain, and commit if beneficial."""
    h, w, _ = canvas.shape
    mask, bx0, by0, ok = build_stroke_mask(btype, x0, y0, cx, cy, x1, y1, r, opacity, seed, h, w)
    if not ok:
        return False, 0.0, 0.0, 0.0, 0.0

    bh, bw = mask.shape
    num_r = 0.0
    num_g = 0.0
    num_b = 0.0
    den = 0.0
    tm_r = 0.0
    tm_g = 0.0
    tm_b = 0.0
    wsum = 0.0

    for iy in range(bh):
        y = by0 + iy
        for ix in range(bw):
            m = mask[iy, ix]
            if m < 1e-4:
                continue
            x = bx0 + ix
            wt = weight[y, x]
            wm = wt * m
            wm2 = wm * m
            om = 1.0 - m

            tr = target[y, x, 0]
            tg = target[y, x, 1]
            tb = target[y, x, 2]
            cr = canvas[y, x, 0]
            cg = canvas[y, x, 1]
            cb = canvas[y, x, 2]

            num_r += wm * (tr - om * cr)
            num_g += wm * (tg - om * cg)
            num_b += wm * (tb - om * cb)
            den += wm2

            tm_r += wm * tr
            tm_g += wm * tg
            tm_b += wm * tb
            wsum += wm

    if den < 1e-5 or wsum < 1e-5:
        return False, 0.0, 0.0, 0.0, 0.0

    mean_r = tm_r / wsum
    mean_g = tm_g / wsum
    mean_b = tm_b / wsum

    sr = num_r / den
    sg = num_g / den
    sb = num_b / den

    lo_r = max(0.0, mean_r - max_color_dev)
    hi_r = min(255.0, mean_r + max_color_dev)
    lo_g = max(0.0, mean_g - max_color_dev)
    hi_g = min(255.0, mean_g + max_color_dev)
    lo_b = max(0.0, mean_b - max_color_dev)
    hi_b = min(255.0, mean_b + max_color_dev)

    sr = min(hi_r, max(lo_r, sr))
    sg = min(hi_g, max(lo_g, sg))
    sb = min(hi_b, max(lo_b, sb))

    ir = float(int(round(sr)))
    ig = float(int(round(sg)))
    ib = float(int(round(sb)))

    err_before = 0.0
    err_after = 0.0
    for iy in range(bh):
        y = by0 + iy
        for ix in range(bw):
            m = mask[iy, ix]
            if m < 1e-4:
                continue
            x = bx0 + ix
            wt = weight[y, x]
            om = 1.0 - m

            tr = target[y, x, 0]
            tg = target[y, x, 1]
            tb = target[y, x, 2]
            cr = canvas[y, x, 0]
            cg = canvas[y, x, 1]
            cb = canvas[y, x, 2]

            dr0 = cr - tr
            dg0 = cg - tg
            db0 = cb - tb
            err_before += wt * (dr0 * dr0 + dg0 * dg0 + db0 * db0)

            nr = om * cr + m * ir
            ng = om * cg + m * ig
            nb = om * cb + m * ib
            dr1 = nr - tr
            dg1 = ng - tg
            db1 = nb - tb
            err_after += wt * (dr1 * dr1 + dg1 * dg1 + db1 * db1)

    gain = err_before - err_after
    if not force_commit and gain < min_gain:
        return False, 0.0, 0.0, 0.0, gain

    for iy in range(bh):
        y = by0 + iy
        for ix in range(bw):
            m = mask[iy, ix]
            if m < 1e-4:
                continue
            x = bx0 + ix
            om = 1.0 - m
            canvas[y, x, 0] = om * canvas[y, x, 0] + m * ir
            canvas[y, x, 1] = om * canvas[y, x, 1] + m * ig
            canvas[y, x, 2] = om * canvas[y, x, 2] + m * ib

    return True, ir, ig, ib, gain


@njit(cache=True)
def trace_streamline(
    sx: float,
    sy: float,
    tx_field: np.ndarray,
    ty_field: np.ndarray,
    ref_smooth: np.ndarray,
    step_len: float,
    max_half_steps: int,
    color_thresh: float,
    angle_bias_x: float,
    angle_bias_y: float,
    bias_strength: float,
):
    """Trace a curved brush path forward and backward along tangent flow, stopping at color edges."""
    h, w, _ = ref_smooth.shape
    ix_s = min(w - 1, max(0, int(sx)))
    iy_s = min(h - 1, max(0, int(sy)))
    r0 = ref_smooth[iy_s, ix_s, 0]
    g0 = ref_smooth[iy_s, ix_s, 1]
    b0 = ref_smooth[iy_s, ix_s, 2]

    vx0 = tx_field[iy_s, ix_s]
    vy0 = ty_field[iy_s, ix_s]
    if bias_strength > 0.0:
        if vx0 * angle_bias_x + vy0 * angle_bias_y < 0.0:
            vx0 = -vx0
            vy0 = -vy0
        vx0 = (1.0 - bias_strength) * vx0 + bias_strength * angle_bias_x
        vy0 = (1.0 - bias_strength) * vy0 + bias_strength * angle_bias_y
    nrm = math.hypot(vx0, vy0)
    if nrm < 1e-5:
        vx0, vy0 = 1.0, 0.0
    else:
        vx0 /= nrm
        vy0 /= nrm

    # Forward trace
    fx, fy = sx, sy
    pvx, pvy = vx0, vy0
    for _ in range(max_half_steps):
        nx = fx + pvx * step_len
        ny = fy + pvy * step_len
        if nx < 0.0 or nx >= w - 1.0 or ny < 0.0 or ny >= h - 1.0:
            break
        inx = int(nx)
        iny = int(ny)
        cd = math.hypot(
            ref_smooth[iny, inx, 0] - r0,
            math.hypot(ref_smooth[iny, inx, 1] - g0, ref_smooth[iny, inx, 2] - b0),
        )
        if cd > color_thresh:
            break
        tvx = tx_field[iny, inx]
        tvy = ty_field[iny, inx]
        if bias_strength > 0.0:
            if tvx * pvx + tvy * pvy < 0.0:
                tvx = -tvx
                tvy = -tvy
            tvx = (1.0 - bias_strength) * tvx + bias_strength * angle_bias_x
            tvy = (1.0 - bias_strength) * tvy + bias_strength * angle_bias_y
        if tvx * pvx + tvy * pvy < 0.0:
            tvx = -tvx
            tvy = -tvy
        tn = math.hypot(tvx, tvy)
        if tn > 1e-5:
            tvx /= tn
            tvy /= tn
            if tvx * pvx + tvy * pvy < 0.48:
                break
            pvx = 0.65 * tvx + 0.35 * pvx
            pvy = 0.65 * tvy + 0.35 * pvy
            pn = math.hypot(pvx, pvy)
            if pn > 1e-5:
                pvx /= pn
                pvy /= pn
        fx, fy = nx, ny

    # Backward trace
    bx, by = sx, sy
    pvx, pvy = -vx0, -vy0
    for _ in range(max_half_steps):
        nx = bx + pvx * step_len
        ny = by + pvy * step_len
        if nx < 0.0 or nx >= w - 1.0 or ny < 0.0 or ny >= h - 1.0:
            break
        inx = int(nx)
        iny = int(ny)
        cd = math.hypot(
            ref_smooth[iny, inx, 0] - r0,
            math.hypot(ref_smooth[iny, inx, 1] - g0, ref_smooth[iny, inx, 2] - b0),
        )
        if cd > color_thresh:
            break
        tvx = tx_field[iny, inx]
        tvy = ty_field[iny, inx]
        if bias_strength > 0.0:
            if tvx * pvx + tvy * pvy < 0.0:
                tvx = -tvx
                tvy = -tvy
            tvx = (1.0 - bias_strength) * tvx - bias_strength * angle_bias_x
            tvy = (1.0 - bias_strength) * tvy - bias_strength * angle_bias_y
        if tvx * pvx + tvy * pvy < 0.0:
            tvx = -tvx
            tvy = -tvy
        tn = math.hypot(tvx, tvy)
        if tn > 1e-5:
            tvx /= tn
            tvy /= tn
            if tvx * pvx + tvy * pvy < 0.48:
                break
            pvx = 0.65 * tvx + 0.35 * pvx
            pvy = 0.65 * tvy + 0.35 * pvy
            pn = math.hypot(pvx, pvy)
            if pn > 1e-5:
                pvx /= pn
                pvy /= pn
        bx, by = nx, ny

    cx = 2.0 * sx - 0.5 * (bx + fx)
    cy = 2.0 * sy - 0.5 * (by + fy)
    cx = max(0.0, min(float(w - 1), cx))
    cy = max(0.0, min(float(h - 1), cy))
    return bx, by, cx, cy, fx, fy


@njit(cache=True)
def run_error_guided_pass(
    canvas: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    roi_mask: np.ndarray,
    tx_field: np.ndarray,
    ty_field: np.ndarray,
    ref_smooth: np.ndarray,
    btype: int,
    r_min: float,
    r_max: float,
    opacity_min: float,
    opacity_max: float,
    step_mult: float,
    max_half_steps: int,
    color_thresh: float,
    max_color_dev: float,
    min_gain_base: float,
    n_candidates: int,
    max_accept: int,
    seed_offset: int,
    err_power: float,
    bias_x: float,
    bias_y: float,
    bias_strength: float,
    peak_pooling: bool,
    local_hand_prob: float,
    local_hand_radius: float,
):
    """Error-guided curved stroke pass with peak-error pooling + local artist-hand momentum.

    Local artist-hand momentum keeps consecutive strokes flowing organically in the same
    neighborhood (so the live brush cursor glides smoothly like an artist's hand) while
    preserving exact z-order fidelity.
    """
    h, w, _ = canvas.shape
    out = np.empty((max_accept, 12), dtype=np.float32)
    accepted = 0

    block_size = max(180, n_candidates // 14)
    gh = (h + 3) // 4
    gw = (w + 3) // 4
    n_cells = gh * gw
    cdf = np.empty(n_cells, dtype=np.float64)
    cell_err = np.empty(n_cells, dtype=np.float32)
    best_dx = np.empty(n_cells, dtype=np.int8)
    best_dy = np.empty(n_cells, dtype=np.int8)

    rng_state = np.uint64(seed_offset * 6364136223846793005 + 1442695040888963407)

    has_hand = False
    hand_x = w * 0.5
    hand_y = h * 0.5
    local_misses = 0

    cand_idx = 0
    while cand_idx < n_candidates and accepted < max_accept:
        total = 0.0
        idx_c = 0
        for gy in range(gh):
            y_base = gy * 4
            y_end = min(h, y_base + 4)
            for gx in range(gw):
                x_base = gx * 4
                x_end = min(w, x_base + 4)

                if peak_pooling:
                    max_p = 0.0
                    bdx = 2
                    bdy = 2
                    for y in range(y_base, y_end):
                        for x in range(x_base, x_end):
                            rm = roi_mask[y, x]
                            if rm > 0.02:
                                dr = canvas[y, x, 0] - target[y, x, 0]
                                dg = canvas[y, x, 1] - target[y, x, 1]
                                db = canvas[y, x, 2] - target[y, x, 2]
                                e = math.sqrt(dr * dr + dg * dg + db * db)
                                p = rm * weight[y, x] * math.pow(e + 0.5, err_power)
                                if p > max_p:
                                    max_p = p
                                    bdx = x - x_base
                                    bdy = y - y_base
                    total += max_p
                    cell_err[idx_c] = float(max_p)
                    best_dx[idx_c] = bdx
                    best_dy[idx_c] = bdy
                else:
                    y = min(h - 1, y_base + 2)
                    x = min(w - 1, x_base + 2)
                    rm = roi_mask[y, x]
                    if rm > 0.02:
                        dr = canvas[y, x, 0] - target[y, x, 0]
                        dg = canvas[y, x, 1] - target[y, x, 1]
                        db = canvas[y, x, 2] - target[y, x, 2]
                        e = math.sqrt(dr * dr + dg * dg + db * db)
                        p = rm * weight[y, x] * math.pow(e + 1.0, err_power)
                    else:
                        p = 0.0
                    total += p
                    cell_err[idx_c] = float(p)
                    best_dx[idx_c] = 2
                    best_dy[idx_c] = 2

                cdf[idx_c] = total
                idx_c += 1

        if total <= 1e-9:
            break

        batch_end = min(n_candidates, cand_idx + block_size)
        while cand_idx < batch_end and accepted < max_accept:
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u0 = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u1 = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u2 = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u3 = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u4 = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0
            rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            u5 = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0

            use_local = has_hand and (local_hand_prob > 0.0) and (u0 < local_hand_prob) and (local_misses < 5)
            chosen_cell = -1

            if use_local:
                # Pick highest-error cell among 8 nearby samples around artist's current hand position
                best_lp = 0.0
                for _k in range(8):
                    rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                    la = float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0 * 6.2831853
                    rng_state = rng_state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                    lr = math.sqrt(float((rng_state >> np.uint64(16)) & np.uint64(0xFFFFFFFF)) / 4294967296.0) * local_hand_radius
                    lx = hand_x + math.cos(la) * lr
                    ly = hand_y + math.sin(la) * lr
                    if 0.0 <= lx < w and 0.0 <= ly < h:
                        lgx = min(gw - 1, int(lx) // 4)
                        lgy = min(gh - 1, int(ly) // 4)
                        lc = lgy * gw + lgx
                        lp = cell_err[lc]
                        if lp > best_lp:
                            best_lp = lp
                            chosen_cell = lc

            if chosen_cell < 0:
                target_val = u1 * total
                lo = 0
                hi = n_cells - 1
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if cdf[mid] < target_val:
                        lo = mid + 1
                    else:
                        hi = mid
                chosen_cell = lo

            gy = chosen_cell // gw
            gx = chosen_cell % gw
            if peak_pooling:
                sx = min(float(w - 1), max(0.0, gx * 4.0 + float(best_dx[chosen_cell]) + (u2 - 0.5) * 0.9))
                sy = min(float(h - 1), max(0.0, gy * 4.0 + float(best_dy[chosen_cell]) + (u3 - 0.5) * 0.9))
            else:
                sx = min(float(w - 1), max(0.0, gx * 4.0 + u2 * 4.0))
                sy = min(float(h - 1), max(0.0, gy * 4.0 + u3 * 4.0))

            r = r_min + (r_max - r_min) * u4
            opacity = opacity_min + (opacity_max - opacity_min) * u5

            x0, y0, cx, cy, x1, y1 = trace_streamline(
                sx,
                sy,
                tx_field,
                ty_field,
                ref_smooth,
                max(1.0, r * step_mult),
                max_half_steps,
                color_thresh,
                bias_x,
                bias_y,
                bias_strength,
            )

            span = math.hypot(x1 - x0, y1 - y0)
            if span < max(1.2, r * 0.65):
                ix_s = min(w - 1, max(0, int(sx)))
                iy_s = min(h - 1, max(0, int(sy)))
                tvx = tx_field[iy_s, ix_s]
                tvy = ty_field[iy_s, ix_s]
                if bias_strength > 0.0:
                    if tvx * bias_x + tvy * bias_y < 0.0:
                        tvx = -tvx
                        tvy = -tvy
                    tvx = (1.0 - bias_strength) * tvx + bias_strength * bias_x
                    tvy = (1.0 - bias_strength) * tvy + bias_strength * bias_y
                tn = math.hypot(tvx, tvy)
                if tn < 1e-5:
                    tvx, tvy = 1.0, 0.0
                else:
                    tvx /= tn
                    tvy /= tn
                half_l = max(1.1, r * 0.85)
                x0 = max(0.0, min(float(w - 1), sx - tvx * half_l))
                y0 = max(0.0, min(float(h - 1), sy - tvy * half_l))
                x1 = max(0.0, min(float(w - 1), sx + tvx * half_l))
                y1 = max(0.0, min(float(h - 1), sy + tvy * half_l))
                cx, cy = sx, sy

            # Orient stroke consistently so cursor flows gracefully from nearest endpoint
            if has_hand:
                d0 = (x0 - hand_x) * (x0 - hand_x) + (y0 - hand_y) * (y0 - hand_y)
                d1 = (x1 - hand_x) * (x1 - hand_x) + (y1 - hand_y) * (y1 - hand_y)
                if d1 < d0:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0

            # Exact quantization matching exported painting-data.js
            x0 = round(x0 * 2.0) * 0.5
            y0 = round(y0 * 2.0) * 0.5
            cx = round(cx * 2.0) * 0.5
            cy = round(cy * 2.0) * 0.5
            x1 = round(x1 * 2.0) * 0.5
            y1 = round(y1 * 2.0) * 0.5
            r = max(0.4, round(r * 10.0) * 0.1)
            opacity = round(opacity * 100.0) * 0.01

            stroke_seed = (
                int(x0 * 2.0) * 31
                + int(y0 * 2.0) * 131
                + int(x1 * 2.0) * 1009
                + int(y1 * 2.0) * 4093
            ) & 0x7FFFFFFF
            min_gain = min_gain_base * (r * r + 2.5)
            ok, sr, sg, sb, _ = evaluate_and_maybe_commit(
                canvas,
                target,
                weight,
                btype,
                x0,
                y0,
                cx,
                cy,
                x1,
                y1,
                r,
                opacity,
                stroke_seed,
                max_color_dev,
                min_gain,
                False,
            )
            if ok:
                out[accepted, 0] = float(btype)
                out[accepted, 1] = x0
                out[accepted, 2] = y0
                out[accepted, 3] = cx
                out[accepted, 4] = cy
                out[accepted, 5] = x1
                out[accepted, 6] = y1
                out[accepted, 7] = r
                out[accepted, 8] = opacity
                out[accepted, 9] = sr
                out[accepted, 10] = sg
                out[accepted, 11] = sb
                accepted += 1
                has_hand = True
                hand_x = x1
                hand_y = y1
                local_misses = 0
                cell_err[chosen_cell] *= 0.35
            else:
                local_misses += 1

            cand_idx += 1

    return out[:accepted]


# ---------------------------------------------------------------------------
# Multi-scale flow fields & semantic ROI masks for Yi Sun-sin portrait
# ---------------------------------------------------------------------------

def compute_tangent_field(gray: np.ndarray, grad_sigma: float, tensor_sigma: float):
    """Return unit tangent vectors (tx, ty) aligned along contours/forms."""
    sm = ndimage.gaussian_filter(gray, grad_sigma)
    gx = ndimage.sobel(sm, axis=1)
    gy = ndimage.sobel(sm, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, tensor_sigma)
    jxy = ndimage.gaussian_filter(gx * gy, tensor_sigma)
    jyy = ndimage.gaussian_filter(gy * gy, tensor_sigma)
    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    theta_tan = theta_grad + np.pi / 2.0
    tx = np.cos(theta_tan).astype(np.float32)
    ty = np.sin(theta_tan).astype(np.float32)
    return tx, ty


def build_semantic_masks(h: int, w: int, gray: np.ndarray) -> dict[str, np.ndarray]:
    """Create soft anatomical & compositional masks driven by the Reference Study."""
    yy, xx = np.mgrid[0:h, 0:w]
    xn = xx / float(w)
    yn = yy / float(h)

    def rect_mask(x0, x1, y0, y1, blur=10.0):
        m = ((xn >= x0) & (xn <= x1) & (yn >= y0) & (yn <= y1)).astype(np.float32)
        return ndimage.gaussian_filter(m, blur) if blur > 0 else m

    def ell_mask(cx, cy, rx, ry, blur=8.0):
        d = ((xn - cx) / rx) ** 2 + ((yn - cy) / ry) ** 2
        m = (d <= 1.0).astype(np.float32)
        return ndimage.gaussian_filter(m, blur) if blur > 0 else m

    gx = ndimage.sobel(ndimage.gaussian_filter(gray, 1.0), axis=1)
    gy = ndimage.sobel(ndimage.gaussian_filter(gray, 1.0), axis=0)
    gmag = np.hypot(gx, gy)
    g_norm = np.clip(gmag / (np.percentile(gmag, 96.5) + 1e-5), 0.0, 1.0).astype(np.float32)

    eye_left = ell_mask(0.445, 0.236, 0.048, 0.022, 3.5)
    eye_right = ell_mask(0.542, 0.236, 0.048, 0.022, 3.5)
    eyes = np.clip(eye_left + eye_right, 0.0, 1.0)

    brows_nose_mouth = np.clip(
        rect_mask(0.395, 0.595, 0.205, 0.232, 3.5)
        + rect_mask(0.455, 0.540, 0.230, 0.292, 3.5)
        + rect_mask(0.422, 0.572, 0.283, 0.342, 3.5),
        0.0,
        1.0,
    )
    face_all = np.clip(ell_mask(0.495, 0.270, 0.138, 0.108, 9.0), 0.0, 1.0)
    beard = np.clip(rect_mask(0.365, 0.625, 0.280, 0.472, 7.0), 0.0, 1.0)
    hat = np.clip(
        rect_mask(0.362, 0.628, 0.024, 0.216, 7.0)
        + rect_mask(0.254, 0.742, 0.114, 0.194, 5.0),
        0.0,
        1.0,
    )
    chest_badge = rect_mask(0.328, 0.682, 0.476, 0.690, 5.0)
    badge_left = rect_mask(0.328, 0.515, 0.476, 0.688, 4.0)
    badge_right = rect_mask(0.495, 0.682, 0.476, 0.688, 4.0)
    belt = rect_mask(0.295, 0.715, 0.652, 0.734, 5.0)
    pelt_left = rect_mask(0.00, 0.235, 0.40, 0.83, 8.0)
    pelt_right = rect_mask(0.79, 1.00, 0.40, 0.80, 8.0)
    pelt = np.clip(pelt_left + pelt_right, 0.0, 1.0)

    portrait_fg = np.clip(
        hat
        + face_all
        + beard
        + pelt
        + rect_mask(0.03, 0.99, 0.35, 1.00, 12.0),
        0.0,
        1.0,
    )
    bg_rough = np.clip(1.0 - portrait_fg, 0.0, 1.0)
    contour_fg = (portrait_fg * np.power(g_norm, 1.45)).astype(np.float32)

    weight = (
        0.75
        + 1.05 * g_norm
        + 1.4 * hat
        + 1.9 * belt
        + 3.2 * chest_badge
        + 3.0 * beard
        + 3.0 * face_all
        + 5.5 * brows_nose_mouth
        + 12.0 * eyes
    ).astype(np.float32)

    return {
        "all": np.ones((h, w), dtype=np.float32),
        "bg": bg_rough,
        "fg": portrait_fg,
        "contour_fg": contour_fg,
        "hat": hat,
        "face": face_all,
        "eye_left": eye_left,
        "eye_right": eye_right,
        "eyes": eyes,
        "features": np.clip(eyes + brows_nose_mouth, 0.0, 1.0),
        "beard": beard,
        "badge": chest_badge,
        "badge_left": badge_left,
        "badge_right": badge_right,
        "belt": belt,
        "pelt": pelt,
        "pelt_left": pelt_left,
        "pelt_right": pelt_right,
        "robe": np.clip(rect_mask(0.02, 0.99, 0.34, 1.00, 7.0) * (1.0 - 0.65 * chest_badge), 0.0, 1.0),
        "weight": weight,
        "g_norm": g_norm,
    }


def make_underpainting_strokes(
    canvas: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stage 1 (Underpainting): Broad serpentine sweeping warm washes from top to bottom."""
    h, w, _ = canvas.shape
    strokes = []
    seed = 1000

    for pass_idx, (n_rows, n_cols, r_lo, r_hi, op_lo, op_hi) in enumerate([
        (14, 9, 95.0, 138.0, 0.64, 0.84),
        (17, 10, 68.0, 100.0, 0.36, 0.56),
    ]):
        for ry in range(n_rows):
            # Serpentine left-to-right then right-to-left so the live brush sweeps naturally!
            cols = range(n_cols) if (ry % 2 == 0) else range(n_cols - 1, -1, -1)
            dir_sign = 1.0 if (ry % 2 == 0) else -1.0
            for rx in cols:
                cx = (rx + 0.5 + rng.uniform(-0.32, 0.32)) * (w / n_cols)
                cy = (ry + 0.5 + rng.uniform(-0.32, 0.32)) * (h / n_rows)
                ang = rng.uniform(-0.28, 0.28) if pass_idx == 0 else rng.uniform(-0.50, 0.50)
                half_l = rng.uniform(85.0, 145.0) if pass_idx == 0 else rng.uniform(55.0, 102.0)
                bend = rng.uniform(-22.0, 22.0)
                dx = math.cos(ang) * half_l * dir_sign
                dy = math.sin(ang) * half_l * dir_sign
                x0 = round(float(np.clip(cx - dx, 0, w - 1)) * 2.0) * 0.5
                y0 = round(float(np.clip(cy - dy, 0, h - 1)) * 2.0) * 0.5
                x1 = round(float(np.clip(cx + dx, 0, w - 1)) * 2.0) * 0.5
                y1 = round(float(np.clip(cy + dy, 0, h - 1)) * 2.0) * 0.5
                pcx = round(float(np.clip(cx - math.sin(ang) * bend, 0, w - 1)) * 2.0) * 0.5
                pcy = round(float(np.clip(cy + math.cos(ang) * bend, 0, h - 1)) * 2.0) * 0.5
                rad = round(float(rng.uniform(r_lo, r_hi)) * 10.0) * 0.1
                op = round(float(rng.uniform(op_lo, op_hi)) * 100.0) * 0.01
                force = (pass_idx == 0)
                ok, sr, sg, sb, _ = evaluate_and_maybe_commit(
                    canvas, target, weight, 0, x0, y0, pcx, pcy, x1, y1, rad, op, seed, 32.0, 10.0, force
                )
                seed += 1
                if ok:
                    strokes.append([0.0, x0, y0, pcx, pcy, x1, y1, rad, op, sr, sg, sb])

    return np.array(strokes, dtype=np.float32) if strokes else np.zeros((0, 12), dtype=np.float32)


def compute_metrics(canvas: np.ndarray, target: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Compute global & region-wise PSNR, SSIM, and MAE."""
    diff = np.abs(canvas - target)
    mse = float(np.mean((canvas - target) ** 2))
    psnr = 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-8))
    ssim_val = float(ssim(target.astype(np.uint8), np.clip(canvas, 0, 255).astype(np.uint8), channel_axis=2))
    metrics = {
        "global_psnr": round(psnr, 3),
        "global_ssim": round(ssim_val, 4),
        "global_mae": round(float(diff.mean()), 3),
    }
    for name in ["eyes", "features", "face", "beard", "hat", "badge", "belt", "robe", "pelt", "bg"]:
        m = masks[name] > 0.35
        if np.any(m):
            reg_mse = float(np.mean((canvas[m] - target[m]) ** 2))
            reg_psnr = 10.0 * math.log10((255.0 ** 2) / max(reg_mse, 1e-8))
            metrics[f"{name}_psnr"] = round(reg_psnr, 2)
            metrics[f"{name}_mae"] = round(float(diff[m].mean()), 2)
    return metrics


def choose_palette_step(stage_arrays: list[np.ndarray]) -> tuple[int, int]:
    """Pick clean RGB quantization step keeping total palette in ~85–135 swatches (target 50–150)."""
    all_rgb = np.vstack([a[:, 9:12] for a in stage_arrays if len(a) > 0])
    candidates = [16, 20, 24, 28, 32, 36, 40]
    counts = {}
    for step in candidates:
        q = np.clip(np.round(all_rgb / step) * step, 0, 255).astype(np.int32)
        uniq = len(set(map(tuple, q.tolist())))
        counts[step] = uniq
    print("Palette swatch counts by RGB step:", counts)
    # Pick step closest to ~110 within [60, 145]
    best_step = min(candidates, key=lambda s: abs(counts[s] - 110))
    return best_step, counts[best_step]


def save_visualizations(
    version_tag: str,
    target: np.ndarray,
    stage_snapshots: list[np.ndarray],
    stage_counts: list[int],
) -> None:
    """Save 9-stage progression sheet, full side-by-side comparison, and 6-zone detail zooms."""
    final = np.clip(stage_snapshots[-1], 0, 255).astype(np.uint8)
    ref_u8 = target.astype(np.uint8)
    h, w, _ = ref_u8.shape

    # 9 stages + Reference in a 2x5 gallery
    fig, axes = plt.subplots(2, 5, figsize=(20, 12), dpi=125)
    cum = 0
    for i in range(9):
        cum += stage_counts[i]
        ax = axes[i // 5, i % 5]
        ax.imshow(np.clip(stage_snapshots[i], 0, 255).astype(np.uint8))
        ax.set_title(f"{i + 1}. {STAGE_NAMES[i]}\n({cum:,} strokes)", fontsize=10)
        ax.axis("off")
    axes[1, 4].imshow(ref_u8)
    axes[1, 4].set_title("Reference", fontsize=10)
    axes[1, 4].axis("off")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"stages_{version_tag}.png")
    plt.close(fig)

    err_map = np.mean(np.abs(final.astype(np.float32) - target), axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.5), dpi=140)
    axes[0].imshow(ref_u8)
    axes[0].set_title("Reference")
    axes[0].axis("off")
    axes[1].imshow(final)
    axes[1].set_title(f"Painted Canvas ({sum(stage_counts):,} strokes)")
    axes[1].axis("off")
    im = axes[2].imshow(err_map, cmap="inferno", vmin=0, vmax=35)
    axes[2].set_title("Residual MAE Map (0-35)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"compare_{version_tag}.png")
    plt.close(fig)

    crops = [
        ("Face & Eyes", (0.37, 0.62, 0.18, 0.35)),
        ("Beard & Collar", (0.36, 0.63, 0.30, 0.47)),
        ("Twin-Tiger Badge", (0.34, 0.67, 0.48, 0.69)),
        ("Belt & Robe Folds", (0.28, 0.72, 0.65, 0.85)),
        ("Samo Hat & Wings", (0.25, 0.74, 0.03, 0.21)),
        ("Leopard Pelt Fur", (0.00, 0.26, 0.40, 0.74)),
    ]
    fig, axes = plt.subplots(2, 6, figsize=(24, 9.5), dpi=135)
    for col, (title, (x0, x1, y0, y1)) in enumerate(crops):
        px0, px1 = int(x0 * w), int(x1 * w)
        py0, py1 = int(y0 * h), int(y1 * h)
        axes[0, col].imshow(ref_u8[py0:py1, px0:px1])
        axes[0, col].set_title(f"Ref: {title}", fontsize=9.5)
        axes[0, col].axis("off")
        axes[1, col].imshow(final[py0:py1, px0:px1])
        axes[1, col].set_title(f"Painted: {title}", fontsize=9.5)
        axes[1, col].axis("off")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"zooms_{version_tag}.png")
    plt.close(fig)

    Image.fromarray(final).save(VIS_DIR / f"painted_{version_tag}.png")


def export_painting_js(
    stage_arrays: list[np.ndarray],
    w: int,
    h: int,
    palette_step: int,
    out_path: Path,
) -> None:
    """Export compact JS data module with 9 stages, representative brush sizes & palette step."""
    stage_brush_sizes = []
    encoded_stages = []
    for arr in stage_arrays:
        if len(arr) > 0:
            med_diam = float(np.median(arr[:, 7]) * 2.0)
            size_label = f"{med_diam:.1f} px" if med_diam < 10.0 else f"{int(round(med_diam))} px"
        else:
            size_label = "1 px"
        stage_brush_sizes.append(size_label)

        flat = []
        for s in arr:
            flat.extend([
                int(round(s[0])),
                int(round(s[1] * 2.0)),
                int(round(s[2] * 2.0)),
                int(round(s[3] * 2.0)),
                int(round(s[4] * 2.0)),
                int(round(s[5] * 2.0)),
                int(round(s[6] * 2.0)),
                int(round(s[7] * 10.0)),
                int(round(s[8] * 100.0)),
                int(round(s[9])),
                int(round(s[10])),
                int(round(s[11])),
            ])
        encoded_stages.append(flat)

    payload = {
        "width": w,
        "height": h,
        "paletteStep": palette_step,
        "stageNames": STAGE_NAMES,
        "stageBrushSizes": stage_brush_sizes,
        "stages": encoded_stages,
    }
    js_text = "window.PAINTING_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    out_path.write_text(js_text, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"Exported {out_path.name}: {size_kb:.1f} KB | brush sizes: {stage_brush_sizes}")


def main() -> None:
    t0 = time.perf_counter()
    version_tag = "v5"

    img = Image.open(REF_PATH).convert("RGB")
    target = np.asarray(img, dtype=np.float32)
    h, w, _ = target.shape
    gray = (0.2126 * target[:, :, 0] + 0.7152 * target[:, :, 1] + 0.0722 * target[:, :, 2]).astype(np.float32)

    print("Building multi-scale tangent flow fields & semantic ROI masks...")
    masks = build_semantic_masks(h, w, gray)
    weight = masks["weight"]

    ref_s1 = ndimage.gaussian_filter(target, sigma=(9.0, 9.0, 0.0)).astype(np.float32)
    ref_s2 = ndimage.gaussian_filter(target, sigma=(3.8, 3.8, 0.0)).astype(np.float32)
    ref_s3 = ndimage.gaussian_filter(target, sigma=(1.4, 1.4, 0.0)).astype(np.float32)
    ref_s4 = ndimage.gaussian_filter(target, sigma=(0.45, 0.45, 0.0)).astype(np.float32)

    tx1, ty1 = compute_tangent_field(gray, 5.0, 12.0)
    tx2, ty2 = compute_tangent_field(gray, 2.6, 6.5)
    tx3, ty3 = compute_tangent_field(gray, 1.3, 3.0)
    tx4, ty4 = compute_tangent_field(gray, 0.65, 1.4)

    canvas = np.full((h, w, 3), 255.0, dtype=np.float32)
    rng = np.random.default_rng(20261120)

    stage_arrays: list[np.ndarray] = []
    stage_snapshots: list[np.ndarray] = []

    def commit_stage(stage_idx: int, parts: list[np.ndarray]) -> None:
        arr = np.vstack(parts) if parts else np.zeros((0, 12), dtype=np.float32)
        stage_arrays.append(arr)
        stage_snapshots.append(canvas.copy())
        psnr_now = compute_metrics(canvas, target, masks)["global_psnr"]
        print(f"Stage {stage_idx + 1}/9 ({STAGE_NAMES[stage_idx]}): {len(arr):,} strokes | {psnr_now} dB")

    # 1. UNDERPAINTING — Broad serpentine warm aged-silk & major tone washes
    s1 = [make_underpainting_strokes(canvas, target, weight, rng)]
    commit_stage(0, s1)

    # 2. BLOCK-IN — Huge rough masses establishing hat, face, crimson robe, badge, belt, pelt
    s2 = [
        run_error_guided_pass(
            canvas, target, weight, masks["all"], tx1, ty1, ref_s1,
            btype=0, r_min=34.0, r_max=76.0, opacity_min=0.46, opacity_max=0.76,
            step_mult=0.55, max_half_steps=5, color_thresh=50.0, max_color_dev=42.0,
            min_gain_base=6.0, n_candidates=1350, max_accept=400, seed_offset=2000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.65, local_hand_radius=240.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx1, ty1, ref_s1,
            btype=1, r_min=24.0, r_max=46.0, opacity_min=0.45, opacity_max=0.72,
            step_mult=0.60, max_half_steps=5, color_thresh=44.0, max_color_dev=42.0,
            min_gain_base=7.0, n_candidates=1000, max_accept=280, seed_offset=4000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.65, local_hand_radius=200.0,
        ),
    ]
    commit_stage(1, s2)

    # 3. BIG SHAPES — Silky background smoothing + broad directional forms
    s3 = [
        run_error_guided_pass(
            canvas, target, weight, masks["bg"], tx1, ty1, ref_s1,
            btype=0, r_min=32.0, r_max=70.0, opacity_min=0.20, opacity_max=0.42,
            step_mult=0.55, max_half_steps=5, color_thresh=24.0, max_color_dev=26.0,
            min_gain_base=1.0, n_candidates=1150, max_accept=440, seed_offset=9000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.70, local_hand_radius=220.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["all"], tx2, ty2, ref_s2,
            btype=0, r_min=15.0, r_max=34.0, opacity_min=0.36, opacity_max=0.66,
            step_mult=0.55, max_half_steps=5, color_thresh=40.0, max_color_dev=45.0,
            min_gain_base=5.5, n_candidates=3600, max_accept=1200, seed_offset=12000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.72, local_hand_radius=170.0,
        ),
    ]
    commit_stage(2, s3)

    # 4. FORMS — Medium bristle sculpting of 3D portrait volumes + facial planes
    s4 = [
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx2, ty2, ref_s2,
            btype=1, r_min=10.0, r_max=22.0, opacity_min=0.45, opacity_max=0.76,
            step_mult=0.55, max_half_steps=6, color_thresh=38.0, max_color_dev=48.0,
            min_gain_base=5.5, n_candidates=6200, max_accept=2150, seed_offset=20000,
            err_power=1.7, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.75, local_hand_radius=150.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["face"], tx2, ty2, ref_s2,
            btype=0, r_min=7.5, r_max=17.0, opacity_min=0.30, opacity_max=0.56,
            step_mult=0.50, max_half_steps=4, color_thresh=28.0, max_color_dev=38.0,
            min_gain_base=3.8, n_candidates=1700, max_accept=560, seed_offset=30000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.75, local_hand_radius=95.0,
        ),
    ]
    commit_stage(3, s4)

    # 5. MODELING — Final serene silk background glaze + medium-small bristle modeling
    s5 = [
        run_error_guided_pass(
            canvas, target, weight, masks["bg"], tx2, ty2, ref_s2,
            btype=0, r_min=20.0, r_max=48.0, opacity_min=0.16, opacity_max=0.34,
            step_mult=0.55, max_half_steps=5, color_thresh=18.0, max_color_dev=22.0,
            min_gain_base=0.55, n_candidates=1500, max_accept=560, seed_offset=38000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
            local_hand_prob=0.75, local_hand_radius=180.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx3, ty3, ref_s3,
            btype=1, r_min=4.8, r_max=11.0, opacity_min=0.46, opacity_max=0.80,
            step_mult=0.55, max_half_steps=6, color_thresh=35.0, max_color_dev=56.0,
            min_gain_base=4.8, n_candidates=13000, max_accept=4300, seed_offset=42000,
            err_power=1.8, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.76, local_hand_radius=125.0,
        ),
    ]
    commit_stage(4, s5)

    # 6. REFINING — Tight structural shapes + subtle crimson silk brocade (운보문)
    s6 = [
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx3, ty3, ref_s3,
            btype=2, r_min=2.8, r_max=6.6, opacity_min=0.48, opacity_max=0.84,
            step_mult=0.55, max_half_steps=6, color_thresh=34.0, max_color_dev=62.0,
            min_gain_base=4.8, n_candidates=15500, max_accept=5100, seed_offset=60000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.78, local_hand_radius=110.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["robe"], tx3, ty3, ref_s3,
            btype=2, r_min=1.8, r_max=4.4, opacity_min=0.34, opacity_max=0.64,
            step_mult=0.55, max_half_steps=5, color_thresh=26.0, max_color_dev=44.0,
            min_gain_base=2.0, n_candidates=10000, max_accept=3200, seed_offset=76000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.78, local_hand_radius=110.0,
        ),
    ]
    commit_stage(5, s6)

    # 7. DETAILS — Ink drapery contours (의습선), Samo hat grain, Leopard pelt radial fur
    s7 = [
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx4, ty4, ref_s4,
            btype=2, r_min=1.5, r_max=3.4, opacity_min=0.50, opacity_max=0.86,
            step_mult=0.60, max_half_steps=6, color_thresh=34.0, max_color_dev=70.0,
            min_gain_base=4.8, n_candidates=17000, max_accept=5000, seed_offset=90000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.80, local_hand_radius=95.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["contour_fg"], tx4, ty4, ref_s4,
            btype=3, r_min=0.60, r_max=1.55, opacity_min=0.58, opacity_max=0.92,
            step_mult=0.72, max_half_steps=10, color_thresh=44.0, max_color_dev=90.0,
            min_gain_base=2.1, n_candidates=22000, max_accept=6600, seed_offset=110000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.80, local_hand_radius=95.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["hat"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.40, opacity_min=0.50, opacity_max=0.88,
            step_mult=0.62, max_half_steps=8, color_thresh=38.0, max_color_dev=82.0,
            min_gain_base=1.8, n_candidates=11500, max_accept=3400, seed_offset=130000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=75.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["pelt_left"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.45, opacity_min=0.46, opacity_max=0.88,
            step_mult=0.72, max_half_steps=8, color_thresh=48.0, max_color_dev=88.0,
            min_gain_base=1.8, n_candidates=9500, max_accept=2800, seed_offset=140000,
            err_power=1.9, bias_x=-0.84, bias_y=0.54, bias_strength=0.40, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=80.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["pelt_right"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.45, opacity_min=0.46, opacity_max=0.88,
            step_mult=0.72, max_half_steps=8, color_thresh=48.0, max_color_dev=88.0,
            min_gain_base=1.8, n_candidates=9500, max_accept=2800, seed_offset=148000,
            err_power=1.9, bias_x=0.84, bias_y=0.54, bias_strength=0.40, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=80.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["robe"], tx4, ty4, ref_s4,
            btype=2, r_min=1.25, r_max=3.2, opacity_min=0.32, opacity_max=0.64,
            step_mult=0.55, max_half_steps=5, color_thresh=26.0, max_color_dev=50.0,
            min_gain_base=1.7, n_candidates=9500, max_accept=2800, seed_offset=151000,
            err_power=1.7, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=95.0,
        ),
    ]
    commit_stage(6, s7)

    # 8. FINE DETAILS — Face fine modeling, twin-tiger chest badge embroidery, official belt, se-pil beard
    badge_belt = np.clip(masks["badge"] + masks["belt"], 0.0, 1.0)
    s8 = [
        run_error_guided_pass(
            canvas, target, weight, masks["face"], tx4, ty4, ref_s4,
            btype=2, r_min=0.95, r_max=2.2, opacity_min=0.40, opacity_max=0.82,
            step_mult=0.55, max_half_steps=5, color_thresh=26.0, max_color_dev=65.0,
            min_gain_base=1.8, n_candidates=8000, max_accept=2400, seed_offset=154000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=65.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, badge_belt, tx4, ty4, ref_s4,
            btype=2, r_min=1.05, r_max=2.3, opacity_min=0.56, opacity_max=0.90,
            step_mult=0.55, max_half_steps=5, color_thresh=34.0, max_color_dev=84.0,
            min_gain_base=3.2, n_candidates=12000, max_accept=3500, seed_offset=156000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=70.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["badge_left"], tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=1.20, opacity_min=0.58, opacity_max=0.94,
            step_mult=0.50, max_half_steps=4, color_thresh=40.0, max_color_dev=96.0,
            min_gain_base=1.5, n_candidates=17000, max_accept=5200, seed_offset=166000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=55.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["badge_right"], tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=1.20, opacity_min=0.58, opacity_max=0.94,
            step_mult=0.50, max_half_steps=4, color_thresh=40.0, max_color_dev=96.0,
            min_gain_base=1.5, n_candidates=17000, max_accept=5200, seed_offset=176000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=55.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["belt"], tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=1.10, opacity_min=0.58, opacity_max=0.94,
            step_mult=0.50, max_half_steps=5, color_thresh=40.0, max_color_dev=96.0,
            min_gain_base=1.4, n_candidates=11000, max_accept=3300, seed_offset=184000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=60.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["beard"], tx4, ty4, ref_s4,
            btype=3, r_min=0.48, r_max=1.00, opacity_min=0.54, opacity_max=0.92,
            step_mult=0.80, max_half_steps=11, color_thresh=54.0, max_color_dev=98.0,
            min_gain_base=1.4, n_candidates=19000, max_accept=6000, seed_offset=190000,
            err_power=1.9, bias_x=0.0, bias_y=1.0, bias_strength=0.60, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=65.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["beard"], tx4, ty4, ref_s4,
            btype=3, r_min=0.42, r_max=0.78, opacity_min=0.52, opacity_max=0.94,
            step_mult=0.72, max_half_steps=9, color_thresh=58.0, max_color_dev=106.0,
            min_gain_base=0.95, n_candidates=23000, max_accept=7200, seed_offset=201000,
            err_power=1.9, bias_x=0.0, bias_y=1.0, bias_strength=0.65, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=55.0,
        ),
    ]
    commit_stage(7, s8)

    # 9. FINAL TOUCHES — Strictly finest se-pil precision: brows/nose/lips, left & right eyes, micro highlights
    s9 = [
        run_error_guided_pass(
            canvas, target, weight, masks["face"], tx4, ty4, ref_s4,
            btype=3, r_min=0.42, r_max=0.95, opacity_min=0.48, opacity_max=0.90,
            step_mult=0.55, max_half_steps=5, color_thresh=30.0, max_color_dev=84.0,
            min_gain_base=1.1, n_candidates=11500, max_accept=3400, seed_offset=220000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.84, local_hand_radius=55.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["eye_left"], tx4, ty4, ref_s4,
            btype=3, r_min=0.40, r_max=0.80, opacity_min=0.54, opacity_max=0.94,
            step_mult=0.46, max_half_steps=4, color_thresh=28.0, max_color_dev=92.0,
            min_gain_base=0.50, n_candidates=4500, max_accept=1350, seed_offset=232000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.85, local_hand_radius=32.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["eye_right"], tx4, ty4, ref_s4,
            btype=3, r_min=0.40, r_max=0.80, opacity_min=0.54, opacity_max=0.94,
            step_mult=0.46, max_half_steps=4, color_thresh=28.0, max_color_dev=92.0,
            min_gain_base=0.50, n_candidates=4500, max_accept=1350, seed_offset=237000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.85, local_hand_radius=32.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["contour_fg"], tx4, ty4, ref_s4,
            btype=3, r_min=0.42, r_max=0.88, opacity_min=0.54, opacity_max=0.92,
            step_mult=0.60, max_half_steps=7, color_thresh=38.0, max_color_dev=92.0,
            min_gain_base=2.2, n_candidates=13000, max_accept=3600, seed_offset=244000,
            err_power=2.2, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.84, local_hand_radius=75.0,
        ),
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx4, ty4, ref_s4,
            btype=3, r_min=0.42, r_max=0.88, opacity_min=0.54, opacity_max=0.92,
            step_mult=0.52, max_half_steps=5, color_thresh=36.0, max_color_dev=90.0,
            min_gain_base=2.6, n_candidates=14000, max_accept=3800, seed_offset=252000,
            err_power=2.2, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
            local_hand_prob=0.82, local_hand_radius=85.0,
        ),
    ]
    commit_stage(8, s9)

    palette_step, palette_count = choose_palette_step(stage_arrays)
    metrics = compute_metrics(canvas, target, masks)
    stage_counts = [len(a) for a in stage_arrays]
    elapsed = time.perf_counter() - t0

    report = {
        "version": version_tag,
        "elapsed_sec": round(elapsed, 2),
        "total_strokes": int(sum(stage_counts)),
        "palette_step": palette_step,
        "palette_swatches": palette_count,
        "stage_counts": {STAGE_NAMES[i]: stage_counts[i] for i in range(9)},
        "metrics": metrics,
    }
    with open(INT_DIR / f"metrics_{version_tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ({version_tag}) ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_visualizations(version_tag, target, stage_snapshots, stage_counts)
    export_painting_js(stage_arrays, w, h, palette_step, ROOT / "painting-data.js")


if __name__ == "__main__":
    main()
