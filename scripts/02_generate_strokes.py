"""Generate layered, form-following curved brush strokes to paint reference.png (v2).

Improvements in v2:
- Added Brush Type 3 (btype=3, Ultra-Fine Pointed Round / Se-pil Hairline Brush) matching app.js 1:1.
- Upgraded error-guided sampler from center-pixel 4x4 grid to peak-error 4x4 max-pooling,
  guaranteeing 1-pixel beard hairs, eyelid lines, tiger spots, and robe ink contours never get missed.
- Silky 2-layer atmospheric background underwash + dedicated low-opacity bg glazing to remove blotchiness.
- Dedicated Stage 4 feature passes: contour-ridge ink lines (의습선), subtle crimson brocade,
  Samo wing wood-grain, outward radial leopard-pelt fur, twin-tiger micro-embroidery,
  ultra-fine vertical beard hairs (세필 수염), and razor-crisp eyes/brows/nose/lips.
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
        # Type 0: Soft Round Wash (radial gradient dabs along quadratic Bezier)
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
        # Type 1: Bristle Brush (soft base + 5 tapered parallel filaments)
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
        # Type 2: Fine Soft-Round Brush (2 concentric passes)
        _rasterize_bezier_stroke(trans, bx0, by0, x0, y0, cx, cy, x1, y1, r * 1.05, opacity * 0.38)
        _rasterize_bezier_stroke(trans, bx0, by0, x0, y0, cx, cy, x1, y1, r * 0.62, opacity * 0.78)
    else:
        # Type 3: Ultra-Fine Pointed Hair / Se-pil Round Liner Brush
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
):
    """Sample high-error locations inside roi_mask (using peak-error 4x4 pooling for thin features)."""
    h, w, _ = canvas.shape
    out = np.empty((max_accept, 12), dtype=np.float32)
    accepted = 0

    block_size = max(180, n_candidates // 14)
    gh = (h + 3) // 4
    gw = (w + 3) // 4
    n_cells = gh * gw
    cdf = np.empty(n_cells, dtype=np.float64)
    best_dx = np.empty(n_cells, dtype=np.int8)
    best_dy = np.empty(n_cells, dtype=np.int8)

    rng_state = np.uint64(seed_offset * 6364136223846793005 + 1442695040888963407)

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
                    # Find exact peak-error pixel inside the 4x4 cell so 1px hairs/lines never slip through
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
                    best_dx[idx_c] = 2
                    best_dy[idx_c] = 2

                cdf[idx_c] = total
                idx_c += 1

        if total <= 1e-9:
            break

        batch_end = min(n_candidates, cand_idx + block_size)
        while cand_idx < batch_end and accepted < max_accept:
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

            target_val = u1 * total
            lo = 0
            hi = n_cells - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if cdf[mid] < target_val:
                    lo = mid + 1
                else:
                    hi = mid
            gy = lo // gw
            gx = lo % gw
            if peak_pooling:
                sx = min(float(w - 1), max(0.0, gx * 4.0 + float(best_dx[lo]) + (u2 - 0.5) * 0.9))
                sy = min(float(h - 1), max(0.0, gy * 4.0 + float(best_dy[lo]) + (u3 - 0.5) * 0.9))
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
    """Create soft anatomical & compositional masks based on 01_coordinate_grid.png."""
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

    eyes = np.clip(
        ell_mask(0.445, 0.236, 0.048, 0.022, 3.5) + ell_mask(0.542, 0.236, 0.048, 0.022, 3.5),
        0.0,
        1.0,
    )
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
        + 1.8 * belt
        + 3.0 * chest_badge
        + 2.8 * beard
        + 3.0 * face_all
        + 5.5 * brows_nose_mouth
        + 11.0 * eyes
    ).astype(np.float32)

    return {
        "all": np.ones((h, w), dtype=np.float32),
        "bg": bg_rough,
        "fg": portrait_fg,
        "contour_fg": contour_fg,
        "hat": hat,
        "face": face_all,
        "eyes": eyes,
        "features": np.clip(eyes + brows_nose_mouth, 0.0, 1.0),
        "beard": beard,
        "badge": chest_badge,
        "belt": belt,
        "pelt": pelt,
        "pelt_left": pelt_left,
        "pelt_right": pelt_right,
        "robe": np.clip(rect_mask(0.02, 0.99, 0.34, 1.00, 7.0) * (1.0 - 0.65 * chest_badge), 0.0, 1.0),
        "weight": weight,
        "g_norm": g_norm,
    }


def make_blocking_wash_strokes(
    canvas: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stage 1 opening: 2-layer broad organic atmospheric washes across the blank canvas."""
    h, w, _ = canvas.shape
    strokes = []
    seed = 1000

    # Layer A: Broad horizontal-angled soft washes establishing warm silk ground & major bands
    for pass_idx, (n_rows, n_cols, r_lo, r_hi, op_lo, op_hi) in enumerate([
        (14, 9, 95.0, 138.0, 0.62, 0.82),
        (18, 11, 68.0, 102.0, 0.38, 0.56),
    ]):
        for ry in range(n_rows):
            for rx in range(n_cols):
                cx = (rx + 0.5 + rng.uniform(-0.34, 0.34)) * (w / n_cols)
                cy = (ry + 0.5 + rng.uniform(-0.34, 0.34)) * (h / n_rows)
                ang = rng.uniform(-0.32, 0.32) if pass_idx == 0 else rng.uniform(-0.55, 0.55)
                half_l = rng.uniform(85.0, 145.0) if pass_idx == 0 else rng.uniform(55.0, 105.0)
                bend = rng.uniform(-22.0, 22.0)
                x0 = round(float(np.clip(cx - math.cos(ang) * half_l, 0, w - 1)) * 2.0) * 0.5
                y0 = round(float(np.clip(cy - math.sin(ang) * half_l, 0, h - 1)) * 2.0) * 0.5
                x1 = round(float(np.clip(cx + math.cos(ang) * half_l, 0, w - 1)) * 2.0) * 0.5
                y1 = round(float(np.clip(cy + math.sin(ang) * half_l, 0, h - 1)) * 2.0) * 0.5
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


def save_visualizations(
    version_tag: str,
    target: np.ndarray,
    stage_snapshots: list[np.ndarray],
    stage_counts: list[int],
) -> None:
    """Save 4-stage progression, full side-by-side comparison, and 6-zone high-res detail crops."""
    final = np.clip(stage_snapshots[-1], 0, 255).astype(np.uint8)
    ref_u8 = target.astype(np.uint8)
    h, w, _ = ref_u8.shape

    stage_names = ["1. Blocking in", "2. Form and color", "3. Refinement", "4. Fine details"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 7), dpi=130)
    cum = 0
    for i in range(4):
        cum += stage_counts[i]
        axes[i].imshow(np.clip(stage_snapshots[i], 0, 255).astype(np.uint8))
        axes[i].set_title(f"{stage_names[i]}\n({cum:,} strokes)", fontsize=10)
        axes[i].axis("off")
    axes[4].imshow(ref_u8)
    axes[4].set_title("Reference", fontsize=10)
    axes[4].axis("off")
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


def export_painting_js(stage_arrays: list[np.ndarray], w: int, h: int, out_path: Path) -> None:
    """Export compact JS data module loaded by index.html."""
    stage_meta = [
        "Blocking in",
        "Form and color",
        "Refinement",
        "Fine details",
    ]
    encoded_stages = []
    for arr in stage_arrays:
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
        "stageNames": stage_meta,
        "stages": encoded_stages,
    }
    js_text = "window.PAINTING_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    out_path.write_text(js_text, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"Exported {out_path.name}: {size_kb:.1f} KB")


def main() -> None:
    t0 = time.perf_counter()
    version_tag = "v3"

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

    # =========================================================================
    # STAGE 1: BLOCKING IN (huge soft washes + rough tonal masses)
    # =========================================================================
    print("\n--- Stage 1: Blocking in ---")
    s1_parts = [make_blocking_wash_strokes(canvas, target, weight, rng)]
    s1_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["all"], tx1, ty1, ref_s1,
            btype=0, r_min=34.0, r_max=76.0, opacity_min=0.46, opacity_max=0.76,
            step_mult=0.55, max_half_steps=5, color_thresh=50.0, max_color_dev=42.0,
            min_gain_base=6.0, n_candidates=1300, max_accept=380, seed_offset=2000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    s1_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx1, ty1, ref_s1,
            btype=1, r_min=22.0, r_max=44.0, opacity_min=0.45, opacity_max=0.72,
            step_mult=0.60, max_half_steps=5, color_thresh=44.0, max_color_dev=42.0,
            min_gain_base=7.0, n_candidates=1000, max_accept=280, seed_offset=4000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    s1 = np.vstack(s1_parts)
    stage_arrays.append(s1)
    stage_snapshots.append(canvas.copy())
    print(f"Stage 1 strokes: {len(s1):,} | {compute_metrics(canvas, target, masks)['global_psnr']} dB")

    # =========================================================================
    # STAGE 2: FORM AND COLOR (silky background glazing + medium 3D sculpting)
    # =========================================================================
    print("\n--- Stage 2: Form and color ---")
    s2_parts = []
    # Silky atmospheric background wash smoothing (eliminates Stage 1 blotchiness)
    s2_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["bg"], tx1, ty1, ref_s1,
            btype=0, r_min=34.0, r_max=72.0, opacity_min=0.20, opacity_max=0.42,
            step_mult=0.55, max_half_steps=5, color_thresh=24.0, max_color_dev=26.0,
            min_gain_base=1.0, n_candidates=1100, max_accept=420, seed_offset=9000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    # Broad medium soft washes across full portrait
    s2_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["all"], tx2, ty2, ref_s2,
            btype=0, r_min=15.0, r_max=34.0, opacity_min=0.36, opacity_max=0.66,
            step_mult=0.55, max_half_steps=5, color_thresh=40.0, max_color_dev=45.0,
            min_gain_base=5.5, n_candidates=3600, max_accept=1200, seed_offset=12000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    # Directional bristle sculpting of portrait forms
    s2_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx2, ty2, ref_s2,
            btype=1, r_min=10.0, r_max=22.0, opacity_min=0.45, opacity_max=0.76,
            step_mult=0.55, max_half_steps=6, color_thresh=38.0, max_color_dev=48.0,
            min_gain_base=6.0, n_candidates=5800, max_accept=1950, seed_offset=20000,
            err_power=1.7, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    # Warm facial plane modeling
    s2_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["face"], tx2, ty2, ref_s2,
            btype=0, r_min=7.5, r_max=17.0, opacity_min=0.30, opacity_max=0.56,
            step_mult=0.50, max_half_steps=4, color_thresh=28.0, max_color_dev=38.0,
            min_gain_base=4.0, n_candidates=1600, max_accept=520, seed_offset=30000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    s2 = np.vstack(s2_parts)
    stage_arrays.append(s2)
    stage_snapshots.append(canvas.copy())
    print(f"Stage 2 strokes: {len(s2):,} | {compute_metrics(canvas, target, masks)['global_psnr']} dB")

    # =========================================================================
    # STAGE 3: REFINEMENT (tightened forms, subtle silk glazing, brocade ground)
    # =========================================================================
    print("\n--- Stage 3: Refinement ---")
    s3_parts = []
    # Final silky calm background glaze
    s3_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["bg"], tx2, ty2, ref_s2,
            btype=0, r_min=20.0, r_max=48.0, opacity_min=0.16, opacity_max=0.34,
            step_mult=0.55, max_half_steps=5, color_thresh=18.0, max_color_dev=22.0,
            min_gain_base=0.6, n_candidates=1400, max_accept=520, seed_offset=38000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=False,
        )
    )
    # Medium-small bristle refinement on foreground
    s3_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx3, ty3, ref_s3,
            btype=1, r_min=5.0, r_max=11.0, opacity_min=0.46, opacity_max=0.80,
            step_mult=0.55, max_half_steps=6, color_thresh=35.0, max_color_dev=56.0,
            min_gain_base=5.0, n_candidates=12500, max_accept=4000, seed_offset=42000,
            err_power=1.8, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )
    # Tight fine-round structural sculpting across foreground
    s3_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx3, ty3, ref_s3,
            btype=2, r_min=3.0, r_max=6.8, opacity_min=0.48, opacity_max=0.84,
            step_mult=0.55, max_half_steps=6, color_thresh=34.0, max_color_dev=62.0,
            min_gain_base=5.0, n_candidates=15000, max_accept=4800, seed_offset=60000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )
    # Subtle crimson robe brocade cloud-and-treasure texture (운보문)
    s3_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["robe"], tx3, ty3, ref_s3,
            btype=2, r_min=2.0, r_max=4.8, opacity_min=0.34, opacity_max=0.62,
            step_mult=0.55, max_half_steps=5, color_thresh=26.0, max_color_dev=42.0,
            min_gain_base=2.2, n_candidates=8500, max_accept=2600, seed_offset=76000,
            err_power=1.6, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )
    s3 = np.vstack(s3_parts)
    stage_arrays.append(s3)
    stage_snapshots.append(canvas.copy())
    print(f"Stage 3 strokes: {len(s3):,} | {compute_metrics(canvas, target, masks)['global_psnr']} dB")

    # =========================================================================
    # STAGE 4: FINE DETAILS (master se-pil brushwork on every signature element)
    # =========================================================================
    print("\n--- Stage 4: Fine details ---")
    s4_parts = []

    # 4A. Medium-fine foreground precision (btype=2)
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx4, ty4, ref_s4,
            btype=2, r_min=1.6, r_max=3.6, opacity_min=0.50, opacity_max=0.86,
            step_mult=0.60, max_half_steps=6, color_thresh=34.0, max_color_dev=70.0,
            min_gain_base=5.0, n_candidates=17000, max_accept=4800, seed_offset=90000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4B. Contour-ridge continuous ink drapery lines (의습선), hat rim & belt frames (btype=3)
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["contour_fg"], tx4, ty4, ref_s4,
            btype=3, r_min=0.60, r_max=1.60, opacity_min=0.58, opacity_max=0.92,
            step_mult=0.72, max_half_steps=10, color_thresh=44.0, max_color_dev=90.0,
            min_gain_base=2.2, n_candidates=22000, max_accept=6500, seed_offset=110000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4B2. Subtle crimson robe brocade clouds (운보문) & soft drapery highlights
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["robe"], tx4, ty4, ref_s4,
            btype=2, r_min=1.3, r_max=3.4, opacity_min=0.32, opacity_max=0.64,
            step_mult=0.55, max_half_steps=5, color_thresh=26.0, max_color_dev=48.0,
            min_gain_base=1.8, n_candidates=11000, max_accept=3200, seed_offset=122000,
            err_power=1.7, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4C. Samo Hat crown gauze texture & wing wood-grain swirls (양각)
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["hat"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.45, opacity_min=0.50, opacity_max=0.88,
            step_mult=0.62, max_half_steps=8, color_thresh=38.0, max_color_dev=82.0,
            min_gain_base=1.8, n_candidates=11500, max_accept=3400, seed_offset=130000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4D. Leopard Pelt (표피) outward feathery fur hairs + crescent rosettes
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["pelt_left"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.45, opacity_min=0.46, opacity_max=0.88,
            step_mult=0.72, max_half_steps=8, color_thresh=48.0, max_color_dev=88.0,
            min_gain_base=1.8, n_candidates=9500, max_accept=2850, seed_offset=140000,
            err_power=1.9, bias_x=-0.84, bias_y=0.54, bias_strength=0.40, peak_pooling=True,
        )
    )
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["pelt_right"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.45, opacity_min=0.46, opacity_max=0.88,
            step_mult=0.72, max_half_steps=8, color_thresh=48.0, max_color_dev=88.0,
            min_gain_base=1.8, n_candidates=9500, max_accept=2850, seed_offset=148000,
            err_power=1.9, bias_x=0.84, bias_y=0.54, bias_strength=0.40, peak_pooling=True,
        )
    )

    # 4E. Twin-Tiger Chest Badge (쌍호흉배) & Belt (각대) micro-embroidery (spots, faces, waves, gold trim)
    badge_belt = np.clip(masks["badge"] + masks["belt"], 0.0, 1.0)
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, badge_belt, tx4, ty4, ref_s4,
            btype=2, r_min=1.1, r_max=2.4, opacity_min=0.56, opacity_max=0.90,
            step_mult=0.55, max_half_steps=5, color_thresh=34.0, max_color_dev=82.0,
            min_gain_base=3.4, n_candidates=12000, max_accept=3500, seed_offset=156000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, badge_belt, tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.35, opacity_min=0.58, opacity_max=0.94,
            step_mult=0.55, max_half_steps=5, color_thresh=38.0, max_color_dev=94.0,
            min_gain_base=2.0, n_candidates=18000, max_accept=5400, seed_offset=170000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )
    # 4E2. Ultra-short curved micro-strokes for ringed tiger spots, eyes, whiskers & woven belt lattice
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, badge_belt, tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=0.90, opacity_min=0.58, opacity_max=0.94,
            step_mult=0.45, max_half_steps=3, color_thresh=42.0, max_color_dev=98.0,
            min_gain_base=1.4, n_candidates=20000, max_accept=5800, seed_offset=182000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4F. Long Flowing Beard, Sideburns & Mustache (세필 수염 — two ultra-fine vertical passes!)
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["beard"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.05, opacity_min=0.54, opacity_max=0.92,
            step_mult=0.80, max_half_steps=11, color_thresh=54.0, max_color_dev=98.0,
            min_gain_base=1.5, n_candidates=18000, max_accept=5600, seed_offset=190000,
            err_power=1.9, bias_x=0.0, bias_y=1.0, bias_strength=0.60, peak_pooling=True,
        )
    )
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["beard"], tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=0.80, opacity_min=0.52, opacity_max=0.94,
            step_mult=0.72, max_half_steps=9, color_thresh=58.0, max_color_dev=105.0,
            min_gain_base=1.0, n_candidates=22000, max_accept=6800, seed_offset=201000,
            err_power=1.9, bias_x=0.0, bias_y=1.0, bias_strength=0.64, peak_pooling=True,
        )
    )

    # 4G. Face, Brows, Nose, Ears & Mouth delicate modeling + crisp hairlines
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["face"], tx4, ty4, ref_s4,
            btype=2, r_min=1.0, r_max=2.4, opacity_min=0.40, opacity_max=0.82,
            step_mult=0.55, max_half_steps=5, color_thresh=26.0, max_color_dev=65.0,
            min_gain_base=2.0, n_candidates=7500, max_accept=2200, seed_offset=210000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["face"], tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=1.15, opacity_min=0.48, opacity_max=0.90,
            step_mult=0.55, max_half_steps=5, color_thresh=30.0, max_color_dev=82.0,
            min_gain_base=1.3, n_candidates=11000, max_accept=3200, seed_offset=220000,
            err_power=1.9, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4H. Eyes ultra-fine signature finishing (eyelids, double-lid creases, irises, catchlights)
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["eyes"], tx4, ty4, ref_s4,
            btype=3, r_min=0.45, r_max=0.90, opacity_min=0.54, opacity_max=0.94,
            step_mult=0.48, max_half_steps=4, color_thresh=28.0, max_color_dev=92.0,
            min_gain_base=0.7, n_candidates=6500, max_accept=1900, seed_offset=232000,
            err_power=2.0, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    # 4I. Final high-residual precision polish across foreground
    s4_parts.append(
        run_error_guided_pass(
            canvas, target, weight, masks["fg"], tx4, ty4, ref_s4,
            btype=3, r_min=0.50, r_max=1.35, opacity_min=0.54, opacity_max=0.90,
            step_mult=0.55, max_half_steps=5, color_thresh=36.0, max_color_dev=88.0,
            min_gain_base=3.2, n_candidates=16000, max_accept=4200, seed_offset=240000,
            err_power=2.2, bias_x=0.0, bias_y=0.0, bias_strength=0.0, peak_pooling=True,
        )
    )

    s4 = np.vstack(s4_parts)
    stage_arrays.append(s4)
    stage_snapshots.append(canvas.copy())

    metrics = compute_metrics(canvas, target, masks)
    stage_counts = [len(a) for a in stage_arrays]
    elapsed = time.perf_counter() - t0

    report = {
        "version": version_tag,
        "elapsed_sec": round(elapsed, 2),
        "total_strokes": int(sum(stage_counts)),
        "stage_counts": {
            "Blocking in": stage_counts[0],
            "Form and color": stage_counts[1],
            "Refinement": stage_counts[2],
            "Fine details": stage_counts[3],
        },
        "metrics": metrics,
    }
    with open(INT_DIR / f"metrics_{version_tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ({version_tag}) ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_visualizations(version_tag, target, stage_snapshots, stage_counts)
    export_painting_js(stage_arrays, w, h, ROOT / "painting-data.js")


if __name__ == "__main__":
    main()
