(function () {
  "use strict";

  const STRIDE = 12;
  // Time-paced playback totaling ~44 seconds (within 30–60s; big strokes slow, details brisk)
  const STAGE_DURATIONS_MS = [4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 4800];

  const data = window.PAINTING_DATA || {
    width: 1280,
    height: 1932,
    paletteStep: 24,
    stageNames: [
      "Underpainting",
      "Block-in",
      "Big shapes",
      "Forms",
      "Modeling",
      "Refining",
      "Details",
      "Fine details",
      "Final touches",
    ],
    stageBrushSizes: ["180 px", "88 px", "46 px", "27 px", "15 px", "8 px", "3.6 px", "1.9 px", "1.4 px"],
    stages: [[], [], [], [], [], [], [], [], []],
  };

  const W = data.width;
  const H = data.height;
  const ASPECT = W / H;
  const NUM_STAGES = data.stageNames.length;
  const PALETTE_STEP = data.paletteStep || 24;

  // DOM elements
  const canvasStage = document.getElementById("canvasStage");
  const canvasFrame = document.getElementById("canvasFrame");
  const canvas = document.getElementById("paintCanvas");
  const ctx = canvas.getContext("2d");
  const overlayCanvas = document.getElementById("overlayCanvas");
  const overlayCtx = overlayCanvas.getContext("2d");

  const brushPreviewCanvas = document.getElementById("brushPreview");
  const brushPreviewCtx = brushPreviewCanvas.getContext("2d");
  const brushTypeNameEl = document.getElementById("brushTypeName");

  const brushSizeInput = document.getElementById("brushSize");
  const brushSizeValueEl = document.getElementById("brushSizeValue");
  const brushOpacityInput = document.getElementById("brushOpacity");
  const brushOpacityValueEl = document.getElementById("brushOpacityValue");

  const colorSwatchEl = document.getElementById("colorSwatch");
  const colorPickerInput = document.getElementById("colorPicker");
  const colorHexEl = document.getElementById("colorHex");
  const colorRgbEl = document.getElementById("colorRgb");

  const topStageCountEl = document.getElementById("topStageCount");
  const topStageNameEl = document.getElementById("topStageName");

  const referenceImageEl = document.getElementById("referenceImage");
  const refCursorEl = document.getElementById("refCursor");

  const statStrokesEl = document.getElementById("statStrokes");
  const statColorsEl = document.getElementById("statColors");
  const stagesListEl = document.getElementById("stagesList");

  const paletteMetaEl = document.getElementById("paletteMeta");
  const paletteGridEl = document.getElementById("paletteGrid");

  const btnPlayPause = document.getElementById("btnPlayPause");
  const btnFinish = document.getElementById("btnFinish");
  const btnReset = document.getElementById("btnReset");
  const railButtons = Array.from(document.querySelectorAll(".rail-btn"));

  canvas.width = W;
  canvas.height = H;
  overlayCanvas.width = W;
  overlayCanvas.height = H;

  // Playback state
  let stageIdx = 0;
  let stageFloat = 0.0; // continuous stroke index within current stage (enables smooth mid-stroke cursor)
  let committedInStage = 0;
  let totalDrawn = 0;
  let isPlaying = false;
  let isFinished = false;
  let rafId = null;
  let lastFrameTime = null;
  let zoomActive = false;
  let renderedDisplayW = 500;

  // Cumulative palette state (never shrinks during painting)
  const paletteMap = new Map(); // quantizedKey -> button element
  let currentPaletteEl = null;

  // Stage list DOM cache
  const stageRowEls = [];

  // Manual tool state
  let activeTool = "brush"; // brush | pencil | eraser | fill
  let manualSize = 24;
  let manualOpacity = 0.75;
  let manualColor = [200, 158, 100];
  let isPointerDown = false;
  let pointerPrev = null;
  let pointerMid = null;

  // -------------------------------------------------------------------------
  // Exact Aspect Ratio Layout (Canvas & Overlay matched to sub-pixel precision)
  // -------------------------------------------------------------------------
  function fitCanvasToStage() {
    const rect = canvasStage.getBoundingClientRect();
    const availW = Math.max(120, rect.width - 40);
    const availH = Math.max(120, rect.height - 32);
    let drawW = availW;
    let drawH = drawW / ASPECT;
    if (drawH > availH) {
      drawH = availH;
      drawW = drawH * ASPECT;
    }
    if (zoomActive) {
      drawW *= 1.35;
      drawH *= 1.35;
    }
    renderedDisplayW = drawW;
    const wCss = `${drawW.toFixed(2)}px`;
    const hCss = `${drawH.toFixed(2)}px`;
    canvasFrame.style.width = wCss;
    canvasFrame.style.height = hCss;
    canvas.style.width = wCss;
    canvas.style.height = hCss;
  }

  window.addEventListener("resize", fitCanvasToStage);
  fitCanvasToStage();

  // -------------------------------------------------------------------------
  // Pure Canvas 2D Brush Primitives (NO drawImage, NO putImageData, NO filters)
  // -------------------------------------------------------------------------
  function detHash(seed, k) {
    let x = (Math.imul(seed, 1664525) + Math.imul(k, 1013904223) + 374761393) & 0x7fffffff;
    x = (x ^ (x >>> 13)) & 0x7fffffff;
    x = (Math.imul(x, 48271) + 12345) & 0x7fffffff;
    x = (x ^ (x >>> 16)) & 0x7fffffff;
    return (x & 0xffff) / 32767.5 - 1.0;
  }

  function evalQuad(x0, y0, cx, cy, x1, y1, t) {
    const mt = 1.0 - t;
    return {
      x: mt * mt * x0 + 2.0 * mt * t * cx + t * t * x1,
      y: mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1,
    };
  }

  // Exact de Casteljau prefix sub-curve for B(0..t)
  function subQuad(x0, y0, cx, cy, x1, y1, t) {
    const cSubX = x0 + (cx - x0) * t;
    const cSubY = y0 + (cy - y0) * t;
    const p1 = evalQuad(x0, y0, cx, cy, x1, y1, t);
    return { x0, y0, cx: cSubX, cy: cSubY, x1: p1.x, y1: p1.y };
  }

  function strokeQuadOn(targetCtx, x0, y0, cx, cy, x1, y1, lineWidth, strokeStyle) {
    targetCtx.lineWidth = lineWidth;
    targetCtx.strokeStyle = strokeStyle;
    targetCtx.beginPath();
    targetCtx.moveTo(x0, y0);
    targetCtx.quadraticCurveTo(cx, cy, x1, y1);
    targetCtx.stroke();
  }

  function drawSoftRoundOn(targetCtx, x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue, maxT) {
    const approxLen = Math.hypot(cx - x0, cy - y0) + Math.hypot(x1 - cx, y1 - cy);
    const step = Math.max(2.5, r * 0.55);
    const nDabs = Math.max(1, Math.min(10, Math.round(approxLen / step) + 1));
    const dabAlpha = nDabs === 1 ? opacity : 1.0 - Math.pow(Math.max(1e-4, 1.0 - opacity), 0.52);

    const c0 = `rgba(${red},${green},${blue},${dabAlpha.toFixed(4)})`;
    const c1 = `rgba(${red},${green},${blue},${(dabAlpha * 0.72).toFixed(4)})`;
    const c2 = `rgba(${red},${green},${blue},0)`;

    const limitDabs = maxT >= 1.0 ? nDabs : Math.max(1, Math.ceil(nDabs * maxT));
    for (let i = 0; i < limitDabs; i++) {
      const t = nDabs === 1 ? 0.5 : i / (nDabs - 1);
      const mt = 1.0 - t;
      const px = mt * mt * x0 + 2.0 * mt * t * cx + t * t * x1;
      const py = mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1;

      const grad = targetCtx.createRadialGradient(px, py, 0, px, py, r);
      grad.addColorStop(0, c0);
      grad.addColorStop(0.55, c1);
      grad.addColorStop(1, c2);
      targetCtx.fillStyle = grad;
      targetCtx.beginPath();
      targetCtx.arc(px, py, r, 0, Math.PI * 2);
      targetCtx.fill();
    }
  }

  function drawBristleOn(targetCtx, x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue, seed, maxT) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const dlen = Math.hypot(dx, dy);
    const nx = dlen < 1e-4 ? 0.0 : -dy / dlen;
    const ny = dlen < 1e-4 ? 1.0 : dx / dlen;
    const invLen = dlen > 1.0 ? 1.0 / dlen : 1.0;

    targetCtx.lineCap = "round";
    targetCtx.lineJoin = "round";

    const baseCurve = maxT < 1.0 ? subQuad(x0, y0, cx, cy, x1, y1, maxT) : { x0, y0, cx, cy, x1, y1 };
    strokeQuadOn(
      targetCtx,
      baseCurve.x0, baseCurve.y0, baseCurve.cx, baseCurve.cy, baseCurve.x1, baseCurve.y1,
      r * 1.84,
      `rgba(${red},${green},${blue},${(opacity * 0.34).toFixed(4)})`
    );

    const offsets = [-0.68, -0.34, 0.0, 0.34, 0.68];
    const weights = [0.38, 0.56, 0.65, 0.54, 0.36];
    const filWidth = Math.max(1.0, r * 0.44);

    for (let i = 0; i < 5; i++) {
      const jOff = detHash(seed, i * 3 + 1) * 0.11;
      const jEnd = detHash(seed, i * 3 + 2) * 0.08 * r;
      const off = (offsets[i] + jOff) * r;
      const endOff = off * 0.78;
      const ux = dx * invLen * jEnd;
      const uy = dy * invLen * jEnd;

      const fx0 = x0 + nx * endOff + ux;
      const fy0 = y0 + ny * endOff + uy;
      const fcx = cx + nx * off;
      const fcy = cy + ny * off;
      const fx1 = x1 + nx * endOff - ux;
      const fy1 = y1 + ny * endOff - uy;

      const fc = maxT < 1.0 ? subQuad(fx0, fy0, fcx, fcy, fx1, fy1, maxT) : { x0: fx0, y0: fy0, cx: fcx, cy: fcy, x1: fx1, y1: fy1 };
      strokeQuadOn(
        targetCtx,
        fc.x0, fc.y0, fc.cx, fc.cy, fc.x1, fc.y1,
        filWidth,
        `rgba(${red},${green},${blue},${(opacity * weights[i]).toFixed(4)})`
      );
    }
  }

  function drawFineRoundOn(targetCtx, x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue, maxT) {
    targetCtx.lineCap = "round";
    targetCtx.lineJoin = "round";
    const c = maxT < 1.0 ? subQuad(x0, y0, cx, cy, x1, y1, maxT) : { x0, y0, cx, cy, x1, y1 };
    strokeQuadOn(targetCtx, c.x0, c.y0, c.cx, c.cy, c.x1, c.y1, r * 2.1, `rgba(${red},${green},${blue},${(opacity * 0.38).toFixed(4)})`);
    strokeQuadOn(targetCtx, c.x0, c.y0, c.cx, c.cy, c.x1, c.y1, r * 1.24, `rgba(${red},${green},${blue},${(opacity * 0.78).toFixed(4)})`);
  }

  function drawHairlineRoundOn(targetCtx, x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue, maxT) {
    targetCtx.lineCap = "round";
    targetCtx.lineJoin = "round";
    const c = maxT < 1.0 ? subQuad(x0, y0, cx, cy, x1, y1, maxT) : { x0, y0, cx, cy, x1, y1 };
    strokeQuadOn(targetCtx, c.x0, c.y0, c.cx, c.cy, c.x1, c.y1, Math.max(0.75, r * 1.48), `rgba(${red},${green},${blue},${opacity.toFixed(4)})`);
  }

  function decodeStroke(arr, offset) {
    const btype = arr[offset];
    const ix0 = arr[offset + 1];
    const iy0 = arr[offset + 2];
    const icx = arr[offset + 3];
    const icy = arr[offset + 4];
    const ix1 = arr[offset + 5];
    const iy1 = arr[offset + 6];
    const r = arr[offset + 7] * 0.1;
    const opacity = arr[offset + 8] * 0.01;
    const red = arr[offset + 9];
    const green = arr[offset + 10];
    const blue = arr[offset + 11];
    const seed =
      (Math.imul(ix0, 31) + Math.imul(iy0, 131) + Math.imul(ix1, 1009) + Math.imul(iy1, 4093)) &
      0x7fffffff;

    return {
      btype,
      x0: ix0 * 0.5,
      y0: iy0 * 0.5,
      cx: icx * 0.5,
      cy: icy * 0.5,
      x1: ix1 * 0.5,
      y1: iy1 * 0.5,
      r,
      opacity,
      red,
      green,
      blue,
      seed,
    };
  }

  function renderStrokeOn(targetCtx, s, maxT) {
    if (s.btype === 0) {
      drawSoftRoundOn(targetCtx, s.x0, s.y0, s.cx, s.cy, s.x1, s.y1, s.r, s.opacity, s.red, s.green, s.blue, maxT);
    } else if (s.btype === 1) {
      drawBristleOn(targetCtx, s.x0, s.y0, s.cx, s.cy, s.x1, s.y1, s.r, s.opacity, s.red, s.green, s.blue, s.seed, maxT);
    } else if (s.btype === 2) {
      drawFineRoundOn(targetCtx, s.x0, s.y0, s.cx, s.cy, s.x1, s.y1, s.r, s.opacity, s.red, s.green, s.blue, maxT);
    } else {
      drawHairlineRoundOn(targetCtx, s.x0, s.y0, s.cx, s.cy, s.x1, s.y1, s.r, s.opacity, s.red, s.green, s.blue, maxT);
    }
  }

  // -------------------------------------------------------------------------
  // Live Overlay Cursors (Canvas Ring + Reference Ring; never painted on artwork)
  // -------------------------------------------------------------------------
  function clearOverlays() {
    overlayCtx.clearRect(0, 0, W, H);
    refCursorEl.classList.remove("visible");
  }

  function renderLiveOverlay(activeStroke, t) {
    overlayCtx.clearRect(0, 0, W, H);
    if (!activeStroke) {
      refCursorEl.classList.remove("visible");
      return;
    }

    // Show smooth in-progress stroke growth on the separate overlay canvas
    if (t > 0.02 && t < 1.0) {
      renderStrokeOn(overlayCtx, activeStroke, t);
    }

    const pt = evalQuad(activeStroke.x0, activeStroke.y0, activeStroke.cx, activeStroke.cy, activeStroke.x1, activeStroke.y1, t);

    // Canvas cursor ring: true brush diameter (with subtle visible floor so tiny 1px hairlines remain legible)
    const canvasScale = W / Math.max(100, renderedDisplayW);
    const minVisibleRadius = 4.0 * canvasScale;
    const ringR = Math.max(activeStroke.r, minVisibleRadius);
    const strokeW = Math.max(1.2, 1.4 * canvasScale);

    overlayCtx.save();
    // Outer white halo for contrast over dark pigments
    overlayCtx.beginPath();
    overlayCtx.arc(pt.x, pt.y, ringR, 0, Math.PI * 2);
    overlayCtx.lineWidth = strokeW * 2.1;
    overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    overlayCtx.stroke();

    // Inner crisp dark/blue ring
    overlayCtx.beginPath();
    overlayCtx.arc(pt.x, pt.y, ringR, 0, Math.PI * 2);
    overlayCtx.lineWidth = strokeW;
    overlayCtx.strokeStyle = "rgba(37, 99, 235, 0.95)";
    overlayCtx.stroke();

    // Tiny center dot when brush is large so center aim stays clear
    if (activeStroke.r > 14) {
      overlayCtx.beginPath();
      overlayCtx.arc(pt.x, pt.y, 1.6 * canvasScale, 0, Math.PI * 2);
      overlayCtx.fillStyle = "rgba(37, 99, 235, 0.85)";
      overlayCtx.fill();
    }
    overlayCtx.restore();

    // Matching ring cursor on the Reference thumbnail at identical relative position
    const relX = (pt.x / W) * 100;
    const relY = (pt.y / H) * 100;
    const refW = referenceImageEl.clientWidth || 118;
    const refDiam = Math.max(7, Math.min(44, ((activeStroke.r * 2.0) / W) * refW));

    refCursorEl.style.left = `${relX.toFixed(2)}%`;
    refCursorEl.style.top = `${relY.toFixed(2)}%`;
    refCursorEl.style.width = `${refDiam.toFixed(1)}px`;
    refCursorEl.style.height = `${refDiam.toFixed(1)}px`;
    refCursorEl.classList.add("visible");
  }

  // -------------------------------------------------------------------------
  // Brush Preview & Top Bar HUD
  // -------------------------------------------------------------------------
  function rgbToHex(r, g, b) {
    const to2 = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
    return `#${to2(r)}${to2(g)}${to2(b)}`.toUpperCase();
  }

  function hexToRgb(hex) {
    const clean = hex.replace("#", "");
    return [
      parseInt(clean.slice(0, 2), 16) || 0,
      parseInt(clean.slice(2, 4), 16) || 0,
      parseInt(clean.slice(4, 6), 16) || 0,
    ];
  }

  function renderBrushPreview(isBristle, sizePx, red, green, blue) {
    const pw = brushPreviewCanvas.width;
    const ph = brushPreviewCanvas.height;
    const cx = pw * 0.5;
    const cy = ph * 0.5;
    brushPreviewCtx.clearRect(0, 0, pw, ph);

    const previewR = Math.max(3.5, Math.min(11, 2.6 + Math.sqrt(sizePx) * 0.62));
    if (isBristle) {
      // Miniature 5-bristle stroke preview
      const offsets = [-0.65, -0.32, 0, 0.32, 0.65];
      brushPreviewCtx.lineCap = "round";
      for (let i = 0; i < offsets.length; i++) {
        const off = offsets[i] * previewR * 0.85;
        brushPreviewCtx.strokeStyle = `rgba(${red},${green},${blue},${i === 2 ? 0.92 : 0.68})`;
        brushPreviewCtx.lineWidth = Math.max(1.1, previewR * 0.24);
        brushPreviewCtx.beginPath();
        brushPreviewCtx.moveTo(5, cy + off + 3);
        brushPreviewCtx.quadraticCurveTo(cx, cy + off - 1, pw - 5, cy + off - 3);
        brushPreviewCtx.stroke();
      }
    } else {
      // Soft radial dab preview
      const g = brushPreviewCtx.createRadialGradient(cx, cy, 0, cx, cy, previewR);
      g.addColorStop(0, `rgba(${red},${green},${blue},0.95)`);
      g.addColorStop(0.6, `rgba(${red},${green},${blue},0.65)`);
      g.addColorStop(1, `rgba(${red},${green},${blue},0)`);
      brushPreviewCtx.fillStyle = g;
      brushPreviewCtx.beginPath();
      brushPreviewCtx.arc(cx, cy, previewR, 0, Math.PI * 2);
      brushPreviewCtx.fill();
    }
  }

  function updateTopBar(sizePx, opacity01, red, green, blue, isBristle) {
    const clampedSize = Math.max(0.8, Math.min(280, sizePx));
    brushSizeInput.value = clampedSize.toFixed(1);
    brushSizeValueEl.textContent = clampedSize < 10 ? `${clampedSize.toFixed(1)} px` : `${Math.round(clampedSize)} px`;

    const opPct = Math.max(5, Math.min(100, Math.round(opacity01 * 100)));
    brushOpacityInput.value = String(opPct);
    brushOpacityValueEl.textContent = `${opPct}%`;

    const rInt = Math.round(red);
    const gInt = Math.round(green);
    const bInt = Math.round(blue);
    const hex = rgbToHex(rInt, gInt, bInt);

    colorSwatchEl.style.background = hex;
    colorPickerInput.value = hex.toLowerCase();
    colorHexEl.textContent = hex;
    colorRgbEl.textContent = `R ${rInt} · G ${gInt} · B ${bInt}`;

    brushTypeNameEl.textContent = isBristle ? "Bristle" : "Soft round";
    renderBrushPreview(isBristle, clampedSize, rInt, gInt, bInt);
  }

  // -------------------------------------------------------------------------
  // Right Sidebar: Stages List, Stats, and Cumulative Quantized Palette
  // -------------------------------------------------------------------------
  const CHECK_SVG =
    '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5 6.5 11.5 12.5 4.5"/></svg>';

  function buildStagesList() {
    stagesListEl.innerHTML = "";
    stageRowEls.length = 0;

    for (let i = 0; i < NUM_STAGES; i++) {
      const li = document.createElement("li");
      li.className = "stage-row";

      const iconSpan = document.createElement("span");
      iconSpan.className = "stage-icon";
      iconSpan.innerHTML = '<span class="circle-empty"></span>';

      const nameSpan = document.createElement("span");
      nameSpan.className = "stage-row-name";
      nameSpan.textContent = data.stageNames[i];

      const sizeSpan = document.createElement("span");
      sizeSpan.className = "stage-row-size";
      sizeSpan.textContent = (data.stageBrushSizes && data.stageBrushSizes[i]) || "";

      li.appendChild(iconSpan);
      li.appendChild(nameSpan);
      li.appendChild(sizeSpan);
      stagesListEl.appendChild(li);
      stageRowEls.push({ li, iconSpan });
    }
  }

  function updateStageUI(activeIdx) {
    const clampedIdx = Math.min(activeIdx, NUM_STAGES - 1);
    const stageName = data.stageNames[clampedIdx];

    topStageCountEl.textContent = `STAGE ${clampedIdx + 1} OF ${NUM_STAGES}`;
    topStageNameEl.textContent = stageName;

    for (let i = 0; i < NUM_STAGES; i++) {
      const row = stageRowEls[i];
      const done = isFinished || i < clampedIdx;
      const active = !isFinished && i === clampedIdx;

      row.li.classList.toggle("done", done);
      row.li.classList.toggle("active", active);

      if (done) {
        row.iconSpan.innerHTML = `<span class="check-done">${CHECK_SVG}</span>`;
      } else if (active) {
        row.iconSpan.innerHTML = '<span class="dot-active"></span>';
      } else {
        row.iconSpan.innerHTML = '<span class="circle-empty"></span>';
      }
    }

    updatePaletteHeader(stageName);
  }

  function updatePaletteHeader(stageName) {
    const n = paletteMap.size;
    const activeName = stageName || data.stageNames[Math.min(stageIdx, NUM_STAGES - 1)];
    paletteMetaEl.textContent = `${n} ${n === 1 ? "color" : "colors"} · ${activeName}`;
  }

  function quantizeChannel(v) {
    return Math.min(255, Math.max(0, Math.round(v / PALETTE_STEP) * PALETTE_STEP));
  }

  function registerPaletteColor(red, green, blue, highlightNow) {
    const qr = quantizeChannel(red);
    const qg = quantizeChannel(green);
    const qb = quantizeChannel(blue);
    const key = (qr << 16) | (qg << 8) | qb;

    let swatchBtn = paletteMap.get(key);
    if (!swatchBtn) {
      const hex = rgbToHex(qr, qg, qb);
      swatchBtn = document.createElement("button");
      swatchBtn.type = "button";
      swatchBtn.className = "palette-swatch";
      swatchBtn.style.background = hex;
      swatchBtn.title = `${hex} (R ${qr}, G ${qg}, B ${qb})`;
      swatchBtn.addEventListener("click", () => {
        pausePainting();
        manualColor = [qr, qg, qb];
        updateTopBar(manualSize, manualOpacity, qr, qg, qb, activeTool === "brush" ? false : false);
        highlightSwatch(swatchBtn);
      });
      paletteMap.set(key, swatchBtn);
      paletteGridEl.appendChild(swatchBtn);
    }

    if (highlightNow) {
      highlightSwatch(swatchBtn);
    }
  }

  function highlightSwatch(el) {
    if (currentPaletteEl === el) return;
    if (currentPaletteEl) {
      currentPaletteEl.classList.remove("current");
    }
    currentPaletteEl = el;
    if (currentPaletteEl) {
      currentPaletteEl.classList.add("current");
    }
  }

  function clearPalette() {
    paletteMap.clear();
    currentPaletteEl = null;
    paletteGridEl.innerHTML = "";
    statColorsEl.textContent = "0";
    updatePaletteHeader(data.stageNames[0]);
  }

  function updateStats() {
    statStrokesEl.textContent = totalDrawn.toLocaleString();
    statColorsEl.textContent = paletteMap.size.toLocaleString();
  }

  // -------------------------------------------------------------------------
  // Canvas Lifecycle & Time-Paced Smooth Animation Loop (30–60 seconds)
  // -------------------------------------------------------------------------
  function clearArtworkCanvas() {
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
    clearOverlays();
  }

  function commitStrokeInCurrentStage(idx, highlightColor) {
    const arr = data.stages[stageIdx];
    const s = decodeStroke(arr, idx * STRIDE);
    renderStrokeOn(ctx, s, 1.0);
    totalDrawn += 1;
    registerPaletteColor(s.red, s.green, s.blue, highlightColor);
    return s;
  }

  function advanceEmptyStages() {
    while (stageIdx < NUM_STAGES && data.stages[stageIdx].length === 0) {
      stageIdx += 1;
      stageFloat = 0.0;
      committedInStage = 0;
    }
  }

  function stepAnimation(now) {
    if (!isPlaying) return;

    if (lastFrameTime === null) {
      lastFrameTime = now;
    }
    const dt = Math.min(80, Math.max(0, now - lastFrameTime));
    lastFrameTime = now;

    advanceEmptyStages();
    if (stageIdx >= NUM_STAGES) {
      completePlayback();
      return;
    }

    const arr = data.stages[stageIdx];
    const stageCount = arr.length / STRIDE;
    const durationMs = STAGE_DURATIONS_MS[stageIdx] || 5000;
    const strokesPerMs = stageCount / durationMs;

    stageFloat += dt * strokesPerMs;

    let targetCommit = Math.min(stageCount, Math.floor(stageFloat));
    let lastStroke = null;

    for (let i = committedInStage; i < targetCommit; i++) {
      const isLastInBatch = i === targetCommit - 1;
      lastStroke = commitStrokeInCurrentStage(i, isLastInBatch);
    }
    committedInStage = targetCommit;

    // If stage completed this frame, move to next stage cleanly
    if (committedInStage >= stageCount) {
      stageIdx += 1;
      stageFloat = 0.0;
      committedInStage = 0;
      advanceEmptyStages();

      if (stageIdx >= NUM_STAGES) {
        if (lastStroke) {
          updateTopBar(lastStroke.r * 2.0, lastStroke.opacity, lastStroke.red, lastStroke.green, lastStroke.blue, lastStroke.btype === 1);
        }
        completePlayback();
        return;
      }
    }

    // Active in-progress stroke for smooth continuous brush cursor movement along Bezier path
    const curArr = data.stages[stageIdx];
    const curCount = curArr.length / STRIDE;
    const activeIdx = Math.min(curCount - 1, committedInStage);
    const activeStroke = decodeStroke(curArr, activeIdx * STRIDE);
    const frac = Math.max(0.0, Math.min(0.995, stageFloat - committedInStage));

    // Register & highlight the active stroke's color live as the brush touches down
    registerPaletteColor(activeStroke.red, activeStroke.green, activeStroke.blue, true);
    renderLiveOverlay(activeStroke, frac);

    updateTopBar(
      activeStroke.r * 2.0,
      activeStroke.opacity,
      activeStroke.red,
      activeStroke.green,
      activeStroke.blue,
      activeStroke.btype === 1
    );
    updateStageUI(stageIdx);
    updateStats();

    rafId = window.requestAnimationFrame(stepAnimation);
  }

  function completePlayback() {
    isPlaying = false;
    isFinished = true;
    lastFrameTime = null;
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
      rafId = null;
    }
    clearOverlays(); // Hide both canvas & reference cursors when painting finishes
    btnPlayPause.textContent = "Play";
    updateStageUI(NUM_STAGES - 1);
    updateStats();
  }

  function startPainting() {
    if (isFinished) {
      resetPainting(true);
      return;
    }
    if (isPlaying) return;
    setActiveTool("brush");
    isPlaying = true;
    lastFrameTime = null;
    btnPlayPause.textContent = "Pause";
    rafId = window.requestAnimationFrame(stepAnimation);
  }

  function pausePainting() {
    isPlaying = false;
    lastFrameTime = null;
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
      rafId = null;
    }
    btnPlayPause.textContent = "Play";
  }

  function finishImmediately() {
    pausePainting();
    let lastStroke = null;
    while (stageIdx < NUM_STAGES) {
      const arr = data.stages[stageIdx];
      const stageCount = arr.length / STRIDE;
      for (let i = committedInStage; i < stageCount; i++) {
        const isVeryLast = stageIdx === NUM_STAGES - 1 && i === stageCount - 1;
        lastStroke = commitStrokeInCurrentStage(i, isVeryLast);
      }
      stageIdx += 1;
      committedInStage = 0;
      stageFloat = 0.0;
    }
    if (lastStroke) {
      updateTopBar(lastStroke.r * 2.0, lastStroke.opacity, lastStroke.red, lastStroke.green, lastStroke.blue, lastStroke.btype === 1);
    }
    completePlayback();
  }

  function resetPainting(autoStart) {
    pausePainting();
    clearArtworkCanvas();
    clearPalette();
    stageIdx = 0;
    stageFloat = 0.0;
    committedInStage = 0;
    totalDrawn = 0;
    isFinished = false;
    setActiveTool("brush");
    updateStageUI(0);
    updateStats();
    if (autoStart) {
      window.requestAnimationFrame(() => startPainting());
    }
  }

  // -------------------------------------------------------------------------
  // Left Tool Rail & Manual Painting
  // -------------------------------------------------------------------------
  function setActiveTool(toolName) {
    if (toolName === "zoom") {
      zoomActive = !zoomActive;
      canvasStage.classList.toggle("zoomed", zoomActive);
      const zoomBtn = railButtons.find((b) => b.dataset.tool === "zoom");
      if (zoomBtn) {
        zoomBtn.classList.toggle("active", zoomActive);
        zoomBtn.setAttribute("aria-pressed", String(zoomActive));
      }
      fitCanvasToStage();
      return;
    }
    activeTool = toolName;
    railButtons.forEach((btn) => {
      if (btn.dataset.tool === "zoom") return;
      const isMatch = btn.dataset.tool === toolName;
      btn.classList.toggle("active", isMatch);
      btn.setAttribute("aria-pressed", String(isMatch));
    });
  }

  railButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tool = btn.dataset.tool;
      if (tool !== "zoom") {
        pausePainting();
      }
      setActiveTool(tool);
    });
  });

  function getCanvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((evt.clientX - rect.left) / rect.width) * W,
      y: ((evt.clientY - rect.top) / rect.height) * H,
    };
  }

  function syncManualFromInputs() {
    manualSize = parseFloat(brushSizeInput.value) || 24;
    manualOpacity = (parseFloat(brushOpacityInput.value) || 75) / 100;
    manualColor = hexToRgb(colorPickerInput.value);
    updateTopBar(manualSize, manualOpacity, manualColor[0], manualColor[1], manualColor[2], false);
  }

  canvas.addEventListener("pointerdown", (evt) => {
    pausePainting();
    clearOverlays();
    syncManualFromInputs();

    if (activeTool === "fill") {
      const [red, green, blue] = manualColor;
      drawSoftRoundOn(ctx, W * 0.5, H * 0.5, W * 0.5, H * 0.5, W * 0.5, H * 0.5, Math.hypot(W, H), manualOpacity * 0.35, red, green, blue, 1.0);
      totalDrawn += 1;
      registerPaletteColor(red, green, blue, true);
      updateStats();
      updatePaletteHeader();
      return;
    }

    canvas.setPointerCapture(evt.pointerId);
    isPointerDown = true;
    const pt = getCanvasPoint(evt);
    pointerPrev = pt;
    pointerMid = pt;
  });

  canvas.addEventListener("pointermove", (evt) => {
    if (!isPointerDown || !pointerPrev) return;
    const pt = getCanvasPoint(evt);
    const dist = Math.hypot(pt.x - pointerPrev.x, pt.y - pointerPrev.y);
    if (dist < 2.2) return;

    const mid = { x: (pointerPrev.x + pt.x) * 0.5, y: (pointerPrev.y + pt.y) * 0.5 };
    const r = Math.max(0.8, manualSize * 0.5);
    const [red, green, blue] = activeTool === "eraser" ? [255, 255, 255] : manualColor;

    if (activeTool === "pencil") {
      drawHairlineRoundOn(ctx, pointerMid.x, pointerMid.y, pointerPrev.x, pointerPrev.y, mid.x, mid.y, Math.min(r, 3.0), manualOpacity, red, green, blue, 1.0);
    } else if (r < 5.0) {
      drawFineRoundOn(ctx, pointerMid.x, pointerMid.y, pointerPrev.x, pointerPrev.y, mid.x, mid.y, r, manualOpacity, red, green, blue, 1.0);
    } else {
      drawSoftRoundOn(ctx, pointerMid.x, pointerMid.y, pointerPrev.x, pointerPrev.y, mid.x, mid.y, r, manualOpacity, red, green, blue, 1.0);
    }

    pointerPrev = pt;
    pointerMid = mid;
    totalDrawn += 1;
    if (activeTool !== "eraser") {
      registerPaletteColor(red, green, blue, true);
      updatePaletteHeader();
    }
    updateStats();
  });

  function endManualStroke() {
    if (!isPointerDown) return;
    if (pointerPrev && pointerMid && Math.hypot(pointerPrev.x - pointerMid.x, pointerPrev.y - pointerMid.y) < 1.0) {
      const r = Math.max(0.8, manualSize * 0.5);
      const [red, green, blue] = activeTool === "eraser" ? [255, 255, 255] : manualColor;
      drawSoftRoundOn(ctx, pointerPrev.x, pointerPrev.y, pointerPrev.x, pointerPrev.y, pointerPrev.x, pointerPrev.y, r, manualOpacity, red, green, blue, 1.0);
      totalDrawn += 1;
      if (activeTool !== "eraser") {
        registerPaletteColor(red, green, blue, true);
        updatePaletteHeader();
      }
      updateStats();
    }
    isPointerDown = false;
    pointerPrev = null;
    pointerMid = null;
  }

  canvas.addEventListener("pointerup", endManualStroke);
  canvas.addEventListener("pointercancel", endManualStroke);

  brushSizeInput.addEventListener("input", () => {
    pausePainting();
    syncManualFromInputs();
  });

  brushOpacityInput.addEventListener("input", () => {
    pausePainting();
    syncManualFromInputs();
  });

  colorPickerInput.addEventListener("input", () => {
    pausePainting();
    syncManualFromInputs();
  });

  btnPlayPause.addEventListener("click", () => {
    if (isPlaying) pausePainting();
    else startPainting();
  });

  btnFinish.addEventListener("click", finishImmediately);
  btnReset.addEventListener("click", () => resetPainting(true));

  // Inspection hook
  window.__paintApp = {
    finishImmediately,
    resetPainting,
    getState: () => ({
      stageIdx,
      stageName: data.stageNames[Math.min(stageIdx, NUM_STAGES - 1)],
      totalDrawn,
      colorsMixed: paletteMap.size,
      isPlaying,
      isFinished,
    }),
  };

  // Init UI, start completely blank, then paint smoothly across all 9 stages
  buildStagesList();
  const params = new URLSearchParams(window.location.search);
  resetPainting(false);
  if (params.get("blank") === "1") {
    // Stay blank for blank-canvas inspection
  } else if (params.get("instant") === "1") {
    window.requestAnimationFrame(() => finishImmediately());
  } else {
    window.requestAnimationFrame(() => startPainting());
  }
})();
