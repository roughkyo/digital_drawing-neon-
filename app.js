(function () {
  "use strict";

  const STRIDE = 12;
  const STAGE_RATES = [12, 42, 125, 360]; // strokes per animation frame per stage

  const canvas = document.getElementById("paintCanvas");
  const ctx = canvas.getContext("2d");
  const canvasStage = document.getElementById("canvasStage");

  const strokeCounterEl = document.getElementById("strokeCounter");
  const stageLabelEl = document.getElementById("stageLabel");
  const stagePips = Array.from(document.querySelectorAll(".stage-pip"));

  const brushSizeInput = document.getElementById("brushSize");
  const brushSizeValueEl = document.getElementById("brushSizeValue");
  const sizePreviewDot = document.getElementById("sizePreviewDot");

  const brushOpacityInput = document.getElementById("brushOpacity");
  const brushOpacityValueEl = document.getElementById("brushOpacityValue");

  const colorSwatchEl = document.getElementById("colorSwatch");
  const colorPickerInput = document.getElementById("colorPicker");
  const colorHexEl = document.getElementById("colorHex");

  const modeSoftBtn = document.getElementById("modeSoft");
  const modeBristleBtn = document.getElementById("modeBristle");

  const btnPlayPause = document.getElementById("btnPlayPause");
  const btnFinish = document.getElementById("btnFinish");
  const btnReset = document.getElementById("btnReset");

  const data = window.PAINTING_DATA || {
    width: 1280,
    height: 1932,
    stageNames: ["Blocking in", "Form and color", "Refinement", "Fine details"],
    stages: [[], [], [], []],
  };

  canvas.width = data.width;
  canvas.height = data.height;
  const ASPECT = data.width / data.height;

  // State
  let stageIdx = 0;
  let strokeInStage = 0;
  let totalDrawn = 0;
  let isPlaying = false;
  let isFinished = false;
  let rafId = null;

  // Manual brush state
  let manualBrushMode = 0; // 0 = soft round, 1 = bristle
  let manualSize = 24;
  let manualOpacity = 0.75;
  let manualColor = [200, 158, 100];
  let isPointerDown = false;
  let pointerPrev = null;
  let pointerMid = null;

  // -------------------------------------------------------------------------
  // Exact Canvas aspect ratio fitting inside stage container
  // -------------------------------------------------------------------------
  function fitCanvasToStage() {
    const rect = canvasStage.getBoundingClientRect();
    const availW = Math.max(100, rect.width - 4);
    const availH = Math.max(100, rect.height - 4);
    let drawW = availW;
    let drawH = drawW / ASPECT;
    if (drawH > availH) {
      drawH = availH;
      drawW = drawH * ASPECT;
    }
    canvas.style.width = `${drawW.toFixed(2)}px`;
    canvas.style.height = `${drawH.toFixed(2)}px`;
  }

  window.addEventListener("resize", fitCanvasToStage);
  fitCanvasToStage();

  // -------------------------------------------------------------------------
  // Pure Canvas 2D Brush Primitives (NO drawImage, NO putImageData, NO filters)
  // Matches scripts/02_generate_strokes.py exact mathematical models
  // -------------------------------------------------------------------------
  function detHash(seed, k) {
    let x = (Math.imul(seed, 1664525) + Math.imul(k, 1013904223) + 374761393) & 0x7fffffff;
    x = (x ^ (x >>> 13)) & 0x7fffffff;
    x = (Math.imul(x, 48271) + 12345) & 0x7fffffff;
    x = (x ^ (x >>> 16)) & 0x7fffffff;
    return (x & 0xffff) / 32767.5 - 1.0;
  }

  function drawSoftRoundStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue) {
    const approxLen = Math.hypot(cx - x0, cy - y0) + Math.hypot(x1 - cx, y1 - cy);
    const step = Math.max(2.5, r * 0.55);
    const nDabs = Math.max(1, Math.min(10, Math.round(approxLen / step) + 1));
    const dabAlpha = nDabs === 1 ? opacity : 1.0 - Math.pow(Math.max(1e-4, 1.0 - opacity), 0.52);

    const c0 = `rgba(${red},${green},${blue},${dabAlpha.toFixed(4)})`;
    const c1 = `rgba(${red},${green},${blue},${(dabAlpha * 0.72).toFixed(4)})`;
    const c2 = `rgba(${red},${green},${blue},0)`;

    for (let i = 0; i < nDabs; i++) {
      const t = nDabs === 1 ? 0.5 : i / (nDabs - 1);
      const mt = 1.0 - t;
      const px = mt * mt * x0 + 2.0 * mt * t * cx + t * t * x1;
      const py = mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1;

      const grad = ctx.createRadialGradient(px, py, 0, px, py, r);
      grad.addColorStop(0, c0);
      grad.addColorStop(0.55, c1);
      grad.addColorStop(1, c2);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function strokeQuad(x0, y0, cx, cy, x1, y1, lineWidth, strokeStyle) {
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(cx, cy, x1, y1);
    ctx.stroke();
  }

  function drawBristleStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue, seed) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const dlen = Math.hypot(dx, dy);
    const nx = dlen < 1e-4 ? 0.0 : -dy / dlen;
    const ny = dlen < 1e-4 ? 1.0 : dx / dlen;
    const invLen = dlen > 1.0 ? 1.0 / dlen : 1.0;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Soft paint body under bristles
    strokeQuad(
      x0, y0, cx, cy, x1, y1,
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

      strokeQuad(
        x0 + nx * endOff + ux,
        y0 + ny * endOff + uy,
        cx + nx * off,
        cy + ny * off,
        x1 + nx * endOff - ux,
        y1 + ny * endOff - uy,
        filWidth,
        `rgba(${red},${green},${blue},${(opacity * weights[i]).toFixed(4)})`
      );
    }
  }

  function drawFineRoundStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    strokeQuad(
      x0, y0, cx, cy, x1, y1,
      r * 2.1,
      `rgba(${red},${green},${blue},${(opacity * 0.38).toFixed(4)})`
    );
    strokeQuad(
      x0, y0, cx, cy, x1, y1,
      r * 1.24,
      `rgba(${red},${green},${blue},${(opacity * 0.78).toFixed(4)})`
    );
  }

  function drawHairlineRoundStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    strokeQuad(
      x0, y0, cx, cy, x1, y1,
      Math.max(0.75, r * 1.48),
      `rgba(${red},${green},${blue},${opacity.toFixed(4)})`
    );
  }

  function drawEncodedStroke(arr, offset) {
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

    const x0 = ix0 * 0.5;
    const y0 = iy0 * 0.5;
    const cx = icx * 0.5;
    const cy = icy * 0.5;
    const x1 = ix1 * 0.5;
    const y1 = iy1 * 0.5;

    if (btype === 0) {
      drawSoftRoundStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue);
    } else if (btype === 1) {
      const seed =
        (Math.imul(ix0, 31) + Math.imul(iy0, 131) + Math.imul(ix1, 1009) + Math.imul(iy1, 4093)) &
        0x7fffffff;
      drawBristleStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue, seed);
    } else if (btype === 2) {
      drawFineRoundStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue);
    } else {
      drawHairlineRoundStroke(x0, y0, cx, cy, x1, y1, r, opacity, red, green, blue);
    }

    return { btype, r, opacity, red, green, blue };
  }

  // -------------------------------------------------------------------------
  // HUD updates
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

  function updateToolbarDisplay(sizePx, opacity01, red, green, blue, brushMode) {
    const clampedSize = Math.max(1, Math.min(140, sizePx));
    brushSizeInput.value = clampedSize.toFixed(1);
    brushSizeValueEl.textContent = clampedSize < 10 ? `${clampedSize.toFixed(1)} px` : `${Math.round(clampedSize)} px`;

    const dotPx = Math.max(3, Math.min(18, 2 + Math.sqrt(clampedSize) * 1.4));
    sizePreviewDot.style.width = `${dotPx.toFixed(1)}px`;
    sizePreviewDot.style.height = `${dotPx.toFixed(1)}px`;

    const opPct = Math.max(5, Math.min(100, Math.round(opacity01 * 100)));
    brushOpacityInput.value = String(opPct);
    brushOpacityValueEl.textContent = `${opPct}%`;

    const hex = rgbToHex(red, green, blue);
    colorSwatchEl.style.background = hex;
    colorPickerInput.value = hex.toLowerCase();
    colorHexEl.textContent = hex;

    const isBristle = brushMode === 1;
    modeSoftBtn.classList.toggle("active", !isBristle);
    modeBristleBtn.classList.toggle("active", isBristle);
  }

  function updateStageDisplay(idx) {
    const safeIdx = Math.min(idx, data.stageNames.length - 1);
    stageLabelEl.textContent = data.stageNames[safeIdx];
    stagePips.forEach((pip, i) => {
      pip.classList.toggle("done", i < safeIdx || isFinished);
      pip.classList.toggle("active", !isFinished && i === safeIdx);
    });
  }

  function updateCounter() {
    strokeCounterEl.textContent = `${totalDrawn.toLocaleString()} ${totalDrawn === 1 ? "stroke" : "strokes"}`;
  }

  // -------------------------------------------------------------------------
  // Canvas lifecycle & animated painting loop
  // -------------------------------------------------------------------------
  function clearCanvas() {
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  function advanceToNextNonEmptyStage() {
    while (stageIdx < data.stages.length) {
      const count = data.stages[stageIdx].length / STRIDE;
      if (strokeInStage < count) return true;
      stageIdx += 1;
      strokeInStage = 0;
    }
    return false;
  }

  function stepAnimation() {
    if (!isPlaying) return;

    if (!advanceToNextNonEmptyStage()) {
      isPlaying = false;
      isFinished = true;
      btnPlayPause.textContent = "Play";
      updateStageDisplay(data.stageNames.length - 1);
      updateCounter();
      return;
    }

    const activeStage = stageIdx;
    const arr = data.stages[activeStage];
    const stageCount = arr.length / STRIDE;
    const rate = STAGE_RATES[activeStage] || 60;
    const endIdx = Math.min(stageCount, strokeInStage + rate);

    let lastInfo = null;
    for (let i = strokeInStage; i < endIdx; i++) {
      lastInfo = drawEncodedStroke(arr, i * STRIDE);
      totalDrawn += 1;
    }
    strokeInStage = endIdx;

    if (lastInfo) {
      updateToolbarDisplay(
        lastInfo.r * 2.0,
        lastInfo.opacity,
        lastInfo.red,
        lastInfo.green,
        lastInfo.blue,
        lastInfo.btype === 1 ? 1 : 0
      );
    }
    updateStageDisplay(activeStage);
    updateCounter();

    rafId = window.requestAnimationFrame(stepAnimation);
  }

  function startPainting() {
    if (isFinished) {
      resetPainting(true);
      return;
    }
    if (isPlaying) return;
    isPlaying = true;
    btnPlayPause.textContent = "Pause";
    rafId = window.requestAnimationFrame(stepAnimation);
  }

  function pausePainting() {
    isPlaying = false;
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
      rafId = null;
    }
    btnPlayPause.textContent = "Play";
  }

  function finishImmediately() {
    pausePainting();
    let lastInfo = null;
    while (advanceToNextNonEmptyStage()) {
      const arr = data.stages[stageIdx];
      const stageCount = arr.length / STRIDE;
      for (let i = strokeInStage; i < stageCount; i++) {
        lastInfo = drawEncodedStroke(arr, i * STRIDE);
        totalDrawn += 1;
      }
      strokeInStage = stageCount;
    }
    isFinished = true;
    if (lastInfo) {
      updateToolbarDisplay(
        lastInfo.r * 2.0,
        lastInfo.opacity,
        lastInfo.red,
        lastInfo.green,
        lastInfo.blue,
        lastInfo.btype === 1 ? 1 : 0
      );
    }
    updateStageDisplay(data.stageNames.length - 1);
    updateCounter();
    btnPlayPause.textContent = "Play";
  }

  function resetPainting(autoStart) {
    pausePainting();
    clearCanvas();
    stageIdx = 0;
    strokeInStage = 0;
    totalDrawn = 0;
    isFinished = false;
    updateCounter();
    updateStageDisplay(0);
    if (autoStart) {
      // Wait one paint frame so the blank canvas is visibly presented first
      window.requestAnimationFrame(() => startPainting());
    }
  }

  // -------------------------------------------------------------------------
  // Manual painting support (works seamlessly as a real paint program)
  // -------------------------------------------------------------------------
  function getCanvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((evt.clientX - rect.left) / rect.width) * canvas.width,
      y: ((evt.clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  function syncManualFromInputs() {
    manualSize = parseFloat(brushSizeInput.value) || 24;
    manualOpacity = (parseFloat(brushOpacityInput.value) || 75) / 100;
    manualColor = hexToRgb(colorPickerInput.value);
    updateToolbarDisplay(
      manualSize,
      manualOpacity,
      manualColor[0],
      manualColor[1],
      manualColor[2],
      manualBrushMode
    );
  }

  canvas.addEventListener("pointerdown", (evt) => {
    pausePainting();
    syncManualFromInputs();
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
    if (dist < 2.5) return;

    const mid = { x: (pointerPrev.x + pt.x) * 0.5, y: (pointerPrev.y + pt.y) * 0.5 };
    const r = Math.max(0.8, manualSize * 0.5);
    const [red, green, blue] = manualColor;

    if (manualBrushMode === 1) {
      const seed = (totalDrawn * 131 + Math.round(pt.x) * 31 + Math.round(pt.y) * 17) & 0x7fffffff;
      drawBristleStroke(pointerMid.x, pointerMid.y, pointerPrev.x, pointerPrev.y, mid.x, mid.y, r, manualOpacity, red, green, blue, seed);
    } else if (r < 5.0) {
      drawFineRoundStroke(pointerMid.x, pointerMid.y, pointerPrev.x, pointerPrev.y, mid.x, mid.y, r, manualOpacity, red, green, blue);
    } else {
      drawSoftRoundStroke(pointerMid.x, pointerMid.y, pointerPrev.x, pointerPrev.y, mid.x, mid.y, r, manualOpacity, red, green, blue);
    }

    pointerPrev = pt;
    pointerMid = mid;
    totalDrawn += 1;
    updateCounter();
  });

  function endManualStroke() {
    if (!isPointerDown) return;
    if (pointerPrev && pointerMid && Math.hypot(pointerPrev.x - pointerMid.x, pointerPrev.y - pointerMid.y) < 1.0) {
      const r = Math.max(0.8, manualSize * 0.5);
      const [red, green, blue] = manualColor;
      drawSoftRoundStroke(pointerPrev.x, pointerPrev.y, pointerPrev.x, pointerPrev.y, pointerPrev.x, pointerPrev.y, r, manualOpacity, red, green, blue);
      totalDrawn += 1;
      updateCounter();
    }
    isPointerDown = false;
    pointerPrev = null;
    pointerMid = null;
  }

  canvas.addEventListener("pointerup", endManualStroke);
  canvas.addEventListener("pointercancel", endManualStroke);

  // Controls
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

  modeSoftBtn.addEventListener("click", () => {
    manualBrushMode = 0;
    modeSoftBtn.classList.add("active");
    modeBristleBtn.classList.remove("active");
  });

  modeBristleBtn.addEventListener("click", () => {
    manualBrushMode = 1;
    modeBristleBtn.classList.add("active");
    modeSoftBtn.classList.remove("active");
  });

  btnPlayPause.addEventListener("click", () => {
    if (isPlaying) pausePainting();
    else startPainting();
  });

  btnFinish.addEventListener("click", finishImmediately);
  btnReset.addEventListener("click", () => resetPainting(true));

  // Expose tiny inspection hooks for verification without affecting visuals
  window.__paintApp = {
    finishImmediately,
    resetPainting,
    getState: () => ({
      stageIdx,
      stageName: data.stageNames[Math.min(stageIdx, data.stageNames.length - 1)],
      totalDrawn,
      isPlaying,
      isFinished,
    }),
  };

  // Start completely blank, then paint live from Stage 1
  const params = new URLSearchParams(window.location.search);
  resetPainting(false);
  if (params.get("blank") === "1") {
    // Stay blank if explicitly requested for blank-start inspection
  } else if (params.get("instant") === "1") {
    window.requestAnimationFrame(() => finishImmediately());
  } else {
    window.requestAnimationFrame(() => startPainting());
  }
})();
