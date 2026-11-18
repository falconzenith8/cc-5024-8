"use strict";

const state = {
  session: null,
  full: { w: 0, h: 0 },
  roi: null,          // [x, y, w, h] in source pixels
  analysis: null,
  analysisRoi: null,   // [x, y, w, h] absolute source coords of the analyzed crop
  manual: [],         // {period_px, angle_deg}
  rejected: [],       // {period_px, angle_deg}
  result: null,
  resultImages: null,
  regions: [],        // zone parameters, page coordinates
  dirtyRegions: new Set(),
  regionCounter: 0,
  regionResult: null,
  regionImages: null,
  suggestion: null,
};

const $ = (id) => document.getElementById(id);

function banner(msg, kind = "ok") {
  const el = $("banner");
  el.textContent = msg;
  el.className = "banner " + kind;
  el.classList.remove("hidden");
  if (kind === "ok") {
    clearTimeout(banner._t);
    banner._t = setTimeout(() => el.classList.add("hidden"), 5000);
  }
}

function clearBanner() {
  $("banner").classList.add("hidden");
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = typeof data === "object" && data.error ? data.error : String(data);
    throw new Error(msg);
  }
  return data;
}

// ---------- source loading ----------

async function loadFixtures() {
  const { fixtures } = await api("/api/fixtures");
  const sel = $("fixture-select");
  sel.innerHTML = "";
  for (const f of fixtures) {
    const o = document.createElement("option");
    o.value = f.id;
    o.textContent = f.label;
    sel.appendChild(o);
  }
}

async function loadFixture() {
  clearBanner();
  const fixture = $("fixture-select").value;
  const info = await api("/api/load_fixture", { fixture });
  await adoptSession(info);
}

async function uploadFile(file) {
  clearBanner();
  const res = await fetch("/api/upload", { method: "POST", body: file });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "upload failed");
  await adoptSession(data);
}

async function adoptSession(info) {
  state.session = info.session;
  state.full = { w: info.shape[1], h: info.shape[0] };
  state.roi = null;
  state.analysis = null;
  state.analysisRoi = null;
  state.result = null;
  state.resultImages = null;
  state.regions = [];
  state.dirtyRegions = new Set();
  state.regionResult = null;
  state.regionImages = null;
  state.suggestion = null;
  state.manual = [];
  state.rejected = [];
  drawSource();
  const truth = info.truth && Object.keys(info.truth).length
    ? " · " + JSON.stringify(info.truth) : "";
  $("source-meta").textContent =
    `${info.kind} · ${info.shape[1]}×${info.shape[0]} · ${info.channels} channel(s)${truth}`;
  $("analyze").disabled = false;
  $("run").disabled = true;
  $("run-regions").disabled = true;
  $("analysis-summary").textContent = "Not analyzed yet.";
  $("warnings").innerHTML = "";
  $("peaks-table").querySelector("tbody").innerHTML = "";
  $("lattice-explained").textContent = "";
  $("result-summary").textContent = "Not run yet.";
  $("metrics-table").querySelector("tbody").innerHTML = "";
  $("audit").textContent = "";
  $("weight-readout").textContent = "";
  $("suggest-notes").innerHTML = "";
  $("region-warnings").innerHTML = "";
  $("weight-overlay-canvas").classList.add("hidden");
  $("weight-overlay-canvas").removeAttribute("src");
  $("weights-figure").classList.add("hidden");
  $("download-weights").classList.add("hidden");
  $("show-weights").checked = false;
  renderRegions();
  for (const id of ["overview-img", "spectrum-img", "result-input",
                    "result-suppressed", "result-output", "result-difference",
                    "result-weights"]) {
    $(id).removeAttribute("src");
  }
  renderManualList();
}

// ---------- source canvas + ROI ----------

const canvas = () => $("source-canvas");

function drawSource() {
  if (!state.session) return;
  const img = new Image();
  img.onload = () => {
    const c = canvas();
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0);
    state.display = { w: img.naturalWidth, h: img.naturalHeight };
    drawRoiBox();
    drawRegionBoxes();
  };
  img.src = `/api/image?session=${encodeURIComponent(state.session)}`;
}

let drag = null;

canvas().addEventListener("pointerdown", (e) => {
  if (!state.session) return;
  const r = canvas().getBoundingClientRect();
  drag = {
    x0: (e.clientX - r.left) * canvas().width / r.width,
    y0: (e.clientY - r.top) * canvas().height / r.height,
  };
  canvas().setPointerCapture(e.pointerId);
});

canvas().addEventListener("pointermove", (e) => {
  if (!drag) {
    if (state.regionResult) {
      const r = canvas().getBoundingClientRect();
      const px = (e.clientX - r.left) * state.full.w / r.width;
      const py = (e.clientY - r.top) * state.full.h / r.height;
      updateWeightReadout(px, py);
    }
    return;
  }
  const r = canvas().getBoundingClientRect();
  const x1 = (e.clientX - r.left) * canvas().width / r.width;
  const y1 = (e.clientY - r.top) * canvas().height / r.height;
  const box = $("roi-box");
  box.classList.remove("hidden");
  box.style.left = Math.min(drag.x0, x1) + "px";
  box.style.top = Math.min(drag.y0, y1) + "px";
  box.style.width = Math.abs(x1 - drag.x0) + "px";
  box.style.height = Math.abs(y1 - drag.y0) + "px";
});

canvas().addEventListener("pointerup", (e) => {
  if (!drag) return;
  const r = canvas().getBoundingClientRect();
  const x1 = (e.clientX - r.left) * canvas().width / r.width;
  const y1 = (e.clientY - r.top) * canvas().height / r.height;
  const scale = state.full.w / canvas().width;
  const x = Math.round(Math.min(drag.x0, x1) * scale);
  const y = Math.round(Math.min(drag.y0, y1) * scale);
  const w = Math.round(Math.abs(x1 - drag.x0) * scale);
  const h = Math.round(Math.abs(y1 - drag.y0) * scale);
  drag = null;
  if (w < 32 || h < 32) {
    state.roi = null;
    $("roi-box").classList.add("hidden");
    banner("Selection too small; using the whole image.", "error");
    return;
  }
  state.roi = [x, y, w, h];
  drawRoiBox();
});

function drawRoiBox() {
  const box = $("roi-box");
  if (!state.roi || !state.display) {
    box.classList.add("hidden");
    return;
  }
  const sx = canvas().getBoundingClientRect().width / state.full.w;
  const sy = canvas().getBoundingClientRect().height / state.full.h;
  const [x, y, w, h] = state.roi;
  box.classList.remove("hidden");
  box.style.left = (x * sx) + "px";
  box.style.top = (y * sy) + "px";
  box.style.width = (w * sx) + "px";
  box.style.height = (h * sy) + "px";
}

window.addEventListener("resize", () => {
  drawRoiBox();
  drawRegionBoxes();
  drawRegionWeights();
});

// ---------- zones ----------

const REGION_STATUS_CLASS = {
  candidate: "fallback",
  pending: "fallback",
  confirmed: "ok",
  manual: "ok",
  ok: "ok",
  no_screen_detected: "fallback",
  too_small_for_auto_analysis: "fallback",
  outside_image: "failed",
  error: "failed",
};

function peaksToText(peaks) {
  return (peaks || []).map((p) =>
    `${Number(p.period_px).toFixed(2)}@${Number(p.angle_deg).toFixed(1)}`).join(", ");
}

function parsePeaksText(text) {
  const out = [];
  for (const part of text.split(",")) {
    const t = part.trim();
    if (!t) continue;
    const m = t.match(/^([\d.]+)\s*[@/]\s*([\d.]+)$/);
    if (!m) return null;
    const period = Number(m[1]);
    const angle = Number(m[2]);
    if (!(period >= 2.5 && period <= 64) || !(angle >= 0 && angle <= 180)) return null;
    out.push({ period_px: period, angle_deg: angle, source: "operator" });
  }
  return out;
}

function newRegion(rect) {
  state.regionCounter += 1;
  return {
    id: `zone-${state.regionCounter}`,
    name: `Zone ${state.regionCounter}`,
    rect: rect.map((v) => Math.round(v)),
    feather_px: 28,
    opacity: 1,
    enabled: true,
    status: "manual",
    auto_peaks: false,
    peaks: [],
    seed_peaks: [],
    depends_on: [],
    params: {
      method: "frequency_notch",
      strength: 1,
      radius_cyc_per_px: null,
      apodization: 0.35,
      harmonic_orders: [1, 2],
      include_lattice_lines: true,
    },
    notes: [],
  };
}

function markRegionDirty(id) {
  state.dirtyRegions.add(id);
  $("recompute-regions").disabled = false;
}

function regionById(id) {
  return state.regions.find((r) => r.id === id);
}

function addRegion() {
  if (state.regions.length >= 24) {
    banner("At most 24 zones are supported.", "error");
    return;
  }
  const rect = state.roi ? [...state.roi] : [0, 0, state.full.w, state.full.h];
  const r = newRegion(rect);
  state.regions.push(r);
  markRegionDirty(r.id);
  renderRegions();
  banner(`Added ${r.id}. Set its screens or use "Suggest zones" and recompute.`);
}

function clearRegions() {
  state.regions = [];
  state.dirtyRegions.clear();
  state.regionResult = null;
  state.regionImages = null;
  $("region-warnings").innerHTML = "";
  $("suggest-notes").innerHTML = "";
  $("recompute-regions").disabled = true;
  $("run-regions").disabled = true;
  $("weight-readout").textContent = "";
  $("show-weights").checked = false;
  hideWeightOverlay();
  $("weights-figure").classList.add("hidden");
  $("download-weights").classList.add("hidden");
  renderRegions();
}

function hideWeightOverlay() {
  const el = $("weight-overlay-canvas");
  el.classList.add("hidden");
  el.removeAttribute("src");
}

const numberCell = (value, step, min, max, onChange, cls) => {
  const inp = document.createElement("input");
  inp.type = "number";
  inp.step = String(step);
  inp.min = String(min);
  inp.max = String(max);
  inp.value = String(value);
  if (cls) inp.className = cls;
  inp.addEventListener("change", () => onChange(Number(inp.value)));
  return inp;
};

function renderRegions() {
  const tbody = $("regions-table").querySelector("tbody");
  tbody.innerHTML = "";
  state.regions.forEach((r) => {
    const tr = document.createElement("tr");

    const tdOn = document.createElement("td");
    const on = document.createElement("input");
    on.type = "checkbox";
    on.checked = r.enabled;
    on.addEventListener("change", () => {
      r.enabled = on.checked;
      markRegionDirty(r.id);
      drawRegionBoxes();
    });
    tdOn.appendChild(on);

    const tdName = document.createElement("td");
    const name = document.createElement("input");
    name.type = "text";
    name.value = r.name || r.id;
    name.addEventListener("change", () => { r.name = name.value; drawRegionBoxes(); });
    tdName.appendChild(name);

    const tdRect = document.createElement("td");
    tdRect.className = "rect-cell";
    ["x", "y", "w", "h"].forEach((label, i) => {
      const inp = numberCell(r.rect[i], 1, i < 2 ? 0 : 2, 20000, (v) => {
      r.rect[i] = Math.round(v);
      markRegionDirty(r.id);
      drawRegionBoxes();
    });
      inp.title = label;
      tdRect.appendChild(inp);
    });

    const tdFeather = document.createElement("td");
    tdFeather.appendChild(numberCell(r.feather_px, 2, 0, 512, (v) => {
      r.feather_px = v;
      markRegionDirty(r.id);
    }));

    const tdOpacity = document.createElement("td");
    tdOpacity.appendChild(numberCell(r.opacity, 0.05, 0, 1, (v) => {
      r.opacity = v;
      markRegionDirty(r.id);
    }));

    const tdStrength = document.createElement("td");
    tdStrength.appendChild(numberCell(r.params.strength, 0.05, 0, 1, (v) => {
      r.params.strength = v;
      markRegionDirty(r.id);
    }));

    const tdMethod = document.createElement("td");
    const method = document.createElement("select");
    for (const [v, label] of [["frequency_notch", "notch"],
                              ["spatial_fir", "FIR"]]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      method.appendChild(o);
    }
    method.value = r.params.method;
    method.addEventListener("change", () => {
      r.params.method = method.value;
      markRegionDirty(r.id);
    });
    tdMethod.appendChild(method);

    const tdStatus = document.createElement("td");
    const st = document.createElement("span");
    const statusCls = REGION_STATUS_CLASS[r.status] || "";
    st.className = "region-status " + statusCls;
    const analysis = r.analysis_status && r.analysis_status !== r.status
      ? ` (analysis: ${r.analysis_status})` : "";
    st.textContent = (r.status || "—") + analysis;
    st.title = (r.notes || []).join("\n") +
      (r.analysis_source ? `\nsource: ${r.analysis_source}` : "");
    tdStatus.appendChild(st);
    if (r.status === "candidate" || r.status === "pending") {
      const confirmBtn = document.createElement("button");
      confirmBtn.textContent = "confirm";
      confirmBtn.style.marginLeft = "6px";
      confirmBtn.addEventListener("click", () => {
        r.status = "confirmed";
        r.requires_confirmation = false;
        renderRegions();
      });
      tdStatus.appendChild(confirmBtn);
    }

    const tdDel = document.createElement("td");
    const del = document.createElement("button");
    del.textContent = "delete";
    del.addEventListener("click", () => {
      state.regions = state.regions.filter((q) => q.id !== r.id);
      state.dirtyRegions.delete(r.id);
      renderRegions();
      drawRegionBoxes();
    });
    tdDel.appendChild(del);

    for (const td of [tdOn, tdName, tdRect, tdFeather, tdOpacity, tdStrength,
                      tdMethod, tdStatus, tdDel]) {
      tr.appendChild(td);
    }
    tbody.appendChild(tr);

    // Screens row: per-zone period/angle, editable in place.
    const trPeaks = document.createElement("tr");
    trPeaks.className = "region-peaks-row";
    const tdLabel = document.createElement("td");
    trPeaks.appendChild(tdLabel);
    const tdPeaks = document.createElement("td");
    tdPeaks.colSpan = 8;
    const label = document.createElement("span");
    label.className = "meta";
    label.textContent = "screens: ";
    const peaksInput = document.createElement("input");
    peaksInput.type = "text";
    peaksInput.style.width = "100%";
    peaksInput.placeholder = "period@angle, e.g. 6.00@33.0, 9.50@12.0";
    peaksInput.value = peaksToText(r.peaks);
    peaksInput.addEventListener("change", () => {
      const parsed = parsePeaksText(peaksInput.value);
      if (parsed === null) {
        banner(`Zone ${r.id}: screens must be "period@angle" pairs inside the valid ranges.`, "error");
        peaksInput.value = peaksToText(r.peaks);
        return;
      }
      r.peaks = parsed;
      r.auto_peaks = false;
      if (!parsed.length) r.status = "manual";
      markRegionDirty(r.id);
      renderRegions();
    });
    tdPeaks.appendChild(label);
    tdPeaks.appendChild(peaksInput);
    trPeaks.appendChild(tdPeaks);
    tbody.appendChild(trPeaks);
  });

  const has = state.regions.length > 0;
  $("run-regions").disabled = !has || !state.session;
  if (!has) $("recompute-regions").disabled = true;
  drawRegionBoxes();
  drawRegionWeights();
}

function drawRegionBoxes() {
  const host = $("region-boxes");
  host.innerHTML = "";
  if (!state.session || !state.regions.length) return;
  const c = canvas();
  const rectCss = c.getBoundingClientRect();
  if (!rectCss.width) return;
  const sx = rectCss.width / state.full.w;
  const sy = rectCss.height / state.full.h;
  state.regions.forEach((r) => {
    const box = document.createElement("div");
    box.className = "region-box" + (r.enabled ? "" : " disabled");
    box.style.left = (r.rect[0] * sx) + "px";
    box.style.top = (r.rect[1] * sy) + "px";
    box.style.width = (r.rect[2] * sx) + "px";
    box.style.height = (r.rect[3] * sy) + "px";
    const tag = document.createElement("span");
    tag.className = "region-label";
    tag.textContent = `${r.name || r.id} · f${fmt(r.feather_px, 0)} · ` +
      `s${fmt(r.params.strength, 2)}`;
    box.appendChild(tag);
    host.appendChild(box);
  });
}

function drawRegionWeights() {
  const el = $("weight-overlay-canvas");
  if (!state.regionImages || !$("show-weights").checked) {
    el.classList.add("hidden");
    return;
  }
  const src = state.regionImages["weight_overlay.png"];
  if (!src) return;
  // The overlay covers exactly the processed crop; place it over that rect.
  const roi = (state.regionResult && state.regionResult.roi) || [0, 0, state.full.w, state.full.h];
  const c = canvas();
  const rectCss = c.getBoundingClientRect();
  const sx = rectCss.width / state.full.w;
  const sy = rectCss.height / state.full.h;
  el.style.left = (roi[0] * sx) + "px";
  el.style.top = (roi[1] * sy) + "px";
  el.style.width = (roi[2] * sx) + "px";
  el.style.height = (roi[3] * sy) + "px";
  el.src = "data:image/png;base64," + src;
  el.classList.remove("hidden");
}

function weightAt(pageX, pageY) {
  const res = state.regionResult;
  if (!res || !res.weights || !res.weights.grid) return null;
  const roi = res.roi;
  if (pageX < roi[0] || pageY < roi[1] ||
      pageX >= roi[0] + roi[2] || pageY >= roi[1] + roi[3]) return null;
  const g = res.weights.grid;
  const gh = g.shape[0], gw = g.shape[1];
  const gx = Math.min(gw - 1, Math.floor((pageX - roi[0]) / roi[2] * gw));
  const gy = Math.min(gh - 1, Math.floor((pageY - roi[1]) / roi[3] * gh));
  const per = {};
  for (const r of res.regions) {
    const grid = g.per_region[r.id];
    if (grid) per[r.id] = grid[gy][gx];
  }
  return {
    region_id: null,
    effective_strength: g.effective_strength[gy][gx],
    base: g.base[gy][gx],
    per_region: per,
  };
}

function updateWeightReadout(pageX, pageY) {
  const el = $("weight-readout");
  const w = weightAt(pageX, pageY);
  if (!w) { el.textContent = ""; return; }
  const parts = state.regions
    .filter((r) => w.per_region[r.id] !== undefined && w.per_region[r.id] > 0.005)
    .map((r) => `${r.name || r.id} ${fmt(w.per_region[r.id], 2)}`);
  const base = Number($("base-strength").value) || 0;
  const uncovered = w.base > 0.999 - 1e-6;
  let where;
  if (w.effective_strength < 0.005) {
    where = uncovered
      ? " · no zone covers this pixel and base strength is 0, so no suppression is applied here"
      : " · weights are near zero here, so effectively no suppression";
  } else if (parts.length) {
    where = ` · constituent zone weights: ${parts.join(", ")}`;
  } else {
    where = " · outside every zone, filtered by the base layer only";
  }
  el.textContent =
    `At (${Math.round(pageX)}, ${Math.round(pageY)}): effective strength ` +
    `${fmt(w.effective_strength, 2)} (base layer ${fmt(w.base, 2)} × ` +
    `strength ${fmt(base, 2)})` + where;
}

// ---------- analysis ----------

function pct(x) {
  return (100 * x).toFixed(1) + "%";
}

function fmt(x, d = 2) {
  return (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
}

function classifyTag(p) {
  const c = p.classification || "";
  if (p.selected && c === "halftone_screen") return ["screen", "screen"];
  if (c === "manual") return ["manual", "manual"];
  if (c === "manual_override") return ["manual", "override"];
  if (c === "manually_rejected") return ["rejected", "rejected"];
  if (c === "ambiguous_periodic") return ["ambiguous", "ambiguous"];
  if (c === "real_texture") return ["texture", "real texture"];
  return ["", c || "—"];
}

async function analyze() {
  clearBanner();
  const btn = $("analyze");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  try {
    const data = await api("/api/analyze", {
      session: state.session,
      roi: state.roi,
      deep_channels: $("deep-channels").checked,
      flat_tile_assist: $("flat-assist").checked,
    });
    state.analysis = data.analysis;
    state.analysisRoi = data.roi_absolute;
    $("overview-img").src = "data:image/png;base64," + data.images.overview;
    $("spectrum-img").src = "data:image/png;base64," + data.images.spectrum;
    renderAnalysis();
    $("run").disabled = false;
    const n = data.analysis.peaks.filter((p) => p.selected).length;
    banner(`Analysis complete: ${data.analysis.peaks.length} periodic candidate(s), ` +
           `${n} auto-selected as screens.`);
  } catch (err) {
    banner("Analysis failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze region";
  }
}

function renderAnalysis() {
  const a = state.analysis;
  const roi = state.analysisRoi || a.roi;
  $("analysis-summary").textContent =
    `Region ${roi[2]}×${roi[3]} at (${roi[0]}, ${roi[1]}) · analysis scale ` +
    `${fmt(a.analysis_scale, 3)} · auto-selected ${a.auto_selected}` +
    (a.flat_tile_assist_used ? " · flat-patch assist used" : "");

  const warn = $("warnings");
  warn.innerHTML = "";
  for (const w of a.warnings) {
    const d = document.createElement("div");
    d.className = "warning";
    d.textContent = w;
    warn.appendChild(d);
  }

  const tbody = $("peaks-table").querySelector("tbody");
  tbody.innerHTML = "";
  a.peaks.forEach((p, i) => {
    const tr = document.createElement("tr");
    const [cls, label] = classifyTag(p);

    const tdUse = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!p.selected;
    cb.dataset.index = String(i);
    cb.addEventListener("change", () => togglePeak(i, cb.checked));
    tdUse.appendChild(cb);

    const tdPer = document.createElement("td");
    tdPer.className = "num";
    tdPer.textContent = fmt(p.period_px, 3);

    const tdAng = document.createElement("td");
    tdAng.className = "num";
    const disp = p.display_angle_deg !== null && p.display_angle_deg !== undefined
      ? `${fmt(p.display_angle_deg, 1)} (axis ${fmt(p.angle_deg, 1)})`
      : fmt(p.angle_deg, 1);
    tdAng.textContent = disp;

    const tdCls = document.createElement("td");
    const span = document.createElement("span");
    span.className = "tag " + cls;
    span.textContent = label;
    tdCls.appendChild(span);

    const tdConf = document.createElement("td");
    tdConf.className = "num";
    tdConf.textContent = fmt(p.confidence, 2);

    const tdCoh = document.createElement("td");
    tdCoh.className = "num";
    tdCoh.textContent = fmt(p.coherence, 2);

    const tdCh = document.createElement("td");
    tdCh.textContent = (p.channels || []).join(", ");

    const tdWhy = document.createElement("td");
    tdWhy.className = "reasons";
    tdWhy.textContent = (p.reasons || []).join("; ");

    for (const td of [tdUse, tdPer, tdAng, tdCls, tdConf, tdCoh, tdCh, tdWhy]) {
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  });

  const le = $("lattice-explained");
  if (a.lattice_explained && a.lattice_explained.length) {
    le.textContent = "Lattice lines merged into a parent screen (not treated as " +
      "separate screens): " + a.lattice_explained.map((e) =>
        `${fmt(e.period_px, 2)} px / ${fmt(e.angle_deg, 1)}° → ` +
        `${fmt(e.explained_by_period_px, 2)} px / ${fmt(e.explained_by_angle_deg, 1)}° ` +
        `[m,n]=${e.lattice_order.join(",")}`).join(" · ");
  } else {
    le.textContent = "";
  }
}

function togglePeak(index, selected) {
  const p = state.analysis.peaks[index];
  p.selected = selected;
  if (!selected) {
    state.rejected.push({ period_px: p.period_px, angle_deg: p.angle_deg });
  } else {
    state.rejected = state.rejected.filter((r) =>
      Math.abs(r.period_px - p.period_px) / p.period_px > 0.05 ||
      Math.abs(r.angle_deg - p.angle_deg) > 5);
  }
}

// ---------- manual peaks ----------

function renderManualList() {
  const ul = $("manual-list");
  ul.innerHTML = "";
  state.manual.forEach((m, i) => {
    const li = document.createElement("li");
    li.textContent = `${fmt(m.period_px, 2)} px at ${fmt(m.angle_deg, 1)}° `;
    const b = document.createElement("button");
    b.textContent = "remove";
    b.style.marginLeft = "8px";
    b.addEventListener("click", () => {
      state.manual.splice(i, 1);
      renderManualList();
    });
    li.appendChild(b);
    ul.appendChild(li);
  });
}

function addManual() {
  const period = parseFloat($("manual-period").value);
  const angle = parseFloat($("manual-angle").value);
  if (!(period >= 2.5 && period <= 64)) {
    banner("Manual period must be between 2.5 and 64 px.", "error");
    return;
  }
  if (!(angle >= 0 && angle <= 180)) {
    banner("Manual angle must be between 0 and 180 degrees.", "error");
    return;
  }
  state.manual.push({ period_px: period, angle_deg: angle });
  $("manual-period").value = "";
  $("manual-angle").value = "";
  renderManualList();
}

// ---------- run ----------

function readParams() {
  const harmonics = $("harmonics").value.split(",").map((v) => parseInt(v, 10));
  const numOrNull = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return {
    method: $("method").value,
    scale: Number($("scale").value),
    radius_cyc_per_px: numOrNull("radius"),
    apodization: Number($("apodization").value),
    strength: Number($("strength").value),
    harmonic_orders: harmonics,
    pad: numOrNull("pad"),
    feather: numOrNull("feather"),
    include_lattice_lines: $("lattice-lines").checked,
    force_peaks: $("force-peaks").checked,
    deep_channels: $("deep-channels").checked,
    flat_tile_assist: $("flat-assist").checked,
  };
}

async function runPipeline() {
  clearBanner();
  const btn = $("run");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const data = await api("/api/run", {
      session: state.session,
      roi: state.roi,
      params: readParams(),
      manual_peaks: state.manual,
      reject_peaks: state.rejected,
    });
    state.result = data.result;
    state.resultImages = data.images;
    state.lastParamsJson = data.params_json;
    state.regionResult = null;
    state.regionImages = null;
    $("weights-figure").classList.add("hidden");
    $("download-weights").classList.add("hidden");
    $("weight-readout").textContent = "";
    $("show-weights").checked = false;
    hideWeightOverlay();
    for (const k of ["input", "suppressed", "output", "difference"]) {
      $(`result-${k}`).src = "data:image/png;base64," + data.images[k + ".png"];
    }
    renderResult();
    banner("Pipeline finished in " + fmt(data.result.elapsed_s, 2) + " s.");
  } catch (err) {
    banner("Pipeline failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run pipeline";
  }
}

function metricRow(label, value, cls) {
  const tr = document.createElement("tr");
  const td1 = document.createElement("td");
  td1.textContent = label;
  const td2 = document.createElement("td");
  td2.className = "num " + (cls || "");
  td2.textContent = value;
  tr.appendChild(td1);
  tr.appendChild(td2);
  return tr;
}

function retentionClass(v) {
  if (v === null || v === undefined) return "";
  if (v >= 0.9) return "metric-good";
  if (v >= 0.75) return "metric-warn";
  return "metric-bad";
}

function renderResult() {
  const r = state.result;
  const m = r.metrics;
  const ps = m.peak_suppression;
  const cp = m.content_preservation;
  const accepted = r.accepted_peaks;
  $("result-summary").textContent =
    `${r.method} · ${accepted.length} screen(s) · ${r.notch_specs.length} notch(es) · ` +
    `scale ${fmt(r.scale, 2)} · ${r.inputs.input_shape.join("×")} → ` +
    `${r.inputs.output_shape.join("×")}`;

  const tb = $("metrics-table").querySelector("tbody");
  tb.innerHTML = "";

  if (accepted.length) {
    tb.appendChild(metricRow(
      "Screen peak suppression (mean)",
      fmt(ps.mean_reduction_db, 1) + " dB",
      ps.all_reduced ? "metric-good" : "metric-warn"));
    tb.appendChild(metricRow(
      "All accepted peaks reduced > 3 dB",
      ps.all_reduced ? "yes" : "no",
      ps.all_reduced ? "metric-good" : "metric-bad"));
  } else {
    tb.appendChild(metricRow("Screen peak suppression", "no screen applied"));
  }

  for (const [key, label] of [
    ["edge_sharpness", "Edge sharpness retained"],
    ["edge_contrast", "Edge contrast retained"],
    ["hairline_contrast", "Hairline contrast retained"],
    ["detail_energy", "Detail band energy retained"],
  ]) {
    const v = cp.metrics[key].retained_fraction;
    tb.appendChild(metricRow(label + " (vs input)", pct(v), retentionClass(v)));
  }

  const moire = m.moire_vs_area_reference;
  if (moire && moire.ratio !== undefined) {
    tb.appendChild(metricRow(
      "Moiré risk vs area reference",
      fmt(moire.ratio, 2) + "×" + (moire.added_moire_risk ? " (raised)" : " (ok)"),
      moire.added_moire_risk ? "metric-bad" : "metric-good"));
  }

  const seam = m.seam;
  if (seam && seam.border_step_change !== undefined) {
    tb.appendChild(metricRow(
      "Wrap-seam border step change",
      fmt(seam.border_step_change, 3) + "×",
      seam.wrap_seam_suspected ? "metric-bad" : "metric-good"));
  }

  const reg = m.color_registration;
  if (reg && reg.output && reg.output.r) {
    const moved = (reg.output.r.dx || reg.output.r.dy || reg.output.b.dx || reg.output.b.dy);
    tb.appendChild(metricRow(
      "Color registration after processing (R/B vs G)",
      moved ? "moved" : "0,0 preserved",
      moved ? "metric-bad" : "metric-good"));
  }

  const warn = r.warnings;
  if (warn && warn.length) {
    for (const w of warn) tb.appendChild(metricRow("Note", w, "metric-warn"));
  }

  $("audit").textContent = JSON.stringify({    method: r.method,
    peaks_accepted: accepted.map((p) => ({
      period_px: p.period_px, angle_deg: p.angle_deg,
      classification: p.classification, confidence: p.confidence })),
    harmonics: r.harmonics,
    include_lattice_lines: r.include_lattice_lines,
    notch_specs: r.notch_specs.map((s) => ({
      label: s.label, period_px: s.period_px, angle_deg: s.angle_deg,
      harmonic_order: s.harmonic_order,
      center_fx: s.center_fx, center_fy: s.center_fy,
      radius: s.resolved_radius_cyc_per_px,
      apodization: s.apodization })),
    processing_audit: r.processing_audit,
    resample_audit: r.resample_audit,
    resample_plan: r.resample_plan,
  }, null, 2);
}

// ---------- zones: API ----------

async function suggestZones() {
  clearBanner();
  const btn = $("suggest-regions");
  btn.disabled = true;
  btn.textContent = "Measuring…";
  try {
    const data = await api("/api/regions/suggest", { session: state.session });
    state.suggestion = data.suggestion;
    const cands = data.suggestion.candidates || [];
    // Candidates are proposals only. Replace the current list so the operator
    // starts from them, then confirms or edits each one.
    state.regions = cands.map((c) => {
      state.regionCounter += 1;
      return {
        id: c.id || `zone-${state.regionCounter}`,
        name: c.id || `Zone ${state.regionCounter}`,
        rect: c.rect.map((v) => Math.round(v)),
        feather_px: c.feather_px || 28,
        opacity: c.opacity != null ? c.opacity : 1,
        enabled: true,
        status: "candidate",
        requires_confirmation: true,
        auto_peaks: true,
        peaks: (c.peaks || []).map((p) => ({ ...p })),
        seed_peaks: (c.seed_peaks || []).map((p) => ({ ...p })),
        depends_on: [],
        params: { ...(c.params || {}) },
        notes: c.notes || [],
      };
    });
    state.dirtyRegions = new Set(state.regions.map((r) => r.id));
    renderRegions();
    $("recompute-regions").disabled = state.regions.length === 0;

    const host = $("suggest-notes");
    host.innerHTML = "";
    for (const note of [
      data.suggestion.note,
      `coverage estimate ${pct(data.suggestion.coverage_estimate)} · ` +
      `${data.suggestion.tiles_without_screen} tile(s) without periodic evidence`,
    ]) {
      const d = document.createElement("div");
      d.className = "warning";
      d.textContent = note;
      host.appendChild(d);
    }
    if (!cands.length) {
      banner("No zone candidates were found. Add zones manually.", "error");
    } else {
      banner(`Suggested ${cands.length} zone candidate(s). Review, confirm, then ` +
             "recompute before running.");
    }
  } catch (err) {
    banner("Zone suggestion failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Suggest zones from screens";
  }
}

async function recomputeZones() {
  clearBanner();
  if (!state.regions.length) return;
  const btn = $("recompute-regions");
  btn.disabled = true;
  btn.textContent = "Recomputing…";
  try {
    const dirty = [...state.dirtyRegions];
    const data = await api("/api/regions/recompute", {
      session: state.session,
      regions: state.regions,
      dirty_ids: dirty.length ? dirty : null,
      deep_channels: $("deep-channels-regions").checked,
      flat_tile_assist: $("flat-assist").checked,
    });
    state.regions = data.regions;
    state.dirtyRegions.clear();
    renderRegions();
    const host = $("region-warnings");
    host.innerHTML = "";
    for (const w of data.warnings || []) {
      const d = document.createElement("div");
      d.className = "warning";
      d.textContent = w;
      host.appendChild(d);
    }
    for (const e of data.errors || []) {
      const d = document.createElement("div");
      d.className = "warning";
      d.textContent = "error: " + e;
      host.appendChild(d);
    }
    banner(`Re-analyzed ${data.recomputed.length} zone(s)` +
      (dirty.length ? `: ${dirty.join(", ")} (and dependents)` : "."));
  } catch (err) {
    banner("Zone recompute failed: " + err.message, "error");
  } finally {
    btn.disabled = state.regions.length === 0;
    btn.textContent = "Recompute changed zones";
  }
}

function regionPayload() {
  return state.regions.map((r) => ({
    id: r.id,
    name: r.name,
    rect: r.rect,
    feather_px: r.feather_px,
    opacity: r.opacity,
    enabled: r.enabled,
    status: r.status,
    auto_peaks: r.auto_peaks,
    peaks: r.peaks,
    seed_peaks: r.seed_peaks,
    depends_on: r.depends_on || [],
    params: r.params,
  }));
}

async function runZones() {
  clearBanner();
  if (!state.regions.length) {
    banner("Add or suggest zones first.", "error");
    return;
  }
  const base = Number($("base-strength").value);
  if (base > 0) {
    if (!state.analysis) {
      banner("Base strength > 0 needs a whole-image analysis first so the base " +
             "layer has an explicit screen. Analyze, or set base strength to 0.", "error");
      return;
    }
  }
  const btn = $("run-regions");
  btn.disabled = true;
  btn.textContent = "Running zones…";
  try {
    const globalPeaks = (state.analysis ? state.analysis.peaks : [])
      .filter((p) => p.selected)
      .map((p) => ({ period_px: p.period_px, angle_deg: p.angle_deg }));
    const data = await api("/api/regions/run", {
      session: state.session,
      roi: state.roi,
      regions: regionPayload(),
      source_shape: [state.full.h, state.full.w],
      params: { ...readParams(), base_strength: base },
      global_peaks: base > 0 ? globalPeaks : null,
    });
    state.regionResult = data.result;
    state.regionImages = data.images;
    state.lastParamsJson = data.params_json;
    for (const k of ["input", "suppressed", "output", "difference"]) {
      $(`result-${k}`).src = "data:image/png;base64," + data.images[k + ".png"];
    }
    const wfig = $("weights-figure");
    if (data.images["weights.png"]) {
      $("result-weights").src = "data:image/png;base64," + data.images["weights.png"];
      wfig.classList.remove("hidden");
      $("download-weights").classList.remove("hidden");
    } else {
      wfig.classList.add("hidden");
      $("download-weights").classList.add("hidden");
    }
    drawRegionWeights();
    renderRegionResult();
    banner("Zone pipeline finished in " + fmt(data.result.elapsed_s, 2) + " s · " +
      data.result.regions.length + " zone(s), " + data.result.groups.length +
      " parameter group(s).");
  } catch (err) {
    banner("Zone pipeline failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run zone pipeline";
  }
}

function renderRegionResult() {
  const r = state.regionResult;
  if (!r) return;
  $("result-summary").textContent =
    `zone-coordinated · ${r.regions.length} zone(s) · ${r.groups.length} group(s) · ` +
    `scale ${fmt(r.scale, 2)} · strength ${fmt(r.weights.effective_strength_range[0], 2)}–` +
    `${fmt(r.weights.effective_strength_range[1], 2)} · ` +
    `${r.inputs.input_shape.join("×")} → ${r.inputs.output_shape.join("×")}`;

  const tb = $("metrics-table").querySelector("tbody");
  tb.innerHTML = "";

  const cont = r.boundary_continuity;
  const nb = (cont.boundaries || []).length;
  tb.appendChild(metricRow(
    "Zone boundaries checked",
    String(nb),
    ""));
  tb.appendChild(metricRow(
    "Hard boundaries (seam suspected)",
    `${cont.hard_boundaries} of ${nb}` + (cont.seam_suspected ? " · SEAM" : " · none"),
    cont.seam_suspected ? "metric-bad" : "metric-good"));
  if (nb) {
    tb.appendChild(metricRow(
      "Min transition width / feather",
      fmt(cont.min_transition_ratio, 2),
      cont.min_transition_ratio >= 0.5 ? "metric-good" : "metric-warn"));
  }

  tb.appendChild(metricRow(
    "Weight sum range (must be ~1)",
    `${fmt(r.weights.sum_range[0], 3)}–${fmt(r.weights.sum_range[1], 3)}`,
    r.weights.sum_range[0] >= 0.999 && r.weights.sum_range[1] <= 1.001
      ? "metric-good" : "metric-bad"));
  tb.appendChild(metricRow(
    "Pixels covered only by the base layer",
    pct(r.weights.base_fraction),
    r.weights.base_fraction > 0.25 ? "metric-warn" : ""));
  tb.appendChild(metricRow(
    "Groups actually filtered",
    `${r.processing_audit.candidate_passes} pass(es)` +
    (r.processing_audit.mode ? ` · ${r.processing_audit.mode}` : ""),
    "metric-good"));

  for (const rm of r.region_metrics || []) {
    if (rm.status !== "measured") {
      tb.appendChild(metricRow(`Zone ${rm.id}`, rm.status, "metric-warn"));
      continue;
    }
    const cp = rm.content_preservation.worst_retained_fraction;
    const ps = rm.peak_suppression || {};
    tb.appendChild(metricRow(
      `Zone ${rm.id} · weight ${fmt(rm.mean_weight, 2)} · strength ${fmt(rm.strength, 2)}`,
      `detail ${pct(cp)}` +
      (ps.mean_reduction_db !== undefined
        ? ` · suppression ${fmt(ps.mean_reduction_db, 1)} dB` : " · no screen"),
      retentionClass(cp)));
  }

  for (const w of r.warnings || []) {
    tb.appendChild(metricRow("Note", w, "metric-warn"));
  }

  $("audit").textContent = JSON.stringify({
    mode: r.mode,
    weight_rule: r.weights.rule,
    weight_sum_range: r.weights.sum_range,
    base_fraction: r.weights.base_fraction,
    effective_strength_range: r.weights.effective_strength_range,
    weight_legend: r.weights.legend,
    region_mapping: r.region_mapping,
    regions: r.regions.map((q) => ({
      id: q.id, rect_page: q.rect_page, rect_in_crop: q.rect,
      feather_px: q.feather_px, opacity: q.opacity, enabled: q.enabled,
      clipped_to_roi: q.clipped_to_roi,
      peaks: q.peaks, params: q.params,
    })),
    groups: r.groups,
    boundary_continuity: r.boundary_continuity,
    processing_audit: r.processing_audit,
    resample_plan: r.resample_plan,
  }, null, 2);
}

// ---------- downloads ----------

function downloadBlob(data, filename, mime) {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function b64ToBlob(b64, mime) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function downloadJson() {
  if (!state.session) return;
  if (!state.lastParamsJson) {
    banner("Run the pipeline first; the parameters come from the actual run.", "error");
    return;
  }
  downloadBlob(new Blob([state.lastParamsJson], { type: "application/json" }),
    "descreen-parameters.json", "application/json");
}

function downloadPng(which) {
  const src = state.resultImages || state.regionImages;
  if (!src) {
    banner("Run a pipeline first.", "error");
    return;
  }
  const key = which + ".png";
  if (!src[key]) {
    banner(`No ${which} image is available from the last run.`, "error");
    return;
  }
  downloadBlob(b64ToBlob(src[key], "image/png"),
    "descreen-" + which + ".png", "image/png");
}

// ---------- wiring ----------

$("load-fixture").addEventListener("click", () => loadFixture().catch(
  (e) => banner(e.message, "error")));
$("upload").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) uploadFile(f).catch((err) => banner(err.message, "error"));
});
$("analyze").addEventListener("click", analyze);
$("run").addEventListener("click", runPipeline);
$("run-regions").addEventListener("click", runZones);
$("suggest-regions").addEventListener("click", suggestZones);
$("recompute-regions").addEventListener("click", recomputeZones);
$("add-region").addEventListener("click", addRegion);
$("clear-regions").addEventListener("click", clearRegions);
$("show-weights").addEventListener("change", () => {
  if ($("show-weights").checked && !state.regionImages) {
    banner("Run the zone pipeline first to see the weight overlay.", "error");
    $("show-weights").checked = false;
    return;
  }
  drawRegionWeights();
});
$("add-manual").addEventListener("click", addManual);
$("download-json").addEventListener("click", downloadJson);
$("download-output").addEventListener("click", () => downloadPng("output"));
$("download-suppressed").addEventListener("click", () => downloadPng("suppressed"));
$("download-weights").addEventListener("click", () => downloadPng("weights"));

loadFixtures().then(() => loadFixture().catch(
  (e) => banner(e.message, "error")));
