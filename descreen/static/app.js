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
  state.manual = [];
  state.rejected = [];
  drawSource();
  const truth = info.truth && Object.keys(info.truth).length
    ? " · " + JSON.stringify(info.truth) : "";
  $("source-meta").textContent =
    `${info.kind} · ${info.shape[1]}×${info.shape[0]} · ${info.channels} channel(s)${truth}`;
  $("analyze").disabled = false;
  $("run").disabled = true;
  $("analysis-summary").textContent = "Not analyzed yet.";
  $("warnings").innerHTML = "";
  $("peaks-table").querySelector("tbody").innerHTML = "";
  $("lattice-explained").textContent = "";
  $("result-summary").textContent = "Not run yet.";
  $("metrics-table").querySelector("tbody").innerHTML = "";
  $("audit").textContent = "";
  for (const id of ["overview-img", "spectrum-img", "result-input",
                    "result-suppressed", "result-output", "result-difference"]) {
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
  if (!drag) return;
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

window.addEventListener("resize", drawRoiBox);

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
  if (!state.resultImages) {
    banner("Run the pipeline first.", "error");
    return;
  }
  const key = which + ".png";
  downloadBlob(b64ToBlob(state.resultImages[key], "image/png"),
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
$("add-manual").addEventListener("click", addManual);
$("download-json").addEventListener("click", downloadJson);
$("download-output").addEventListener("click", () => downloadPng("output"));
$("download-suppressed").addEventListener("click", () => downloadPng("suppressed"));

loadFixtures().then(() => loadFixture().catch(
  (e) => banner(e.message, "error")));
