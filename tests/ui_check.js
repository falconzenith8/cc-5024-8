// Browser-level verification of the descreen UI.
// Drives the real page in Chromium: loads a fixture, analyzes, selects an ROI,
// adds a manual screen, runs the pipeline, downloads parameters, and asserts
// the rendered numbers come from the live API response.
"use strict";

const { chromium } = require("playwright-core");
const fs = require("fs");
const path = require("path");
const os = require("os");

const BASE = process.env.DESCREEN_URL || "http://127.0.0.1:8765";
const OUT = process.env.DESCREEN_SHOTS || "reports/ui";
const CHROME = process.env.CHROME_PATH ||
  path.join(os.homedir(),
    ".cache/ms-playwright/chromium-1255/chrome-linux-arm64/chrome");

function assert(cond, msg) {
  if (!cond) throw new Error("ASSERTION FAILED: " + msg);
  console.log("  ok  " + msg);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    executablePath: CHROME,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
    env: {
      ...process.env,
      LD_LIBRARY_PATH: [
        process.env.DESCREEN_CHROME_LIBS || "",
        "/tmp/chromedeps/root/usr/lib/aarch64-linux-gnu",
        "/tmp/chromedeps/root/lib/aarch64-linux-gnu",
      ].filter(Boolean).join(":"),
    },
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1600 },
    acceptDownloads: true,
  });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push("pageerror: " + e.message));

  console.log("1. load page");
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.waitForFunction(() =>
    document.querySelector("#source-meta").textContent.includes("×"));
  const meta = await page.textContent("#source-meta");
  console.log("   source-meta:", meta);
  assert(/\d+×\d+/.test(meta), "a fixture loaded automatically");
  await page.screenshot({ path: path.join(OUT, "01-loaded.png"), fullPage: true });

  console.log("2. choose the color palette fixture");
  await page.selectOption("#fixture-select", "palette");
  await page.click("#load-fixture");
  await page.waitForFunction(() =>
    document.querySelector("#source-meta").textContent.includes("palette"));
  await page.waitForTimeout(400);
  const srcImgOk = await page.evaluate(async () => {
    const c = document.querySelector("#source-canvas");
    return c.width > 100 && c.height > 100;
  });
  assert(srcImgOk, "source canvas has the image drawn");

  console.log("3. analyze whole image with per-channel analysis");
  await page.check("#deep-channels");
  await page.click("#analyze");
  await page.waitForFunction(() =>
    !document.querySelector("#analyze").disabled &&
    document.querySelectorAll("#peaks-table tbody tr").length > 0,
    { timeout: 120000 });
  const rows = await page.$$eval("#peaks-table tbody tr", (trs) =>
    trs.map((tr) => ({
      checked: tr.querySelector("input[type=checkbox]").checked,
      period: parseFloat(tr.children[1].textContent),
      angleText: tr.children[2].textContent,
      cls: tr.children[3].textContent.trim(),
      conf: parseFloat(tr.children[4].textContent),
      why: tr.children[7].textContent,
    })));
  console.log("   rows:", JSON.stringify(rows, null, 1));
  const selected = rows.filter((r) => r.checked);
  assert(selected.length >= 2, "at least two screens auto-selected on the color page");
  const specOk = await page.evaluate(() => {
    const s = document.querySelector("#spectrum-img");
    return s && s.src.startsWith("data:image/png;base64,") && s.src.length > 5000;
  });
  assert(specOk, "spectrum image is present");
  const ovOk = await page.evaluate(() => {
    const s = document.querySelector("#overview-img");
    return s && s.src.startsWith("data:image/png;base64,");
  });
  assert(ovOk, "overview image is present");
  await page.screenshot({ path: path.join(OUT, "02-analyzed.png"), fullPage: true });

  console.log("4. drag an ROI on the source canvas");
  const box = await page.locator("#source-canvas").boundingBox();
  const x0 = box.x + box.width * 0.15;
  const y0 = box.y + box.height * 0.45;
  const x1 = box.x + box.width * 0.75;
  const y1 = box.y + box.height * 0.80;
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 6 });
  await page.mouse.move(x1, y1, { steps: 6 });
  await page.mouse.up();
  const roiVisible = await page.isVisible("#roi-box");
  assert(roiVisible, "ROI box is drawn after dragging");

  console.log("5. add a manual screen");
  await page.click("summary");
  await page.fill("#manual-period", "7.95");
  await page.fill("#manual-angle", "0");
  await page.click("#add-manual");
  const manualCount = await page.$$eval("#manual-list li", (l) => l.length);
  assert(manualCount === 1, "manual peak added to the list");
  await page.screenshot({ path: path.join(OUT, "03-roi-manual.png"), fullPage: true });

  console.log("6. run the pipeline");
  await page.fill("#scale", "0.5");
  await page.click("#run");
  await page.waitForFunction(() =>
    !document.querySelector("#run").disabled &&
    document.querySelector("#result-output").src.startsWith("data:image/png"),
    { timeout: 180000 });
  const resultSummary = await page.textContent("#result-summary");
  console.log("   result-summary:", resultSummary);
  assert(/frequency_notch/.test(resultSummary), "result names the method");
  const metricRows = await page.$$eval("#metrics-table tbody tr", (trs) =>
    trs.map((tr) => [tr.children[0].textContent, tr.children[1].textContent]));
  console.log("   metrics:", JSON.stringify(metricRows, null, 1));
  const suppressRow = metricRows.find(([k]) => /suppression/i.test(k));
  assert(suppressRow, "suppression metric is shown");
  const dB = parseFloat(suppressRow[1]);
  assert(dB > 6, "screen suppression is at least 6 dB (measured, not assumed)");
  const seamRow = metricRows.find(([k]) => /seam/i.test(k));
  if (seamRow) {
    assert(parseFloat(seamRow[1]) < 1.25, "no wrap seam introduced");
  }
  const regRow = metricRows.find(([k]) => /registration/i.test(k));
  assert(regRow && /preserved/.test(regRow[1]), "color registration preserved");

  const audit = await page.textContent("#audit");
  assert(audit.includes("center_fx"), "audit trail lists notch centers");
  await page.screenshot({ path: path.join(OUT, "04-result.png"), fullPage: true });

  console.log("7. download parameters and output");
  const dlJson = page.waitForEvent("download");
  await page.click("#download-json");
  const jsonDl = await dlJson;
  const jsonPath = path.join(OUT, "descreen-parameters.json");
  await jsonDl.saveAs(jsonPath);
  const parsed = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  assert(parsed.method === "frequency_notch", "downloaded JSON records the method");
  assert(Array.isArray(parsed.notch_specs) && parsed.notch_specs.length > 0,
    "downloaded JSON records the notch specs");
  assert(parsed.metrics && parsed.metrics.peak_suppression,
    "downloaded JSON includes measured metrics");
  assert(Array.isArray(parsed.manual_peaks) && parsed.manual_peaks.length === 1,
    "downloaded JSON records the manual peak");

  const dlPng = page.waitForEvent("download");
  await page.click("#download-output");
  const pngDl = await dlPng;
  const pngPath = path.join(OUT, "descreen-output.png");
  await pngDl.saveAs(pngPath);
  const sz = fs.statSync(pngPath).size;
  assert(sz > 1000, "downloaded output PNG is a real image (" + sz + " bytes)");

  console.log("7b. the spatial-FIR method also suppresses on the same input");
  await page.selectOption("#method", "spatial_fir");
  await page.click("#run");
  await page.waitForFunction(() =>
    !document.querySelector("#run").disabled &&
    document.querySelector("#result-summary").textContent.includes("spatial_fir"),
    { timeout: 180000 });
  const firSummary = await page.textContent("#result-summary");
  console.log("   result-summary:", firSummary);
  const firRows = await page.$$eval("#metrics-table tbody tr", (trs) =>
    trs.map((tr) => [tr.children[0].textContent, tr.children[1].textContent]));
  const firSuppress = firRows.find(([k]) => /suppression/i.test(k));
  assert(firSuppress && parseFloat(firSuppress[1]) > 6,
    "spatial FIR suppression is at least 6 dB");
  const firSeam = firRows.find(([k]) => /seam/i.test(k));
  if (firSeam) assert(parseFloat(firSeam[1]) < 1.25, "FIR leaves no wrap seam");
  await page.selectOption("#method", "frequency_notch");

  console.log("8. a clean fixture must not auto-select anything");
  await page.selectOption("#fixture-select", "photo_clean");
  await page.click("#load-fixture");
  await page.waitForFunction(() =>
    document.querySelector("#source-meta").textContent.includes("photo"));
  await page.click("#analyze");
  await page.waitForFunction(() =>
    !document.querySelector("#analyze").disabled, { timeout: 120000 });
  const cleanRows = await page.$$eval("#peaks-table tbody tr", (trs) =>
    trs.map((tr) => ({
      checked: tr.querySelector("input[type=checkbox]").checked,
      cls: tr.children[3].textContent.trim(),
    })));
  const selectedOnClean = cleanRows.filter((r) => r.checked);
  console.log("   clean rows:", JSON.stringify(cleanRows));
  assert(selectedOnClean.length === 0,
    "no screen auto-selected on the screen-free photo");
  const warnText = await page.textContent("#warnings");
  assert(/real content|No narrowband screen|not selected/i.test(warnText),
    "warning explains why nothing was selected");
  await page.screenshot({ path: path.join(OUT, "05-clean-no-false-positive.png"),
    fullPage: true });

  console.log("9. console clean");
  assert(consoleErrors.length === 0,
    "no console errors (" + JSON.stringify(consoleErrors) + ")");

  await browser.close();
  console.log("\nUI verification passed. Screenshots in " + OUT);
})().catch(async (err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
