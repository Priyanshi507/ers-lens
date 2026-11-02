const SCHEMA = 1;

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escapeHTML = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const CAP = [["250", "250 kW (before Miami)", "--ink-soft"], ["350", "350 kW (from Miami)", "--battery"]];

const state = { data: null, circuit: null, charts: {} };

async function load() {
  const res = await fetch("data/fia.json");
  if (!res.ok) throw new Error(`data/fia.json returned HTTP ${res.status}`);
  const data = await res.json();
  if (data.schema !== SCHEMA) throw new Error(`data/fia.json is version ${data.schema}, this page reads ${SCHEMA}`);
  return data;
}

function baseOptions(extra = {}) {
  const grid = css("--grid"), ink = css("--ink-soft");
  return {
    animation: false,
    maintainAspectRatio: false,
    indexAxis: extra.indexAxis,
    interaction: { mode: "nearest", axis: extra.indexAxis === "y" ? "y" : "x", intersect: false },
    plugins: { legend: { labels: { color: ink, boxWidth: 14 } }, ...extra.plugins },
    scales: Object.fromEntries(Object.entries(extra.scales || {}).map(([k, s]) => [k, {
      grid: { color: grid }, ticks: { color: ink, ...(s.ticks || {}) }, border: { color: grid },
      ...s, title: s.title ? { display: true, color: ink, text: s.title } : undefined,
    }])),
  };
}

const straightShade = {
  id: "straightShade",
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || opts.from === undefined) return;
    const { ctx, chartArea: area, scales: { x } } = chart;
    ctx.save();
    ctx.fillStyle = css("--battery-wash");
    const left = x.getPixelForValue(opts.from);
    ctx.fillRect(left, area.top, x.getPixelForValue(opts.to) - left, area.bottom - area.top);
    ctx.restore();
  },
};
Chart.register(straightShade);

const name = (c) => c.label + (c.assumed_recharge_mj ? " *" : "");
const draw = (key, canvas, config) => {
  state.charts[key]?.destroy();
  state.charts[key] = new Chart(canvas, config);
};

function renderAnswer() {
  const d = state.data, c1 = d.predictions.find((p) => p.id === "C1");
  $("answer-main").innerHTML = `In the model, the 350 kW cap <strong>takes more energy by super-clipping</strong> at ` +
    `${c1.count} of ${c1.total} circuits, and less at the end of the longest straight at ${d.straight_down.count} of ${d.straight_down.total}.`;
  $("answer-detail").textContent = "The part of super-clipping that spectators see on the long straights shrinks, while super-clipping " +
    "as a whole grows. The extra energy is taken at lower speeds, where the MGU-K can use more than 250 kW: its limit tapers " +
    "from 350 kW at 290 km/h to 250 kW at 310 km/h.";
}

function renderCalendar() {
  const cs = [...state.data.circuits].sort((a, b) => b.caps["350"].superclip_mj - a.caps["350"].superclip_mj);
  draw("energy", $("energy-chart"), {
    type: "bar",
    data: {
      labels: cs.map(name),
      datasets: CAP.map(([cap, label, color]) => ({
        label, data: cs.map((c) => c.caps[cap].superclip_mj), backgroundColor: css(color), borderRadius: 3, barPercentage: 0.9,
      })),
    },
    options: baseOptions({ indexAxis: "y", scales: { x: { title: "Super-clipping energy per lap (MJ)", min: 0 }, y: {} },
      plugins: { tooltip: { callbacks: { label: (t) => `${t.dataset.label}: ${t.parsed.x.toFixed(2)} MJ` } } } }),
  });

  const delta = (c, k) => c.caps["350"][k] - c.caps["250"][k];
  const ws = [...state.data.circuits].sort((a, b) => delta(b, "below_250_mj") - delta(a, "below_250_mj"));
  const up = ws.filter((c) => delta(c, "below_250_mj") > 0 && c.caps["350"].converged).length;
  $("where-note").textContent = `Change from 250 to 350 kW in super-clipping energy taken below and above 250 km/h. ` +
    `Below 250 km/h it rises at ${up} of ${ws.length} circuits (converged solves only). This split was examined after the ` +
    `predictions were scored, so it is exploratory.`;
  draw("where", $("where-chart"), {
    type: "bar",
    data: {
      labels: ws.map(name),
      datasets: [
        { label: "Below 250 km/h", data: ws.map((c) => delta(c, "below_250_mj")), backgroundColor: css("--battery"), borderRadius: 3 },
        { label: "Above 250 km/h", data: ws.map((c) => delta(c, "above_250_mj")), backgroundColor: css("--clip"), borderRadius: 3 },
      ],
    },
    options: baseOptions({ indexAxis: "y", scales: { x: { title: "Change in super-clipping energy (MJ per lap)" }, y: {} },
      plugins: { tooltip: { callbacks: { label: (t) => `${t.dataset.label}: ${t.parsed.x >= 0 ? "+" : ""}${t.parsed.x.toFixed(2)} MJ` } } } }),
  });
}

function renderControls() {
  const el = $("pick-circuit");
  el.replaceChildren(...state.data.circuits.map((c) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = c.label;
    b.setAttribute("aria-pressed", String(c.event === state.circuit));
    b.addEventListener("click", () => { state.circuit = c.event; history.replaceState(null, "", `#${encodeURIComponent(c.event)}`); update(); });
    return b;
  }));
}

function renderLap() {
  const c = state.data.circuits.find((x) => x.event === state.circuit);
  const x = c.caps["250"].v_kmh.map((_, i) => i * c.ds_m);
  const pts = (ys) => ys.map((y, i) => ({ x: x[i], y }));
  const marks = { from: c.straight[0] * c.ds_m, to: c.straight[1] * c.ds_m };
  const xScale = { type: "linear", min: 0, max: x[x.length - 1], title: "Distance around the lap from the slowest point (m)" };
  const lines = (key) => CAP.map(([cap, label, color]) => ({
    label, data: pts(c.caps[cap][key]), borderColor: css(color), borderWidth: cap === "350" ? 1.8 : 1.4, pointRadius: 0,
    borderDash: cap === "250" ? [5, 3] : [],
  }));
  const tip = { callbacks: { title: (t) => `${t[0].parsed.x} m` } };
  draw("speed", $("speed-chart"), { type: "line", data: { datasets: lines("v_kmh") },
    options: baseOptions({ plugins: { straightShade: marks, tooltip: tip }, scales: { x: xScale, y: { title: "km/h" } } }) });
  draw("sc", $("sc-chart"), { type: "line", data: { datasets: lines("superclip_kw").map((d) => ({ ...d, stepped: true })) },
    options: baseOptions({ plugins: { straightShade: marks, tooltip: tip }, scales: { x: xScale, y: { title: "Super-clipping kW", min: 0, suggestedMax: 360 } } }) });

  const a = c.caps["250"], b = c.caps["350"];
  const ok = a.converged && b.converged;
  const change = (v1, v2) => `${v1.toFixed(2)} → ${v2.toFixed(2)}`;
  $("lap-summary").innerHTML = `${escapeHTML(c.label)}: ` + (a.superclip_mj === 0 && b.superclip_mj === 0
    ? "the optimum never super-clips under either cap, so the rule changes nothing here."
    : `under 350 kW the optimum takes <strong>${b.superclip_mj > a.superclip_mj ? "more" : "less"}</strong> energy by super-clipping ` +
      `(${change(a.superclip_mj, b.superclip_mj)} MJ) and ${b.straight_mj < a.straight_mj ? "less" : "more"} on the longest straight ` +
      `(${change(a.straight_mj, b.straight_mj)} MJ).`) +
    (ok ? "" : " One of these solves failed the 0.05 s accuracy rule, so it does not count in any verdict.") +
    (c.assumed_recharge_mj ? ` Recharge limit assumed: ${c.assumed_recharge_mj} MJ.` : "");
  const stat = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
  $("lap-stats").innerHTML = [
    stat("Optimal lap, 250 → 350 kW", `${a.lap_s.toFixed(3)} → ${b.lap_s.toFixed(3)} s`),
    stat("Super-clipping energy", `${change(a.superclip_mj, b.superclip_mj)} MJ`),
    stat("Of it below 250 km/h", `${change(a.below_250_mj, b.below_250_mj)} MJ`),
    stat("Time super-clipping", `${a.superclip_s.toFixed(1)} → ${b.superclip_s.toFixed(1)} s`),
    stat("Longest stretch", `${a.longest_superclip_s.toFixed(2)} → ${b.longest_superclip_s.toFixed(2)} s`),
  ].join("");
}

function renderVerdicts() {
  const head = `<thead><tr><th scope="col">Prediction</th><th scope="col">Result</th><th scope="col">Verdict</th></tr></thead>`;
  const body = state.data.predictions.map((p) =>
    `<tr><th scope="row"><span class="code">${p.id}</span>${escapeHTML(p.text)}</th>` +
    `<td>${p.count} of ${p.total}</td><td class="verdict ${p.held ? "is-held" : "is-not"}">${p.held ? "Held" : "Not held"}</td></tr>`).join("");
  $("verdict-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

function update() {
  renderControls();
  renderLap();
}

async function main() {
  try {
    state.data = await load();
  } catch (err) {
    $("answer-main").textContent = `The FIA-fix data could not be loaded: ${err.message}`;
    return;
  }
  const wanted = decodeURIComponent(location.hash.slice(1));
  state.circuit = state.data.circuits.some((c) => c.event === wanted) ? wanted : "Canada";
  $("generated").textContent = state.data.generated;
  renderAnswer();
  renderCalendar();
  renderVerdicts();
  update();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderCalendar(); renderLap(); });
}

main();
