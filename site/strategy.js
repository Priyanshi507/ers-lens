const SCHEMA = 1;

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escapeHTML = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const CAR_COLOR = { original: "--ink-soft", low: "--battery", high: "--caution" };
const LIMIT_LABEL = { race: "2026 per-race limits", car: "Car-file limit (8.5 MJ)" };
const LIMIT_TEXT = { race: "2026 per-race limits", car: "the car-file limit (8.5 MJ)" };

const ROWS = [
  ["S1", "The optimum clips on the longest straight"],
  ["S2", "It clips for at least 80% of 27 legal cars, at every circuit"],
  ["S3", "It still clips with the MGU-K speed taper removed"],
  ["S4", "Energy is spent at least 20 km/h below full-throttle speed"],
  ["S5", "A tuned simple rule gets within 0.3 s of the optimum"],
  ["S6", "Gradient descent finds the same optimum"],
  ["S7 (exploratory)", "The optimal speed peak sits where real cars' peak does"],
  ["S1 (corrected rules)", "S1 with the 2026 per-race recharge limits"],
  ["S3 (corrected rules)", "S3 with the 2026 per-race recharge limits"],
  ["S4 (corrected rules)", "S4 with the 2026 per-race recharge limits"],
];

const state = { data: null, circuit: "Australia", car: "low", limits: "race", charts: {} };

async function load() {
  const res = await fetch("data/strategy.json");
  if (!res.ok) throw new Error(`data/strategy.json returned HTTP ${res.status}`);
  const data = await res.json();
  if (data.schema !== SCHEMA) throw new Error(`data/strategy.json is version ${data.schema}, this page reads ${SCHEMA}`);
  return data;
}

function readHash() {
  const [circuit, car, limits] = location.hash.slice(1).split("/").map(decodeURIComponent);
  if (state.data.events.includes(circuit)) state.circuit = circuit;
  if (state.data.cars.some((c) => c.id === car)) state.car = car;
  if (limits in LIMIT_LABEL) state.limits = limits;
}

const writeHash = () => history.replaceState(null, "", `#${[state.circuit, state.car, state.limits].map(encodeURIComponent).join("/")}`);
const carSpec = (id) => state.data.cars.find((c) => c.id === id);
const profile = (car = state.car, limits = state.limits, circuit = state.circuit) =>
  state.data.circuits[circuit].cars[car].profiles[limits];

function baseOptions(extra = {}) {
  const grid = css("--grid"), ink = css("--ink-soft");
  return {
    animation: false,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", axis: "x", intersect: false },
    plugins: { legend: { labels: { color: ink, boxWidth: 14 } }, ...extra.plugins },
    scales: Object.fromEntries(Object.entries(extra.scales || {}).map(([k, s]) => [k, {
      grid: { color: grid }, ticks: { color: ink, ...(s.ticks || {}) }, border: { color: grid },
      ...s, title: s.title ? { display: true, color: ink, text: s.title } : undefined,
    }])),
  };
}

// Shades the longest straight and marks the speed peak on every lap chart.
const straightMarks = {
  id: "straightMarks",
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || opts.from === undefined) return;
    const { ctx, chartArea: area, scales: { x } } = chart;
    ctx.save();
    ctx.fillStyle = css("--battery-wash");
    const left = x.getPixelForValue(opts.from);
    ctx.fillRect(left, area.top, x.getPixelForValue(opts.to) - left, area.bottom - area.top);
    if (opts.peak !== undefined) {
      ctx.strokeStyle = css("--clip");
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      const px = x.getPixelForValue(opts.peak);
      ctx.beginPath(); ctx.moveTo(px, area.top); ctx.lineTo(px, area.bottom); ctx.stroke();
    }
    ctx.restore();
  },
};

const hLine = {
  id: "hLine",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || opts.y === undefined) return;
    const { ctx, chartArea: area, scales: { y } } = chart;
    ctx.save();
    ctx.strokeStyle = css("--clip");
    ctx.setLineDash([6, 4]);
    const py = y.getPixelForValue(opts.y);
    ctx.beginPath(); ctx.moveTo(area.left, py); ctx.lineTo(area.right, py); ctx.stroke();
    ctx.fillStyle = css("--clip");
    ctx.font = `500 12px ${css("--body")}`;
    ctx.fillText(opts.label, area.left + 6, py - 6);
    ctx.restore();
  },
};

Chart.register(straightMarks, hLine);

function segButtons(el, options, current, onPick) {
  el.replaceChildren(...options.map(([value, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(value === current));
    b.addEventListener("click", () => onPick(value));
    return b;
  }));
}

function renderControls() {
  segButtons($("pick-circuit"), state.data.events.map((e) => [e, e]), state.circuit, (v) => { state.circuit = v; update(); });
  segButtons($("pick-car"), state.data.cars.map((c) => [c.id, c.label]), state.car, (v) => { state.car = v; update(); });
  segButtons($("pick-limits"), Object.entries(LIMIT_LABEL), state.limits, (v) => { state.limits = v; update(); });
}

function renderVerdicts() {
  const cars = state.data.cars;
  const head = `<thead><tr><th scope="col">Prediction</th>${cars.map((c) =>
    `<th scope="col">${escapeHTML(c.label)}<span class="sub">${escapeHTML(c.detail)}</span></th>`).join("")}<th scope="col">Verdict</th></tr></thead>`;
  const body = ROWS.map(([key, text], i) => {
    const held = cars.map((c) => c.verdicts[key].held);
    const verdict = held.every(Boolean) ? ["Robust: held", "is-held"]
      : held.every((h) => !h) ? ["Robust: not held", "is-not"] : ["Depends on the car", "is-mixed"];
    const group = i === 7 ? `<tr class="group"><th colspan="${cars.length + 2}" scope="rowgroup">With the 2026 per-race recharge limits</th></tr>` : "";
    const cells = cars.map((c) => {
      const v = c.verdicts[key];
      return `<td class="${v.held ? "is-held" : "is-not"}" title="${escapeHTML(v.detail)}">${v.held ? "Held" : "Not held"}</td>`;
    }).join("");
    const label = key.split(" ")[0];
    return `${group}<tr><th scope="row"><span class="code">${label}</span>${escapeHTML(text)}</th>${cells}<td class="verdict ${verdict[1]}">${verdict[0]}</td></tr>`;
  }).join("");
  $("verdict-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

function renderAnswer() {
  const clipsEverywhere = state.data.cars.every((c) => c.verdicts.S1.held && c.verdicts["S1 (corrected rules)"].held);
  $("answer-main").innerHTML = clipsEverywhere
    ? "On every car tested, the fastest lap <strong>cuts electric power before the end of the longest straight</strong>."
    : "The optimum does not clip on every car tested.";
  $("answer-detail").textContent = "Clipping is what an optimizer chooses, not only a side effect of the regulation taper: near top " +
    "speed, drag absorbs power roughly as v³, so a joule deployed there buys little time. How widespread clipping is " +
    "across legal cars, and how far below top speed the energy goes, depend on the car's drag and engine power.";
}

function lapCharts() {
  const p = profile();
  const ds = p.ds_m;
  const x = p.v.map((_, i) => i * ds);
  const marks = { from: p.straight[0] * ds, to: p.straight[1] * ds, peak: p.peak * ds };
  const xScale = { type: "linear", min: 0, max: x[x.length - 1], title: "Distance around the lap (m)" };
  const pts = (ys) => ys.map((y, i) => ({ x: x[i], y }));
  const color = css(CAR_COLOR[state.car]);

  const charts = [
    ["speed", $("speed-chart"), [{ label: "Speed (km/h)", data: pts(p.v), borderColor: css("--ink"), borderWidth: 1.6 }],
      { y: { title: "km/h" } }],
    ["power", $("power-chart"), [
      { label: "Electric power deployed (kW)", data: pts(p.deploy), borderColor: color, backgroundColor: color + "33", fill: "origin", borderWidth: 1.4, stepped: true },
      { label: "Harvested (kW)", data: pts(p.harvest.map((h) => -h)), borderColor: css("--clip"), backgroundColor: css("--clip") + "26", fill: "origin", borderWidth: 1, stepped: true },
      { label: "MGU-K limit at this speed (kW)", data: pts(p.cap), borderColor: css("--ink-soft"), borderDash: [5, 4], borderWidth: 1.2 },
    ], { y: { title: "kW", suggestedMin: -360, suggestedMax: 360 } }],
    ["soc", $("soc-chart"), [{ label: "Battery charge (MJ)", data: pts(p.soc), borderColor: color, borderWidth: 1.6 }],
      { y: { title: "MJ", min: 0, max: 4 } }],
  ];
  for (const [key, canvas, datasets, scales] of charts) {
    state.charts[key]?.destroy();
    state.charts[key] = new Chart(canvas, {
      type: "line",
      data: { datasets: datasets.map((d) => ({ pointRadius: 0, ...d })) },
      options: baseOptions({ plugins: { straightMarks: marks, tooltip: { callbacks: { title: (c) => `${c[0].parsed.x} m` } } },
        scales: { x: xScale, ...scales } }),
    });
  }

  const car = carSpec(state.car);
  const clips = p.clip_ratio !== null && p.clip_ratio <= state.data.thresholds.clip_max;
  $("lap-summary").innerHTML = `${escapeHTML(state.circuit)}, ${escapeHTML(car.label.toLowerCase())} (${escapeHTML(car.detail)}), ` +
    `${escapeHTML(LIMIT_TEXT[state.limits])}: the optimum <strong>${clips ? "clips" : "does not clip"}</strong> on the ` +
    `${((p.straight[1] - p.straight[0]) * ds).toLocaleString("en")} m straight, with power at the speed peak ` +
    `${p.clip_ratio === null ? "undefined" : `${Math.round(p.clip_ratio * 100)}%`} of its earlier mean. ` +
    (p.soc[p.peak] < 0.3
      ? `The battery is almost empty there (${p.soc[p.peak].toFixed(2)} MJ), so power runs out where speed would buy least.`
      : `It cuts power with ${p.soc[p.peak].toFixed(2)} MJ still in the battery: a choice about where energy buys most time, not a shortage.`);
  const stat = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
  $("lap-stats").innerHTML = [
    stat("Optimal lap", `${p.lap_s.toFixed(3)} s`),
    stat("Energy from battery", `${p.deployed_mj.toFixed(2)} MJ`),
    stat("Energy recovered", `${p.harvested_mj.toFixed(2)} MJ`),
    stat("Recharge limit", `${p.harvest_limit_mj.toFixed(1)} MJ`),
    stat("Speed peak to braking", `${p.peak_to_brake_m} m`),
    stat("Battery at the speed peak", `${p.soc[p.peak].toFixed(2)} MJ`),
    stat("Optimizer vs simulator", `${p.converged_gap_s.toFixed(3)} s`),
  ].join("");
}

function toBrake(p) {
  const [a, b] = p.straight;
  return p.v.slice(a, b).map((_, k) => (b - (a + k)) * p.ds_m);
}

function straightChart() {
  const datasets = state.data.cars.map((c) => {
    const p = profile(c.id);
    const d = toBrake(p);
    const [a] = p.straight;
    return { label: `${c.label} (${c.detail})`, data: d.map((x, k) => ({ x, y: p.deploy[a + k] - p.harvest[a + k] })),
      borderColor: css(CAR_COLOR[c.id]), borderWidth: c.id === state.car ? 2.6 : 1.4, pointRadius: 0, stepped: true };
  });
  state.charts.straight?.destroy();
  state.charts.straight = new Chart($("straight-chart"), {
    type: "line", data: { datasets },
    options: baseOptions({ scales: {
      x: { type: "linear", reverse: true, min: 0, title: "Distance to the braking point (m); the car moves left to right" },
      y: { title: "Net electric power (kW)" } } }),
  });
  const parts = state.data.cars.map((c) => `${c.label.toLowerCase()} ${profile(c.id).peak_to_brake_m} m`);
  $("straight-note").textContent = `${state.circuit}, ${LIMIT_TEXT[state.limits]}. Each car deploys along the falling ` +
    "MGU-K limit, cuts electric power at its speed peak, and where it still has recharge allowance it harvests in the last stretch " +
    `while still at full throttle (super-clipping, below zero on the chart). Speed peak to braking point: ${parts.join(", ")}.`;
}

function quantile(sorted, q) {
  const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function realChart() {
  const real = state.data.circuits[state.circuit].real;
  const band = css("--band");
  const datasets = [
    { label: "Real laps, 25th percentile", data: real.to_brake_m.map((x, i) => ({ x, y: real.v_q25[i] })), borderColor: "transparent", pointRadius: 0, fill: false },
    { label: "Real laps, middle half", data: real.to_brake_m.map((x, i) => ({ x, y: real.v_q75[i] })), borderColor: "transparent", backgroundColor: band, pointRadius: 0, fill: "-1" },
    { label: `Real clipping laps, median (n = ${real.laps})`, data: real.to_brake_m.map((x, i) => ({ x, y: real.v_median[i] })), borderColor: css("--ink"), borderWidth: 2, pointRadius: 0 },
    ...state.data.cars.map((c) => {
      const p = profile(c.id);
      const [a] = p.straight;
      return { label: `Optimum, ${c.label.toLowerCase()}`, data: toBrake(p).map((x, k) => ({ x, y: p.v[a + k] })),
        borderColor: css(CAR_COLOR[c.id]), borderWidth: 1.6, borderDash: [6, 3], pointRadius: 0 };
    }),
  ];
  state.charts.real?.destroy();
  state.charts.real = new Chart($("real-chart"), {
    type: "line", data: { datasets },
    options: baseOptions({ plugins: { legend: { labels: { color: css("--ink-soft"), boxWidth: 14, filter: (l) => !l.text.includes("25th") } } },
      scales: { x: { type: "linear", reverse: true, min: 0, title: "Distance to the braking point (m)" }, y: { title: "km/h" } } }),
  });
  const peaks = real.peak_to_brake_m;
  const q = [0.25, 0.5, 0.75].map((f) => quantile(peaks, f));
  const rows = state.data.cars.map((c) => {
    const m = profile(c.id).peak_to_brake_m;
    const inside = m >= q[0] && m <= q[2];
    return `<tr><th scope="row">${escapeHTML(c.label)}</th><td>${m} m</td><td class="${inside ? "is-held" : "is-not"}">${inside ? "Inside" : m < q[0] ? "Shorter" : "Longer"}</td></tr>`;
  }).join("");
  const inside = state.data.cars.map((c) => {
    const n = state.data.events.filter((e) => {
      const ps = state.data.circuits[e].real.peak_to_brake_m;
      const m = profile(c.id, state.limits, e).peak_to_brake_m;
      return m >= quantile(ps, 0.25) && m <= quantile(ps, 0.75);
    }).length;
    return `${c.label.toLowerCase()} ${n} of ${state.data.events.length}`;
  });
  $("real-table").innerHTML = `<thead><tr><th scope="col">Speed peak to braking</th><th scope="col">Distance</th><th scope="col">vs real middle half</th></tr></thead>` +
    `<tbody><tr><th scope="row">Real laps (median)</th><td>${Math.round(q[1])} m</td><td>${Math.round(q[0])}–${Math.round(q[2])} m</td></tr>${rows}</tbody>`;
  $("real-note").textContent = `${real.laps} real ${state.circuit} race laps that clipped, against the optimum on the same straight. ` +
    "Both are lined up at the braking point, which the corner fixes for every car, so no assumption is needed about where full throttle began. " +
    `Circuits where the optimum's speed peak falls inside the real middle half (${LIMIT_TEXT[state.limits]}): ${inside.join(", ")}. ` +
    "Exploratory: real laps include traffic, slipstream and Overtake mode, and the comparison was made after the predictions were tested.";
}

function bangChart() {
  const cars = state.data.cars;
  const parts = [["cap", "At the MGU-K limit", "--battery"], ["zero", "Zero", "--ink-soft"], ["between", "In between", "--clip"]];
  state.charts.bang?.destroy();
  state.charts.bang = new Chart($("bang-chart"), {
    type: "bar",
    data: { labels: cars.map((c) => c.label), datasets: parts.map(([k, label, color]) => ({
      label, data: cars.map((c) => profile(c.id).bang_bang[k] * 100), backgroundColor: css(color), borderRadius: 2 })) },
    options: { ...baseOptions({ plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x.toFixed(1)}%` } } },
      scales: { x: { stacked: true, min: 0, max: 100, title: "Share of full-throttle running (%)" }, y: { stacked: true } } }),
    indexAxis: "y", interaction: { mode: "nearest", axis: "y", intersect: false } },
  });
  const between = cars.map((c) => profile(c.id).bang_bang.between * 100);
  $("bang-note").textContent = `${state.circuit}, ${LIMIT_TEXT[state.limits]}. The optimizer may choose any power in 25 kW steps, ` +
    `yet at most ${Math.max(...between).toFixed(1)}% of full-throttle running uses anything other than full power or none ` +
    `(full: at least ${state.data.thresholds.at_cap * 100}% of the limit; none: at most ${state.data.thresholds.at_zero * 100}%).`;
}

function sweepChart() {
  const events = state.data.events, cars = state.data.cars;
  const offset = { original: -0.22, low: 0, high: 0.22 };
  const datasets = cars.map((c) => ({
    label: c.label,
    data: events.flatMap((e, i) => state.data.circuits[e].cars[c.id].sweep.filter((s) => s.converged)
      .map((s, k) => ({ x: i + offset[c.id] + ((k % 9) - 4) * 0.012, y: s.clip_ratio, car: s.car }))),
    backgroundColor: css(CAR_COLOR[c.id]) + "cc", pointRadius: 3.2,
  }));
  state.charts.sweep?.destroy();
  state.charts.sweep = new Chart($("sweep-chart"), {
    type: "scatter", data: { datasets },
    options: baseOptions({
      plugins: { hLine: { y: state.data.thresholds.clip_max, label: "Clips below this line" },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}, sweep car ${c.raw.car}: ${c.raw.y.toFixed(2)}` } } },
      scales: { x: { type: "linear", min: -0.5, max: events.length - 0.5, grid: { display: false },
        afterBuildTicks: (axis) => { axis.ticks = events.map((_, i) => ({ value: i })); },
        ticks: { callback: (v) => events[v] ?? "", font: { weight: 600 } } },
        y: { min: 0, max: 1.05, title: "Power at the speed peak / earlier mean" } },
    }),
  });
  const excluded = cars.reduce((n, c) => n + events.reduce((m, e) => m + state.data.circuits[e].cars[c.id].sweep.filter((s) => !s.converged).length, 0), 0);
  $("sweep-note").textContent = "Each dot is one of 27 legal variations of a car (engine power ±10%, drag ±20%, mass ±35 kg), with the car-file energy limit. " +
    "A ratio at or below 0.5 counts as clipping. The prediction needed at least 80% below the line at every circuit; " +
    `the high-drag end misses it at Australia. ${excluded} results that failed the 0.05 s convergence check are left out and counted as not clipping.`;
}

function update() {
  renderControls();
  lapCharts();
  straightChart();
  realChart();
  bangChart();
  writeHash();
}

async function main() {
  try {
    state.data = await load();
    readHash();
    $("generated").textContent = state.data.generated;
    renderAnswer();
    renderVerdicts();
    sweepChart();
    update();
    $("share").addEventListener("click", async () => {
      await navigator.clipboard?.writeText(location.href);
      $("share").textContent = "Link copied";
      setTimeout(() => { $("share").textContent = "Copy link to this view"; }, 1600);
    });
    window.addEventListener("hashchange", () => { readHash(); update(); });
  } catch (err) {
    $("answer-main").textContent = "The Strategy Lab data could not be loaded.";
    $("answer-detail").textContent = err.message;
    $("answer-detail").classList.add("error");
  }
}

main();
