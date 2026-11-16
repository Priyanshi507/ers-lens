const SCHEMA = 1;

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escapeHTML = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

// lower: whether a smaller value is the one a rule-maker wants, for the wording of the answer.
const METRICS = {
  drop_kmh: { label: "Speed lost at full throttle", unit: "km/h", digits: 1, real: true,
    note: "Speed lost between the peak and the end of full throttle on the circuit's longest straight: what a viewer sees as the car slowing while flat out." },
  superclip_mj: { label: "Super-clipping energy", unit: "MJ", digits: 2,
    note: "Energy harvested per lap while accelerating at full throttle." },
  coast_mj: { label: "Lift-and-coast energy", unit: "MJ", digits: 2,
    note: "Energy harvested per lap with the throttle closed before braking. Zero unless the rule lets the car lift and coast." },
  lap_s: { label: "Lap time", unit: "s", digits: 3,
    note: "The optimal lap time. Compare it only between rules for the same kind of lap: a qualifying lap spends its battery and a 2027 car has different power." },
};

const state = { data: null, a: "race_250", b: "race_350", metric: "drop_kmh", chart: null };
const rule = (id) => state.data.rules.find((r) => r.id === id);

async function load() {
  const res = await fetch("data/rules.json");
  if (!res.ok) throw new Error(`data/rules.json returned HTTP ${res.status}`);
  const data = await res.json();
  if (data.schema !== SCHEMA) throw new Error(`data/rules.json is version ${data.schema}, this page reads ${SCHEMA}`);
  return data;
}

function fillSelect(el, chosen) {
  const groups = new Map();
  for (const r of state.data.rules) {
    if (!groups.has(r.group)) groups.set(r.group, []);
    groups.get(r.group).push(r);
  }
  el.replaceChildren(...[...groups].map(([name, rules]) => {
    const g = document.createElement("optgroup");
    g.label = name;
    g.append(...rules.map((r) => new Option(r.label, r.id, false, r.id === chosen)));
    return g;
  }));
}

// Real laps are shown only for a rule real cars raced under: same session and the same limit.
function realFor(r, event) {
  const real = r.real && state.data.real[event]?.[r.real];
  return real && r.id.endsWith(`_${real.cap_kw}`) ? real : null;
}

function renderAnswer() {
  const a = rule(state.a), b = rule(state.b), m = METRICS[state.metric];
  if (state.a === state.b) {
    $("answer-main").textContent = "Pick two different rules to compare.";
    $("answer-detail").textContent = "";
    return;
  }
  const events = state.data.circuits.map((c) => c.event);
  const eps = 0.5 * 10 ** -m.digits;
  const diff = events.map((e) => b.rows[e][state.metric] - a.rows[e][state.metric]);
  const down = diff.filter((d) => d < -eps).length, up = diff.filter((d) => d > eps).length;
  const words = { drop_kmh: ["loses less speed at full throttle", "more"], superclip_mj: ["super-clips less energy", "more"],
    coast_mj: ["lifts and coasts for less energy", "more"], lap_s: ["is faster", "slower"] }[state.metric];
  $("answer-main").innerHTML = `Under rule B the optimal lap <strong>${words[0]} at ${down} of ${events.length} circuits</strong>` +
    `, ${words[1]} at ${up}` + (events.length - down - up ? ` and the same at ${events.length - down - up}` : "") + ".";
  $("answer-detail").textContent = `A: ${a.label}. B: ${b.label}. ` +
    (a.group === b.group ? "" : "These are different kinds of lap, so more than one thing differs between them.");
}

function renderMetricPicker() {
  $("pick-metric").replaceChildren(...Object.entries(METRICS).map(([key, m]) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = m.label;
    btn.setAttribute("aria-pressed", String(key === state.metric));
    btn.addEventListener("click", () => { state.metric = key; update(); });
    return btn;
  }));
  $("metric-note").textContent = METRICS[state.metric].note;
}

function renderChart() {
  const a = rule(state.a), b = rule(state.b), m = METRICS[state.metric], k = state.metric;
  const cs = [...state.data.circuits].sort((x, y) => b.rows[y.event][k] - b.rows[x.event][k]);
  const grid = css("--grid"), ink = css("--ink-soft");
  const datasets = [
    { label: "Rule A", data: cs.map((c) => a.rows[c.event][k]), backgroundColor: ink, borderRadius: 3 },
    { label: "Rule B", data: cs.map((c) => b.rows[c.event][k]), backgroundColor: css("--battery"), borderRadius: 3 },
  ];
  const real = m.real ? cs.map((c) => (realFor(b, c.event) || realFor(a, c.event))?.drop_kmh ?? null) : [];
  if (real.some((v) => v !== null)) {
    datasets.push({ label: "Real laps (median)", data: real, type: "line", order: -1, showLine: false, pointStyle: "rectRot",
      pointRadius: 6, pointBackgroundColor: css("--clip"), borderColor: css("--clip") });
  }
  const lap = k === "lap_s";
  state.chart?.destroy();
  state.chart = new Chart($("rule-chart"), {
    type: "bar",
    data: { labels: cs.map((c) => c.label), datasets },
    options: {
      animation: false, maintainAspectRatio: false, indexAxis: "y",
      interaction: { mode: "nearest", axis: "y", intersect: false },
      plugins: { legend: { labels: { color: ink, boxWidth: 14 } },
        tooltip: { callbacks: { label: (t) => `${t.dataset.label}: ${t.parsed.x.toFixed(m.digits)} ${m.unit}` } } },
      scales: {
        x: { grid: { color: grid }, ticks: { color: ink }, border: { color: grid }, beginAtZero: !lap,
          min: lap ? 60 : undefined, title: { display: true, color: ink, text: `${m.label} (${m.unit})` } },
        y: { grid: { color: grid }, ticks: { color: ink }, border: { color: grid } },
      },
    },
  });
}

function renderTable() {
  const a = rule(state.a), b = rule(state.b), m = METRICS[state.metric], k = state.metric;
  const showReal = m.real && (a.real || b.real);
  const f = (v) => v.toFixed(m.digits);
  const head = `<thead><tr><th scope="col">Circuit</th><th scope="col">A (${m.unit})</th><th scope="col">B (${m.unit})</th>` +
    `<th scope="col">B − A</th>${showReal ? '<th scope="col">Real laps, under the limit they ran</th>' : ""}</tr></thead>`;
  const body = state.data.circuits.map((c) => {
    const va = a.rows[c.event][k], vb = b.rows[c.event][k], d = vb - va;
    const real = realFor(b, c.event) || realFor(a, c.event);
    return `<tr><th scope="row">${escapeHTML(c.label)}</th><td>${f(va)}</td><td>${f(vb)}</td>` +
      `<td>${d > 0 ? "+" : ""}${f(d)}</td>` +
      (showReal ? `<td>${real ? `${real.drop_kmh.toFixed(1)} km/h, ${real.laps} laps` : "—"}</td>` : "") + "</tr>";
  }).join("");
  $("rule-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

function update() {
  history.replaceState(null, "", `#${state.a},${state.b},${state.metric}`);
  renderAnswer();
  renderMetricPicker();
  renderChart();
  renderTable();
}

async function main() {
  try {
    state.data = await load();
  } catch (err) {
    $("answer-main").textContent = `The rule data could not be loaded: ${err.message}`;
    return;
  }
  const [a, b, metric] = decodeURIComponent(location.hash.slice(1)).split(",");
  if (rule(a) && rule(b)) Object.assign(state, { a, b });
  if (METRICS[metric]) state.metric = metric;
  $("generated").textContent = state.data.generated;
  fillSelect($("rule-a"), state.a);
  fillSelect($("rule-b"), state.b);
  $("rule-a").addEventListener("change", (e) => { state.a = e.target.value; update(); });
  $("rule-b").addEventListener("change", (e) => { state.b = e.target.value; update(); });
  update();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderChart);
}

main();
