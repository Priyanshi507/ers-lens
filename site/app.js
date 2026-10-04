const SCHEMA = 4;

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seconds = (s) => `${Math.max(0, s).toFixed(2)} s`;
const shortName = (name) => name.replace(/ Grand Prix$/, "");
const escapeHTML = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

const state = { index: null, race: null, cache: new Map(), charts: {} };

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} returned HTTP ${res.status}`);
  const data = await res.json();
  if (data.schema !== SCHEMA) {
    throw new Error(`${path} uses data version ${data.schema}, but this page reads version ${SCHEMA}`);
  }
  return data;
}

function readHash() {
  const [race, driver, lap] = location.hash.slice(1).split("/").map(decodeURIComponent);
  return { race, driver, lap: lap ? Number(lap) : undefined };
}

function writeHash(...parts) {
  history.replaceState(null, "", `#${parts.filter((p) => p !== undefined).map(encodeURIComponent).join("/")}`);
}

function renderRounds() {
  const list = $("rounds");
  list.replaceChildren(...state.index.races.map((race, i) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "round";
    button.dataset.id = race.id;
    button.setAttribute("aria-pressed", "false");
    const value = race.reported ? `${seconds(race.loss_median_s)} per lap` : "Not reported";
    button.innerHTML = `<span class="round-number">Round ${i + 1}</span>
      <span class="round-name">${escapeHTML(shortName(race.name))}</span>
      <span class="round-value${race.reported ? "" : " is-unreported"}">${value}</span>`;
    button.addEventListener("click", () => selectRace(race.id));
    item.append(button);
    return item;
  }));
}

function renderAnswer(race) {
  const box = document.querySelector(".answer");
  box.classList.toggle("is-unreported", !race.reported);
  if (!race.reported) {
    $("answer-main").innerHTML = `The ${escapeHTML(race.name)} is <strong>not reported</strong>.`;
    $("answer-detail").textContent = `${race.reason} A number here would be misleading, so the site leaves it out.`;
    return;
  }
  const straight = race.straight ? `Measured on the longest straight, ${race.straight.length_m.toLocaleString("en")} m. ` : "";
  $("answer-main").innerHTML = `At the ${escapeHTML(race.name)}, a typical car lost
    <strong>${seconds(race.loss_median_s)}</strong> per lap to clipping.`;
  $("answer-detail").textContent = `${straight}Half of all laps lost between ${seconds(race.loss_q25_s)} and ` +
    `${seconds(race.loss_q75_s)}. ${Math.round(race.clipping_share * 100)}% of the ` +
    `${race.measured_laps.toLocaleString("en")} measured laps showed clipping. ` +
    `Each lap's estimate is accurate to about ±${state.index.accuracy.typical_lap_s} s.`;
}

const errorBand = {
  id: "errorBand",
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea: area, scales: { x } } = chart;
    const band = state.index.accuracy.team_band_s;
    const left = x.getPixelForValue(-band);
    ctx.save();
    ctx.fillStyle = css("--band");
    ctx.fillRect(left, area.top, x.getPixelForValue(band) - left, area.bottom - area.top);
    ctx.restore();
  },
};

function renderTeams(race) {
  const teams = race.teams;
  const few = race.teams_too_few_laps;
  const band = state.index.accuracy.team_band_s;
  $("teams-note").textContent = "Each bar is a team's median compared with the median of all laps " +
    "that clipped in this race. A car's engine power and drag aren't public, and that unknown setup " +
    `can shift every lap of the same car by up to about ${band} s. Bars inside the grey band may not ` +
    "be real differences, so treat this as exploratory, not a ranking.";
  $("teams-left-out").textContent = few.length
    ? `Not shown, fewer than ${race.min_team_laps} measured clipping laps: ` +
      `${few.map((t) => `${t.team} (${t.laps})`).join(", ")}.`
    : "";
  state.charts.teams?.destroy();
  const values = teams.map((t) => t.vs_race_s);
  state.charts.teams = new Chart($("team-chart"), {
    type: "bar",
    data: {
      labels: teams.map((t) => `${t.team} (${t.laps} laps)`),
      datasets: [{
        data: values,
        backgroundColor: values.map((v) => (v > 0 ? css("--clip") : css("--battery"))),
        borderRadius: 2,
        barPercentage: 0.7,
      }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => {
              const t = teams[c.dataIndex];
              const sign = t.vs_race_s > 0 ? "+" : "";
              return `${sign}${t.vs_race_s.toFixed(3)} s vs race median (median ${seconds(t.loss_median_s)}, ${t.laps} laps)`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Time lost vs race median (s per lap): more lost to the right" },
          grid: { color: css("--grid") },
          ticks: { callback: (v) => `${v > 0 ? "+" : ""}${Number(v).toFixed(2)}` },
        },
        y: { grid: { display: false } },
      },
    },
    plugins: [errorBand],
  });
}

function lapsFor(driver) {
  return state.race.laps.filter((l) => l.driver === driver && l.trace).sort((a, b) => a.lap - b.lap);
}

function typicalLap(laps) {
  const sorted = [...laps].sort((a, b) => a.loss_s - b.loss_s);
  return sorted[Math.floor(sorted.length / 2)];
}

function raceTypicalLap() {
  const target = state.race.loss_median_s;
  const candidates = state.race.laps.filter((l) => l.trace && l.clipping);
  return candidates.reduce((best, l) =>
    (Math.abs(l.loss_s - target) < Math.abs(best.loss_s - target) ? l : best), candidates[0]);
}

function fillDrivers(selected) {
  const seen = new Map();
  for (const l of state.race.laps) if (l.trace && !seen.has(l.driver)) seen.set(l.driver, l.team);
  const drivers = [...seen].sort((a, b) => a[1].localeCompare(b[1]) || a[0].localeCompare(b[0]));
  $("driver").replaceChildren(...drivers.map(([d, team]) => new Option(`${d} (${team})`, d)));
  $("driver").value = seen.has(selected) ? selected : drivers[0]?.[0];
}

function fillLaps(selected) {
  const laps = lapsFor($("driver").value);
  $("lap").replaceChildren(...laps.map((l) =>
    new Option(`Lap ${l.lap}: ${l.clipping ? seconds(l.loss_s) : "no clipping"}`, l.lap)));
  const pick = laps.find((l) => l.lap === selected) ?? typicalLap(laps);
  if (pick) $("lap").value = pick.lap;
}

const flatShade = {
  id: "flatShade",
  beforeDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea: area, scales: { x } } = chart;
    const left = x.getPixelForValue(opts.from);
    ctx.save();
    ctx.fillStyle = css("--battery-wash");
    ctx.fillRect(left, area.top, x.getPixelForValue(opts.to) - left, area.bottom - area.top);
    ctx.restore();
  },
};

function renderLap() {
  const driver = $("driver").value;
  const lap = lapsFor(driver).find((l) => l.lap === Number($("lap").value));
  if (!lap) return;
  writeHash(state.race.id, driver, lap.lap);

  const range = `with a 90% range of ${seconds(lap.loss_low_s)} to ${seconds(lap.loss_high_s)}`;
  const drop = lap.power_drop_kw
    ? ` Electric power fell by about ${lap.power_drop_kw} kW after the speed peak.`
    : lap.power_drop_implausible
      ? " The power drop on this lap came out above the motor's 350 kW limit, so it isn't shown."
      : "";
  $("lap-summary").innerHTML = lap.clipping
    ? `${escapeHTML(driver)} lost <strong>${seconds(lap.loss_s)}</strong> to clipping on lap ${lap.lap}, ${range}.${drop}`
    : `${escapeHTML(driver)} didn't clip on lap ${lap.lap}: speed kept rising until the driver lifted.`;

  const { d, v, flat, peak } = lap.trace;
  const points = d.map((x, i) => ({ x, y: v[i] }));
  state.charts.lap?.destroy();
  state.charts.lap = new Chart($("lap-chart"), {
    type: "scatter",
    data: {
      datasets: [
        { data: points, showLine: true, borderColor: css("--ink"), borderWidth: 2, pointRadius: 0, pointHitRadius: 8 },
        ...(lap.clipping ? [{ data: [points[peak]], pointRadius: 6, pointBackgroundColor: css("--clip"), pointBorderWidth: 0 }] : []),
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        flatShade: { from: d[flat[0]], to: d[flat[1]] },
        tooltip: { callbacks: { label: (c) => `${c.parsed.x} m: ${c.parsed.y} km/h` } },
      },
      scales: {
        x: { type: "linear", title: { display: true, text: "Distance along the straight (m)" }, grid: { color: css("--grid") } },
        y: { title: { display: true, text: "Speed (km/h)" }, grid: { color: css("--grid") } },
      },
    },
    plugins: [flatShade],
  });
}

async function loadRace(id) {
  if (!state.cache.has(id)) state.cache.set(id, getJSON(`data/races/${id}.json`));
  return state.cache.get(id);
}

async function selectRace(id, wanted = {}) {
  document.querySelectorAll(".round").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  try {
    state.race = await loadRace(id);
  } catch (err) {
    state.cache.delete(id);
    showError(err);
    return;
  }
  renderAnswer(state.race);
  const hasLaps = state.race.reported && state.race.laps.some((l) => l.trace);
  $("teams").hidden = !state.race.reported || state.race.teams.length === 0;
  $("laps").hidden = !hasLaps;
  if (!$("teams").hidden) renderTeams(state.race);
  if (hasLaps) {
    const typical = wanted.driver ? wanted : raceTypicalLap() ?? {};
    fillDrivers(typical.driver);
    fillLaps(typical.lap);
    renderLap();
  } else {
    writeHash(id);
  }
}

function showError(err) {
  $("answer-main").innerHTML = `<span class="error">Couldn't load the race data.</span>`;
  $("answer-detail").textContent = `${err.message}. Reload the page to try again.`;
  console.error(err);
}

async function copyLink() {
  const button = $("share");
  try {
    await navigator.clipboard.writeText(location.href);
    button.textContent = "Link copied";
  } catch {
    button.textContent = "Copy the address bar instead";
  }
  setTimeout(() => { button.textContent = "Copy link to this lap"; }, 2000);
}

async function main() {
  if (typeof Chart === "undefined") {
    showError(new Error("The chart library failed to load"));
    return;
  }
  Chart.defaults.font.family = css("--body");
  Chart.defaults.color = css("--ink-soft");
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) Chart.defaults.animation = false;

  $("driver").addEventListener("change", () => { fillLaps(); renderLap(); });
  $("lap").addEventListener("change", renderLap);
  $("share").addEventListener("click", copyLink);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.race) selectRace(state.race.id, { driver: $("driver").value, lap: Number($("lap").value) });
  });

  try {
    state.index = await getJSON("data/index.json");
  } catch (err) {
    showError(err);
    return;
  }
  $("generated").textContent = state.index.generated;
  renderRounds();
  const wanted = readHash();
  const races = state.index.races;
  const start = races.find((r) => r.id === wanted.race) ?? [...races].reverse().find((r) => r.reported) ?? races[0];
  selectRace(start.id, start.id === wanted.race ? wanted : {});
}

main();
