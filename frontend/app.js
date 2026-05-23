const API_URL = "http://localhost:8000";

const form          = document.getElementById("predict-form");
const resultSection = document.getElementById("result-section");
const errorSection  = document.getElementById("error-section");
const errorMsg      = document.getElementById("error-msg");

let datasetRequested = false;

// Tab switching
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.panel}`).classList.add("active");

    if (btn.dataset.panel === "datos" && !datasetRequested) {
      datasetRequested = true;
      loadDataset(1);
    }
  });
});

loadMetrics();
loadCorrelation();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideResults();

  const btn = form.querySelector("button[type='submit']");
  btn.disabled  = true;
  btn.innerHTML = '<span class="btn-spinner"></span>Prediciendo...';

  const body = {
    temperature:          parseFloat(document.getElementById("temperature").value),
    apparent_temperature: parseFloat(document.getElementById("apparent_temperature").value),
    wind_speed:           parseFloat(document.getElementById("wind_speed").value),
    wind_bearing:         parseFloat(document.getElementById("wind_bearing").value),
    visibility:           parseFloat(document.getElementById("visibility").value),
    pressure:             parseFloat(document.getElementById("pressure").value),
  };

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });

    if (!res.ok) throw new Error(`Error del servidor: ${res.status}`);

    const [data] = await Promise.all([
      res.json(),
      new Promise(r => setTimeout(r, 1500)),
    ]);
    renderResult(data);
  } catch (err) {
    showError(err.message.includes("fetch")
      ? "No se pudo conectar con el servidor. ¿Está corriendo la API?"
      : err.message
    );
  } finally {
    btn.disabled  = false;
    btn.innerHTML = "Predecir";
  }
});

function renderResult({ clase, probabilidades, regla }) {
  const icons  = { Alta: "🔴", Media: "🟡", Baja: "🟢" };
  const rangos = {
    Alta:  "Humedad relativa ≥ 70%",
    Media: "Humedad relativa entre 40% y 69%",
    Baja:  "Humedad relativa < 40%",
  };

  const badge = document.getElementById("clase-badge");
  badge.textContent = `${icons[clase] ?? ""} ${clase}`;
  badge.className   = "badge-" + clase.toLowerCase();
  badge.classList.remove("badge-pop");
  void badge.offsetWidth;
  badge.classList.add("badge-pop");

  document.getElementById("clase-rango").textContent = rangos[clase] ?? "";

  const probsEl = document.getElementById("probs");
  probsEl.innerHTML = "";
  for (const [cls, prob] of Object.entries(probabilidades)) {
    const pct = (prob * 100).toFixed(1);
    probsEl.innerHTML += `
      <div class="prob-row">
        <span class="prob-label">${cls}</span>
        <div class="prob-bar-wrap">
          <div class="prob-bar bar-${cls.toLowerCase()}" style="width:0%" data-target="${pct}%"></div>
        </div>
        <span class="prob-val">${pct}%</span>
      </div>`;
  }

  document.getElementById("regla").textContent = regla.replace(/→/g, "→");

  resultSection.classList.remove("hidden", "reveal");
  void resultSection.offsetWidth;
  resultSection.classList.add("reveal");

  requestAnimationFrame(() => {
    probsEl.querySelectorAll(".prob-bar").forEach(bar => {
      bar.style.width = bar.dataset.target;
    });
  });
}

async function loadMetrics() {
  const container = document.getElementById("metrics-content");
  try {
    const res     = await fetch(`${API_URL}/metrics`);
    if (!res.ok) throw new Error();
    renderMetrics(await res.json(), container);
  } catch {
    container.innerHTML = '<p style="color:#9b9a97;font-size:.85rem">No se pudieron cargar las métricas. Verificá que la API esté corriendo.</p>';
  }
}

function renderMetrics(metrics, container) {
  const summary = `
    <div class="metrics-grid">
      <div class="metric-box"><div class="value">${(metrics.accuracy * 100).toFixed(1)}%</div><div class="label">Accuracy</div></div>
      <div class="metric-box"><div class="value">${(metrics.f1_macro * 100).toFixed(1)}%</div><div class="label">F1 Macro</div></div>
      <div class="metric-box"><div class="value">${(metrics.cv_f1_mean * 100).toFixed(1)}%</div><div class="label">CV F1 (media)</div></div>
      <div class="metric-box"><div class="value">±${(metrics.cv_f1_std * 100).toFixed(1)}%</div><div class="label">CV F1 (desvío)</div></div>
    </div>`;

  const sorted = Object.entries(metrics.feature_importances).sort((a, b) => b[1] - a[1]);
  const maxVal = sorted[0][1];

  const importances = sorted.map(([feat, val]) => `
    <div class="importance-row">
      <span class="importance-label">${feat}</span>
      <div class="importance-bar-wrap">
        <div class="importance-bar" style="width:${(val / maxVal * 100).toFixed(1)}%"></div>
      </div>
      <span class="importance-val">${(val * 100).toFixed(1)}%</span>
    </div>`).join("");

  container.innerHTML = `
    ${summary}
    <h3>Importancia de variables</h3>
    <div style="margin-top:.75rem">${importances}</div>`;
}

async function loadCorrelation() {
  const container = document.getElementById("correlation-matrix");
  try {
    const res = await fetch(`${API_URL}/correlation`);
    if (!res.ok) throw new Error();
    renderCorrelationMatrix(await res.json(), container);
  } catch {
    container.innerHTML = '<p style="color:#9b9a97;font-size:.85rem">No se pudo cargar la matriz de correlación.</p>';
  }
}

function renderCorrelationMatrix({ labels, values }, container) {
  const names = {
    temperature:          "Temperatura",
    apparent_temperature: "Temp. aparente",
    humidity:             "Humedad",
    wind_speed:           "Vel. viento",
    wind_bearing:         "Dir. viento",
    visibility:           "Visibilidad",
    pressure:             "Presión",
  };

  const display = labels.map(l => names[l] || l);

  let html = '<table class="corr-table"><thead><tr><th></th>';
  display.forEach(n => { html += `<th>${n}</th>`; });
  html += '</tr></thead><tbody>';

  values.forEach((row, i) => {
    html += `<tr><th>${display[i]}</th>`;
    row.forEach((val, j) => {
      const bg = corrColor(val, i === j);
      const fg = colorIsDark(bg) ? "#ffffff" : "#37352f";
      html += `<td style="background:${bg};color:${fg}">${val.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

function corrColor(val, diagonal) {
  if (diagonal) return "#f7f6f5";
  if (val >= 0) {
    const t = val;
    return `rgb(${lerp(255,55,t)},${lerp(255,53,t)},${lerp(255,47,t)})`;
  }
  const t = -val;
  return `rgb(${lerp(255,224,t)},${lerp(255,115,t)},${lerp(255,115,t)})`;
}

function lerp(a, b, t) { return Math.round(a + (b - a) * t); }

function colorIsDark(rgb) {
  const [r, g, b] = rgb.match(/\d+/g).map(Number);
  return 0.299 * r + 0.587 * g + 0.114 * b < 140;
}

async function loadDataset(page) {
  const container = document.getElementById("dataset-content");
  container.innerHTML = '<p style="color:#9b9a97;font-size:.85rem">Cargando...</p>';
  try {
    const res = await fetch(`${API_URL}/dataset?page=${page}&page_size=20`);
    if (!res.ok) throw new Error();
    renderDataset(await res.json(), container);
  } catch {
    container.innerHTML = '<p style="color:#9b9a97;font-size:.85rem">No se pudo cargar el dataset.</p>';
  }
}

function renderDataset({ data, page, total, total_pages }, container) {
  const cols = {
    formatted_date:       "Fecha",
    temperature:          "Temp. (°C)",
    apparent_temperature: "Temp. aparente (°C)",
    humidity:             "Humedad",
    wind_speed:           "Vel. viento (km/h)",
    wind_bearing:         "Dir. viento (°)",
    visibility:           "Visibilidad (km)",
    pressure:             "Presión (mbar)",
  };
  const keys = Object.keys(cols);

  let html = '<div class="dataset-table-wrap"><table class="dataset-table"><thead><tr>';
  keys.forEach(k => { html += `<th>${cols[k]}</th>`; });
  html += '</tr></thead><tbody>';

  data.forEach(row => {
    html += '<tr>';
    keys.forEach(k => {
      let val = row[k];
      if (val === null || val === undefined) {
        val = "—";
      } else if (k === "formatted_date") {
        val = String(val).slice(0, 16);
      } else if (k === "humidity") {
        val = (parseFloat(val) * 100).toFixed(1) + "%";
      } else if (typeof val === "number") {
        val = val.toFixed(2);
      }
      html += `<td>${val}</td>`;
    });
    html += '</tr>';
  });

  html += `</tbody></table></div>
    <div class="pagination">
      <span>${total.toLocaleString("es-AR")} registros</span>
      <div class="pagination-controls">
        <button class="btn-page" onclick="loadDataset(${page - 1})" ${page <= 1 ? "disabled" : ""}>← Anterior</button>
        <span>Página ${page.toLocaleString("es-AR")} de ${total_pages.toLocaleString("es-AR")}</span>
        <button class="btn-page" onclick="loadDataset(${page + 1})" ${page >= total_pages ? "disabled" : ""}>Siguiente →</button>
      </div>
    </div>`;

  container.innerHTML = html;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorSection.classList.remove("hidden");
}

function hideResults() {
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");
}
