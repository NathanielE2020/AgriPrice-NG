(function () {
  "use strict";

  const state = {
    overview: null,
    series: { aggregate: [], market: [], world_bank: [] },
    dataset: document.body && document.body.dataset.deployScope === "approved_market" ? "market" : "aggregate",
    selectedSeries: { aggregate: null, market: null, world_bank: null },
    history: null,
    forecast: null,
    charts: {},
    toastTimer: null,
  };

  const modelNames = {
    naive_last_observation: "Naive last observation",
    random_forest: "Random Forest",
    gradient_boosting: "Gradient Boosting",
    xgboost: "XGBoost",
    stacking: "Hybrid Stacking",
    lstm: "LSTM",
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const datasetLabels = {
    aggregate: "Aggregate",
    market: "Market-level",
    world_bank: "World Bank estimates",
  };

  function configureDatasetSelectors(availableDatasets) {
    const allowed = new Set(availableDatasets || []);
    ["#globalDataset", "#analysisDataset", "#forecastDataset"].forEach((selector) => {
      const select = $(selector);
      if (!select) return;
      const current = allowed.has(select.value) ? select.value : null;
      select.innerHTML = Object.keys(datasetLabels)
        .filter((dataset) => allowed.has(dataset))
        .map((dataset) => `<option value="${dataset}">${datasetLabels[dataset]}</option>`)
        .join("");
      if (current) select.value = current;
    });
    if (!allowed.has(state.dataset)) state.dataset = availableDatasets && availableDatasets.length ? availableDatasets[0] : "market";
  }

  function escapeHTML(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatNumber(value, digits = 0) {
    if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
    return Number(value).toLocaleString("en-NG", { maximumFractionDigits: digits, minimumFractionDigits: digits });
  }

  function formatPrice(value) {
    if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
    return `₦${formatNumber(value, 2)}`;
  }

  function formatDate(month) {
    if (!month) return "—";
    const [year, monthNumber] = String(month).split("-").map(Number);
    if (!year || !monthNumber) return month;
    return new Date(year, monthNumber - 1, 1).toLocaleDateString("en-NG", { month: "short", year: "numeric" });
  }

  function formatObservationDate(value) {
    if (!value) return "—";
    const text = String(value);
    if (/^\d{4}-\d{2}-\d{2}/.test(text)) {
      const parsed = new Date(`${text.slice(0, 10)}T00:00:00`);
      if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleDateString("en-NG", { day: "2-digit", month: "short", year: "numeric" });
    }
    return formatDate(text);
  }

  function formatModel(model, fallback) {
    return fallback || modelNames[model] || model || "—";
  }

  function showToast(message, isError = false) {
    const toast = $("#toastMessage");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast-message visible${isError ? " error" : ""}`;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { toast.className = "toast-message"; }, 3500);
  }

  function setInlineError(selector, message) {
    const node = $(selector);
    if (!node) return;
    node.textContent = message || "The API could not complete this request.";
    node.hidden = !message;
  }

  async function apiGet(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    let payload;
    try { payload = await response.json(); } catch (error) { payload = {}; }
    if (!response.ok) {
      const detail = payload && payload.error && payload.error.message ? payload.error.message : "The dashboard API returned an error.";
      const failure = new Error(detail);
      failure.status = response.status;
      throw failure;
    }
    return payload;
  }

  function destroyChart(name) {
    if (state.charts[name]) {
      state.charts[name].destroy();
      state.charts[name] = null;
    }
  }

  const forecastBoundaryPlugin = {
    id: "forecastBoundary",
    afterDraw(chart, args, options) {
      if (!options || options.index === undefined || !chart.scales.x) return;
      const x = chart.scales.x.getPixelForValue(options.index);
      const top = chart.chartArea.top;
      const bottom = chart.chartArea.bottom;
      const context = chart.ctx;
      context.save();
      context.setLineDash([4, 4]);
      context.strokeStyle = "#c89d42";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, bottom);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = "#977025";
      context.font = "600 9px DM Sans, sans-serif";
      context.fillText("Forecast starts", Math.min(x + 6, chart.chartArea.right - 76), top + 12);
      context.restore();
    },
  };

  function chartOptions(yTitle = "Price (NGN)") {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#173329",
          padding: 11,
          titleFont: { family: "DM Sans", size: 11, weight: "600" },
          bodyFont: { family: "DM Sans", size: 11 },
          callbacks: { label: (context) => `${context.dataset.label || "Price"}: ${formatPrice(context.parsed.y)}` },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#87958d", font: { family: "DM Sans", size: 10 }, maxTicksLimit: 8, maxRotation: 0 },
          border: { display: false },
        },
        y: {
          beginAtZero: false,
          grid: { color: "#edf1ee" },
          ticks: { color: "#87958d", font: { family: "DM Sans", size: 10 }, callback: (value) => `₦${Number(value).toLocaleString("en-NG", { notation: "compact", maximumFractionDigits: 1 })}` },
          border: { display: false },
          title: { display: true, text: yTitle, color: "#87958d", font: { family: "DM Sans", size: 10, weight: "500" } },
        },
      },
    };
  }

  function getSeries(dataset) {
    return state.series[dataset] || [];
  }

  function getSeriesItem(dataset, seriesId) {
    return getSeries(dataset).find((item) => item.series_id === seriesId) || null;
  }

  function renderLatestObservation(series, target = "analysis") {
    if (!series) return;
    const latestPrice = series.latest_actual_price;
    const latestDate = series.latest_observation_date || series.last_observation_date || series.last_observed_month;
    const age = series.observation_age_months;
    const freshness = series.freshness_status || "unavailable";
    const label = freshness.charAt(0).toUpperCase() + freshness.slice(1);
    const priceText = latestPrice === null || latestPrice === undefined ? "No recorded price available" : `${formatPrice(latestPrice)} · ${series.unit || ""}`;
    const location = series.geography_name || series.market || series.state || "";
    const context = `${series.commodity || "Selected series"}${location ? ` · ${location}` : ""} · ${series.price_type || ""}${series.source === "world_bank" ? " · World Bank model estimate" : ""}`;
    const estimateNote = series.source === "world_bank"
      ? `World Bank close estimate; trust ${formatNumber(series.latest_trust_score, 1)}. Low/high: ${formatPrice(series.latest_low_estimate)} — ${formatPrice(series.latest_high_estimate)}. This is modelled/imputed, not a direct survey price.`
      : context;
    if (target === "forecast") {
      $("#forecastLatestPrice").textContent = series.source === "world_bank" ? `Latest World Bank estimate: ${priceText}` : `Latest available recorded price: ${priceText}`;
      $("#forecastLatestContext").textContent = estimateNote;
      $("#forecastLatestDate").textContent = latestDate || "Unavailable";
      $("#forecastLatestAge").textContent = age === null || age === undefined ? "Unavailable" : `${age} month${Number(age) === 1 ? "" : "s"}`;
      const badge = $("#forecastFreshness");
      badge.textContent = label;
      badge.className = `freshness-badge freshness-${freshness}`;
      return;
    }
    $("#analysisLatestPrice").textContent = latestPrice === null || latestPrice === undefined ? "No price estimate available" : `${formatPrice(latestPrice)} · ${series.unit || ""}`;
    $("#analysisLatestContext").textContent = estimateNote;
    $("#analysisLatestDate").textContent = latestDate || "Unavailable";
    $("#analysisLatestAge").textContent = age === null || age === undefined ? "Unavailable" : `${age} month${Number(age) === 1 ? "" : "s"}`;
    const badge = $("#analysisFreshness");
    badge.textContent = label;
    badge.className = `freshness-badge freshness-${freshness}`;
    $("#analysisFreshnessExplanation").textContent = series.freshness_explanation || "Data age describes recency only; it does not indicate price accuracy.";
  }

  function setHistoryDateDefaults(series) {
    if (!series) return;
    const start = $("#analysisStartDate");
    const end = $("#analysisEndDate");
    if (start) start.value = series.first_observed_month || "";
    if (end) end.value = series.last_observed_month || "";
  }

  function populateSeriesSelect(select, dataset, preferred) {
    if (!select) return null;
    const items = getSeries(dataset);
    if (!items.length) {
      select.innerHTML = "<option value=\"\">No series available</option>";
      select.disabled = true;
      return null;
    }
    select.disabled = false;
    const current = preferred && items.some((item) => item.series_id === preferred) ? preferred : items[0].series_id;
    select.innerHTML = items.map((item) => {
      const location = item.geography_name || item.market || item.state || "";
      const label = `${item.commodity} · ${item.unit} · ${item.price_type}${location ? ` · ${location}` : ""}`;
      return `<option value="${escapeHTML(item.series_id)}"${item.series_id === current ? " selected" : ""}>${escapeHTML(label)}</option>`;
    }).join("");
    return current;
  }

  function syncSeriesControls(dataset, preferred) {
    const current = preferred || state.selectedSeries[dataset] || null;
    const selected = populateSeriesSelect($("#analysisSeries"), dataset, current);
    populateSeriesSelect($("#forecastSeries"), dataset, selected);
    populateSeriesSelect($("#performanceSeries"), dataset, selected);
    state.selectedSeries[dataset] = selected;
    [$("#analysisDataset"), $("#forecastDataset"), $("#globalDataset")].forEach((select) => { if (select) select.value = dataset; });
    updateForecastModels(dataset, selected);
    const selectedSeries = getSeriesItem(dataset, selected);
    setHistoryDateDefaults(selectedSeries);
    renderLatestObservation(selectedSeries, "analysis");
    renderLatestObservation(selectedSeries, "forecast");
  }

  function updateForecastModels(dataset, seriesId) {
    const select = $("#forecastModel");
    const note = $("#modelAvailabilityNote");
    if (!select) return;
    const series = getSeriesItem(dataset, seriesId);
    const models = (series && series.models ? series.models : []).filter((item) => item.available);
    select.innerHTML = models.length ? models.map((item) => `<option value="${escapeHTML(item.model)}">${escapeHTML(item.label)}</option>`).join("") : "<option value=\"\">No model available</option>";
    select.disabled = !models.length;
    if (note) note.textContent = models.length ? `${models.length} validated model${models.length === 1 ? "" : "s"} available for this series.` : "No validated forecast artifact is available for this series.";
  }

  function renderOverview(overview) {
    const worldBank = overview.sources && overview.sources.world_bank && overview.sources.world_bank.available ? overview.sources.world_bank : null;
    const facts = state.dataset === "world_bank" && worldBank ? worldBank : (overview.dataset || {});
    const modelStatus = state.dataset === "world_bank"
      ? (overview.model_status || {}).world_bank
      : state.dataset === "aggregate"
        ? (overview.model_status || {}).aggregate_final_validation
        : (overview.model_status || {}).market;
    const modelFamilies = Object.keys(modelStatus || {}).length;
    const stats = [
      [state.dataset === "world_bank" ? "World Bank estimates" : "Recorded observations", formatNumber(facts.historical_observations), state.dataset === "world_bank" ? "Normalized close estimates" : "Across the source dataset", "leaf"],
      ["Commodities", formatNumber(facts.commodity_count), "Represented in the data", "grain"],
      [state.dataset === "world_bank" ? "Actual markets" : "Observed markets", formatNumber(facts.market_count), state.dataset === "world_bank" ? "Aggregate labels excluded" : "Market locations", "market"],
      ["States", formatNumber(facts.state_count), state.dataset === "world_bank" ? "Distinct source labels" : "With observed records", "state"],
      ["Model families", formatNumber(modelFamilies), `Available in ${state.dataset} view`, "model"],
    ];
    const grid = $("#statGrid");
    if (grid) grid.innerHTML = stats.map(([label, value, sub, icon]) => `<div class="stat-card"><span class="stat-label">${label}</span><div class="stat-value">${value}</div><div class="stat-sub">${sub}</div></div>`).join("");
    $("#latestDate").textContent = facts.latest_observation_date || facts.date_end || "Unavailable";
    $("#overallDateRange").textContent = facts.earliest_observation_date && facts.latest_observation_date
      ? `${facts.earliest_observation_date} — ${facts.latest_observation_date}`
      : "Overall range unavailable";
    $("#frequencyValue").textContent = facts.frequency ? facts.frequency.charAt(0).toUpperCase() + facts.frequency.slice(1) : "Unavailable";
    $("#currencyValue").textContent = facts.currency || "Unavailable";
    $("#aboutCoverage").textContent = facts.date_start && facts.date_end ? `${facts.date_start}–${facts.date_end}` : "Unavailable";
    $("#aboutFrequency").textContent = facts.frequency ? facts.frequency.charAt(0).toUpperCase() + facts.frequency.slice(1) : "Unavailable";
    $("#aboutCurrency").textContent = facts.currency || "Unavailable";
    $("#footerStatus").textContent = "API connected";
  }

  function renderSeriesTiles() {
    const grid = $("#overviewSeriesGrid");
    if (!grid) return;
    const dataset = state.dataset;
    const items = getSeries(dataset);
    $("#seriesCount").textContent = `${items.length} series`;
    if (!items.length) { grid.innerHTML = '<div class="loading-line">No approved series are available for this view.</div>'; return; }
    grid.innerHTML = items.map((item) => { const location = item.geography_name || item.market || item.state || ""; const estimate = dataset === "world_bank" ? " · model estimate" : ""; return `<button class="series-tile" type="button" data-overview-series="${escapeHTML(item.series_id)}"><span class="series-tile-heading"><strong>${escapeHTML(item.commodity)}</strong><span class="series-type">${escapeHTML(dataset)}</span></span><small>${escapeHTML(item.unit)} · ${escapeHTML(item.price_type)}${location ? ` · ${escapeHTML(location)}` : ""}${estimate}</small><small>${formatNumber(item.observed_months)} observed months · ${escapeHTML(item.first_observed_month)} — ${escapeHTML(item.last_observed_month)}</small></button>`; }).join("");
    $$("[data-overview-series]").forEach((button) => button.addEventListener("click", () => {
      state.selectedSeries[dataset] = button.dataset.overviewSeries;
      syncSeriesControls(dataset, state.selectedSeries[dataset]);
      loadHistory(dataset, state.selectedSeries[dataset]);
      showSection("analysis-section");
    }));
  }

  function setChartEmpty(selector, visible) {
    const node = $(selector);
    if (node) node.hidden = !visible;
  }

  function renderHistoryChart(history, series) {
    destroyChart("history");
    destroyChart("overview");
    const labels = history.map((row) => row.month);
    const values = history.map((row) => row.observed && row.price !== null ? row.price : null);
    const config = {
      type: "line",
      data: { labels, datasets: [{ label: "Observed price", data: values, borderColor: "#367c5b", backgroundColor: "rgba(54,124,91,.10)", borderWidth: 2.2, pointRadius: 2.2, pointHoverRadius: 4, pointBackgroundColor: "#367c5b", fill: true, tension: .25, spanGaps: false }] },
      options: chartOptions(),
    };
    const overviewConfig = { ...config, data: { ...config.data, datasets: [{ ...config.data.datasets[0], pointRadius: 1.8, borderWidth: 2, fill: false }] }, options: { ...chartOptions(), plugins: { ...chartOptions().plugins, tooltip: config.options.plugins.tooltip } } };
    if (window.Chart && labels.length) {
      state.charts.history = new Chart($("#historyChart"), config);
      state.charts.overview = new Chart($("#overviewChart"), overviewConfig);
      setChartEmpty("#historyEmpty", false);
      setChartEmpty("#overviewChartEmpty", false);
    } else {
      setChartEmpty("#historyEmpty", true);
      setChartEmpty("#overviewChartEmpty", true);
    }
    const observedCount = history.filter((row) => row.observed && row.price !== null).length;
    const title = `${series.commodity} · ${series.unit} · ${series.price_type}${series.market ? ` · ${series.market}` : ""}`;
    $("#historyTitle").textContent = title;
    $("#overviewChartTitle").textContent = `${series.commodity} price movement`;
    $("#overviewChartMeta").textContent = `${formatNumber(observedCount)} observed months · ${series.unit} · ${series.price_type}`;
    $("#historyCoverage").textContent = `${series.first_observed_month} — ${series.last_observed_month}`;
    renderLatestObservation(series, "analysis");
  }

  function renderHistoryTable(history, series) {
    const body = $("#historyTableBody");
    const observedCount = history.filter((row) => row.observed && row.price !== null).length;
    $("#historyCount").textContent = `${formatNumber(observedCount)} observed`;
    if (!body) return;
    body.innerHTML = history.slice().reverse().map((row) => {
      const status = row.observed && row.price !== null;
      const marketCount = row.n_markets !== undefined && row.n_markets !== null ? formatNumber(row.n_markets) : (row.geography_name || row.market || row.state || "—");
      const estimated = series.source === "world_bank";
      return `<tr><td>${escapeHTML(formatObservationDate(row.date || row.month))}</td><td>${escapeHTML(series.commodity)}</td><td class="price-cell${status ? "" : " muted-cell"}">${status ? escapeHTML(formatPrice(row.price)) : "—"}</td><td>${escapeHTML(series.unit)}</td><td>${escapeHTML(series.price_type)}</td><td>${escapeHTML(series.market || marketCount)}</td><td class="${status ? "status-observed" : "status-missing"}">${status ? (estimated ? "Estimated" : "Observed") : "Missing"}</td></tr>`;
    }).join("");
  }

  async function loadHistory(dataset, seriesId) {
    const series = getSeriesItem(dataset, seriesId);
    if (!series) return;
    state.selectedSeries[dataset] = seriesId;
    try {
      const params = { dataset, series_id: seriesId };
      if (dataset === $("#analysisDataset").value) {
        const startDate = $("#analysisStartDate").value;
        const endDate = $("#analysisEndDate").value;
        if (startDate) params.start_date = startDate;
        if (endDate) params.end_date = endDate;
        if (startDate && endDate && startDate > endDate) {
          throw new Error("Start month must not be later than end month.");
        }
      }
      const query = new URLSearchParams(params);
      const payload = await apiGet(`/api/history?${query.toString()}`);
      state.history = payload;
      renderHistoryChart(payload.history || [], payload.series || series);
      renderHistoryTable(payload.history || [], payload.series || series);
      renderLatestObservation(payload.series || series, "analysis");
      const requested = payload.requested_range || {};
      if (requested.start_month || requested.end_month) {
        $("#historyCoverage").textContent = `${requested.start_month || series.first_observed_month} — ${requested.end_month || series.last_observed_month}`;
      }
    } catch (error) {
      setInlineError("#forecastError", "");
      showToast(error.message, true);
      $("#overviewChartMeta").textContent = "History unavailable";
      setChartEmpty("#historyEmpty", true);
      setChartEmpty("#overviewChartEmpty", true);
    }
  }

  function renderForecastChart(payload) {
    destroyChart("forecast");
    const historical = (payload.historical || []).map((row) => ({ month: row.month, price: row.observed ? row.price : null }));
    const forecast = payload.forecast || [];
    const labels = historical.map((row) => row.month).concat(forecast.map((row) => row.forecast_month));
    const actualValues = historical.map((row) => row.price).concat(forecast.map(() => null));
    const forecastValues = historical.map(() => null).concat(forecast.map((row) => row.forecast_price));
    const boundaryIndex = historical.length;
    const options = chartOptions();
    options.plugins.forecastBoundary = { index: boundaryIndex };
    const config = {
      type: "line",
      data: { labels, datasets: [
        { label: "Observed price", data: actualValues, borderColor: "#367c5b", backgroundColor: "rgba(54,124,91,.08)", borderWidth: 2.2, pointRadius: 2, pointHoverRadius: 4, pointBackgroundColor: "#367c5b", fill: true, tension: .25, spanGaps: false },
        { label: "Forecast estimate", data: forecastValues, borderColor: "#c89d42", backgroundColor: "rgba(200,157,66,.08)", borderWidth: 2.2, borderDash: [6, 4], pointRadius: 3, pointHoverRadius: 5, pointBackgroundColor: "#c89d42", fill: false, tension: .2, spanGaps: false },
      ] },
      options,
      plugins: [forecastBoundaryPlugin],
    };
    if (window.Chart) state.charts.forecast = new Chart($("#forecastChart"), config);
  }

  function renderForecastMeta(payload) {
    const metadata = payload.metadata || {};
    const series = payload.series || {};
    renderLatestObservation(series, "forecast");
    const warning = $("#forecastFreshnessWarning");
    if (warning) {
      warning.textContent = payload.freshness_warning || (payload.freshness_explanation && payload.freshness_status !== "fresh" ? payload.freshness_explanation : "");
      warning.hidden = !warning.textContent;
    }
    const entries = [
      ["Commodity", metadata.commodity || series.commodity],
      ["Unit", metadata.unit || series.unit],
      ["Price type", metadata.price_type || series.price_type],
      ["Model", payload.model_label || formatModel(metadata.model || payload.model)],
      ["Transformation", metadata.transformation || payload.transformation],
      ["Origin month", metadata.origin_month || payload.origin_month],
      ["Last actual date", payload.latest_observation_date || metadata.latest_observation_date || series.latest_observation_date],
      ["Latest actual price", formatPrice(payload.latest_actual_price ?? metadata.latest_actual_price ?? series.latest_actual_price)],
      ["Horizon", `${metadata.horizon_months || payload.horizon_months} month${Number(metadata.horizon_months || payload.horizon_months) === 1 ? "" : "s"}`],
      ["Method", metadata.recursive_method || payload.recursive_method],
    ];
    $("#forecastMeta").innerHTML = entries.map(([label, value]) => `<div class="meta-item"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value || "Unavailable")}</strong></div>`).join("");
    $("#forecastTitle").textContent = `${series.commodity || metadata.commodity || "Series"} forecast`;
  }

  function renderForecastTable(payload) {
    const body = $("#forecastTableBody");
    if (!body) return;
    body.innerHTML = (payload.forecast || []).map((row) => `<tr><td>${escapeHTML(formatDate(row.forecast_month))}</td><td class="price-cell">${escapeHTML(formatPrice(row.forecast_price))}</td><td>${escapeHTML(payload.model_label || formatModel(payload.model))}</td><td>${escapeHTML(`${row.horizon_months} month${row.horizon_months === 1 ? "" : "s"}`)}</td><td class="status-observed">Estimate</td></tr>`).join("");
  }

  async function generateForecast() {
    const dataset = $("#forecastDataset").value;
    const seriesId = $("#forecastSeries").value;
    const model = $("#forecastModel").value;
    const horizon = $("#forecastHorizon").value;
    const button = $("#generateForecast");
    setInlineError("#forecastError", "");
    if (!seriesId || !model || !["1", "3", "6"].includes(horizon)) { setInlineError("#forecastError", "Select a supported series, model, and horizon before generating a forecast."); return; }
    button.disabled = true;
    button.innerHTML = "<span>…</span> Generating…";
    try {
      const query = new URLSearchParams({ dataset, series_id: seriesId, model, horizon });
      const payload = await apiGet(`/api/forecast?${query.toString()}`);
      if (!payload.forecast || payload.forecast.length !== Number(horizon)) throw new Error("The API returned an incomplete forecast.");
      state.forecast = payload;
      renderForecastChart(payload);
      renderForecastMeta(payload);
      renderForecastTable(payload);
      $("#forecastEmpty").hidden = true;
      $("#forecastResult").hidden = false;
      showToast("Forecast generated from the saved model artifact.");
    } catch (error) {
      $("#forecastResult").hidden = true;
      $("#forecastEmpty").hidden = false;
      setInlineError("#forecastError", `${error.message}${error.status ? ` (HTTP ${error.status})` : ""}`);
    } finally {
      button.disabled = false;
      button.innerHTML = "<span>↗</span> Generate forecast";
    }
  }

  function renderPerformance(payload, dataset) {
    const records = (payload.records || []).filter((row) => row.status === "evaluated" && row.evaluation_validity !== false);
    destroyChart("rmse");
    destroyChart("mae");
    $("#performanceMethodologyText").textContent = dataset === "aggregate"
      ? "Models are compared using identical test observations within each aggregate commodity series. The original model training cutoffs were not identical; therefore, this comparison controls test-date differences but does not represent a fully controlled experiment with identical training cutoffs."
      : "These are preserved market-level chronological evaluation records for the selected series. Aggregate common-test-date metrics are not applied to market-level models, and unavailable model artifacts remain unavailable.";
    if (!records.length) {
      setInlineError("#performanceError", "No valid evaluation records are available for this series.");
      $("#performanceTableBody").innerHTML = "";
      return;
    }
    setInlineError("#performanceError", "");
    const best = records.reduce((winner, row) => Number(row.rmse) < Number(winner.rmse) ? row : winner, records[0]);
    const first = records.map((row) => row.first_test_date).filter(Boolean).sort()[0];
    const last = records.map((row) => row.last_test_date).filter(Boolean).sort().slice(-1)[0];
    $("#performanceSummary").innerHTML = `<strong>${escapeHTML(formatModel(best.model, best.model_label))}</strong> has the lowest RMSE in this series (${escapeHTML(formatNumber(best.rmse, 2))}). Test coverage: ${escapeHTML(first || "—")} — ${escapeHTML(last || "—")}.`;
    $("#performanceDates").textContent = `${first || "—"} — ${last || "—"}`;
    const labels = records.map((row) => formatModel(row.model, row.model_label));
    const barConfig = (metric, name) => ({ type: "bar", data: { labels, datasets: [{ label: name, data: records.map((row) => Number(row[metric])), backgroundColor: records.map((row) => row.model === best.model ? "#2f9b6d" : "#9fc6ae"), borderRadius: 5, borderSkipped: false, barPercentage: .62, categoryPercentage: .72 }] }, options: { ...chartOptions(), indexAxis: "y", plugins: { ...chartOptions().plugins, tooltip: { ...chartOptions().plugins.tooltip, callbacks: { label: (context) => `${name}: ${formatNumber(context.parsed.x, 2)}` } } }, scales: { ...chartOptions().scales, x: { ...chartOptions().scales.y, beginAtZero: true, ticks: { color: "#87958d", font: { family: "DM Sans", size: 10 }, callback: (value) => formatNumber(value, 0) }, grid: { color: "#edf1ee" }, border: { display: false } }, y: { ...chartOptions().scales.x, ticks: { color: "#52655b", font: { family: "DM Sans", size: 10 } }, grid: { display: false }, border: { display: false } } } } });
    if (window.Chart) {
      state.charts.rmse = new Chart($("#rmseChart"), barConfig("rmse", "RMSE"));
      state.charts.mae = new Chart($("#maeChart"), barConfig("mae", "MAE"));
    }
    $("#performanceTableBody").innerHTML = records.map((row) => `<tr class="${row.model === best.model ? "best-row" : ""}"><td>${escapeHTML(formatModel(row.model, row.model_label))}</td><td>${escapeHTML(formatNumber(row.rmse, 2))}</td><td>${escapeHTML(formatNumber(row.mae, 2))}</td><td>${escapeHTML(formatNumber(row.mape, 2))}%</td><td>${escapeHTML(formatNumber(row.r2, 3))}</td><td>${escapeHTML(formatNumber(row.bias, 2))}</td><td>${escapeHTML(formatNumber(row.n_test))}</td><td class="validity">Valid</td></tr>`).join("");
  }

  async function loadPerformance(dataset, seriesId) {
    if (!seriesId) return;
    try {
      const query = new URLSearchParams({ dataset, series_id: seriesId });
      const payload = await apiGet(`/api/performance?${query.toString()}`);
      renderPerformance(payload, dataset);
    } catch (error) {
      setInlineError("#performanceError", error.message);
    }
  }

  function showSection(sectionId) {
    $$(".dashboard-section").forEach((section) => { section.hidden = section.id !== sectionId; section.classList.toggle("active", section.id === sectionId); });
    const target = $(`[data-section-target="${sectionId}"]`);
    $$(".side-link").forEach((link) => link.classList.toggle("active", link === target));
    const title = { "overview-section": "Overview", "analysis-section": "Price analysis", "forecast-section": "Price forecasting", "performance-section": "Model performance", "about-section": "About the project" }[sectionId] || "Overview";
    $("#pageTitle").textContent = title;
    const sidebar = $("#sidebar");
    const menu = $("#mobileMenu");
    if (sidebar) sidebar.classList.remove("open");
    if (menu) menu.setAttribute("aria-expanded", "false");
    if (sectionId === "performance-section") loadPerformance(state.dataset, state.selectedSeries[state.dataset]);
  }

  function switchDataset(dataset) {
    if (!state.series[dataset]) return;
    state.dataset = dataset;
    if (state.overview) renderOverview(state.overview);
    renderSeriesTiles();
    syncSeriesControls(dataset);
    loadHistory(dataset, state.selectedSeries[dataset]);
    if (!$("#performance-section").hidden) loadPerformance(dataset, state.selectedSeries[dataset]);
  }

  function bindEvents() {
    $$("[data-section-target]").forEach((button) => button.addEventListener("click", () => showSection(button.dataset.sectionTarget)));
    $("#mobileMenu").addEventListener("click", () => {
      const sidebar = $("#sidebar");
      const open = sidebar.classList.toggle("open");
      $("#mobileMenu").setAttribute("aria-expanded", String(open));
    });
    $("#globalDataset").addEventListener("change", (event) => switchDataset(event.target.value));
    $("#analysisDataset").addEventListener("change", (event) => switchDataset(event.target.value));
    $("#forecastDataset").addEventListener("change", (event) => { switchDataset(event.target.value); updateForecastModels(event.target.value, state.selectedSeries[event.target.value]); });
    $("#analysisSeries").addEventListener("change", (event) => {
      state.selectedSeries[state.dataset] = event.target.value;
      $("#forecastSeries").value = event.target.value;
      $("#performanceSeries").value = event.target.value;
      const selected = getSeriesItem(state.dataset, event.target.value);
      setHistoryDateDefaults(selected);
      renderLatestObservation(selected, "analysis");
      renderLatestObservation(selected, "forecast");
      loadHistory(state.dataset, event.target.value);
    });
    $("#forecastSeries").addEventListener("change", (event) => {
      const dataset = $("#forecastDataset").value;
      state.selectedSeries[dataset] = event.target.value;
      updateForecastModels(dataset, event.target.value);
      renderLatestObservation(getSeriesItem(dataset, event.target.value), "forecast");
    });
    $("#performanceSeries").addEventListener("change", (event) => loadPerformance(state.dataset, event.target.value));
    $("#refreshHistory").addEventListener("click", () => loadHistory(state.dataset, $("#analysisSeries").value));
    [$("#analysisStartDate"), $("#analysisEndDate")].forEach((input) => input.addEventListener("change", () => loadHistory(state.dataset, $("#analysisSeries").value)));
    $("#generateForecast").addEventListener("click", generateForecast);
  }

  async function init() {
    bindEvents();
    try {
      const [overview, series] = await Promise.all([apiGet("/api/overview"), apiGet("/api/series")]);
      state.overview = overview;
      state.series = { aggregate: series.aggregate || [], market: series.market || [], world_bank: series.world_bank || [] };
      configureDatasetSelectors(overview.available_datasets || Object.keys(series).filter((key) => state.series[key]));
      renderOverview(overview);
      syncSeriesControls(state.dataset);
      renderSeriesTiles();
      await loadHistory(state.dataset, state.selectedSeries[state.dataset]);
      $("#footerStatus").textContent = "API connected";
    } catch (error) {
      $("#footerStatus").textContent = "API unavailable";
      showToast(error.message, true);
      $("#overviewSeriesGrid").innerHTML = '<div class="loading-line">The dashboard API could not be reached. Refresh to try again.</div>';
      $("#statGrid").innerHTML = '<div class="inline-alert error-alert" style="grid-column:1/-1">The overview could not be loaded from the Flask backend.</div>';
    }
  }

  document.addEventListener("DOMContentLoaded", init);
}());
