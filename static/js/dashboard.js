(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    dataset: "wfp",
    overview: null,
    series: { wfp: [], nbs: [] },
    selected: { wfp: "", nbs: "" },
    history: [],
    charts: {},
  };

  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }

  function formatNumber(value, digits = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat("en-NG", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(number) : "—";
  }

  function formatPrice(value) {
    return value === null || value === undefined || value === "" ? "—" : `₦${formatNumber(value, 2)}`;
  }

  function formatMonth(value) {
    if (!value) return "—";
    const match = String(value).match(/^(\d{4})-(\d{2})/);
    if (!match) return String(value);
    return new Intl.DateTimeFormat("en-NG", { month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, 1)));
  }

  async function apiGet(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error("The dashboard returned an unreadable response."); }
    if (!response.ok) {
      const error = payload && payload.error ? payload.error : {};
      const failure = new Error(error.message || "The request could not be completed.");
      failure.status = response.status;
      failure.code = error.code;
      throw failure;
    }
    return payload;
  }

  function showToast(message, isError = false) {
    const node = $("#toastMessage");
    if (!node) return;
    node.textContent = message;
    node.classList.toggle("error", isError);
    node.classList.add("visible");
    window.setTimeout(() => node.classList.remove("visible"), 4200);
  }

  function setMessage(selector, message, visible = true) {
    const node = $(selector);
    if (!node) return;
    node.textContent = message;
    node.hidden = !visible;
  }

  function getSeries(dataset = state.dataset) {
    return state.series[dataset] || [];
  }

  function getSeriesItem(dataset, id) {
    return getSeries(dataset).find((item) => item.series_id === id) || null;
  }

  function configureDatasetSelectors(dataset) {
    state.dataset = dataset === "nbs" ? "nbs" : "wfp";
    ["#globalDataset", "#analysisDataset", "#forecastDataset"].forEach((selector) => {
      const node = $(selector);
      if (node) node.value = state.dataset;
    });
  }

  function fillSeriesSelectors(dataset, preferredId) {
    const items = getSeries(dataset);
    const selectors = ["#analysisSeries", "#forecastSeries", "#performanceSeries"];
    const current = preferredId || state.selected[dataset];
    const nextId = items.some((item) => item.series_id === current) ? current : (items[0] ? items[0].series_id : "");
    state.selected[dataset] = nextId;
    const optionMarkup = items.map((item) => {
      const geo = item.market || item.geography_name || "";
      const flag = item.price_flag ? ` · ${item.price_flag}` : "";
      const unit = item.unit || "Not stated";
      return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(item.commodity)} · ${escapeHTML(unit)} · ${escapeHTML(item.price_type)} · ${escapeHTML(geo)}${escapeHTML(flag)}</option>`;
    }).join("");
    selectors.forEach((selector) => {
      const node = $(selector);
      if (!node) return;
      node.innerHTML = optionMarkup || '<option value="">No series available</option>';
      node.querySelectorAll("option").forEach((option) => {
        const item = items.find((candidate) => candidate.series_id === option.value);
        if (item) option.textContent += item.forecast_available ? " [historical forecast]" : " [forecast unavailable]";
      });
      node.value = nextId;
    });
    updateForecastAvailability(dataset, nextId);
    return getSeriesItem(dataset, nextId);
  }

  function updateForecastAvailability(dataset, seriesId) {
    const series = getSeriesItem(dataset, seriesId);
    const model = $("#forecastModel");
    const button = $("#generateForecast");
    const available = Boolean(series && series.forecast_available && series.champion_model);
    if (model) {
      model.innerHTML = available
        ? `<option value="${escapeHTML(series.champion_model)}">${escapeHTML(series.champion_model.replaceAll("_", " "))} · validation-selected</option>`
        : '<option value="">No saved forecast for this series</option>';
      model.disabled = !available;
    }
    if (button) {
      button.disabled = !available;
      button.textContent = available ? "Show saved forecast" : "Forecast unavailable";
    }
    const note = $("#modelAvailabilityNote");
    if (note) note.textContent = available
      ? `Historical-only recursive output from ${series.forecast_origin_month || series.last_observed_month}; not a current market estimate.`
      : (series ? (series.training_note || "No validated forecast is available for this series.") : "Select an available series.");
    if (!available) {
      $("#forecastResults").hidden = true;
      $("#forecastEmpty").hidden = false;
    }
  }

  function renderOverview() {
    const facts = state.overview?.datasets?.[state.dataset] || {};
    const nbs = state.dataset === "nbs";
    const representedPlaces = nbs ? facts.geography_count : facts.market_count;
    const stats = [
      ["Source rows", formatNumber(facts.historical_observations), nbs ? "NBS item/geography/month records" : "Original WFP price records"],
      ["Commodities / items", formatNumber(facts.commodity_count), "Source labels retained"],
      [nbs ? "Geography labels" : "Observed markets", formatNumber(representedPlaces), nbs ? "National and zone" : "Market IDs in records"],
      ["States represented", nbs ? "Not available" : formatNumber(facts.state_count), nbs ? "Workbook is not a full state panel" : "WFP admin1 labels"],
      ["Models trained", formatNumber(state.overview?.model_status?.available_model_count), state.overview?.model_status?.message || "Model status unavailable"],
    ];
    const grid = $("#statGrid");
    if (grid) grid.innerHTML = stats.map(([label, value, note]) => `<article class="stat-card"><span class="stat-label">${escapeHTML(label)}</span><strong class="stat-value">${escapeHTML(value)}</strong><span class="stat-sub">${escapeHTML(note)}</span></article>`).join("");
    const latest = nbs
      ? (facts.latest_observation_date || facts.date_end || "Unavailable")
      : (facts.latest_actual_observation_date || "Unavailable");
    $("#coverageLabel").textContent = nbs ? "Latest NBS report month" : "Latest pure-actual WFP observation";
    $("#latestDate").textContent = latest;
    $("#overallDateRange").textContent = facts.date_start && facts.date_end
      ? `${nbs ? "NBS report range" : "All WFP price flags"}: ${facts.date_start} — ${facts.date_end}`
      : "Date range unavailable";
    $("#frequencyValue").textContent = nbs ? "Monthly reports" : "Monthly reporting";
    $("#currencyValue").textContent = facts.currency || "NGN";
    $("#aboutCoverage").textContent = facts.date_start && facts.date_end ? `${facts.date_start} — ${facts.date_end}` : "Unavailable";
    $("#aboutFrequency").textContent = nbs ? "Monthly reports" : "Monthly reporting";
    $("#aboutCurrency").textContent = facts.currency || "NGN";
    renderSeriesTiles();
  }

  function renderSeriesTiles() {
    const grid = $("#overviewSeriesGrid");
    const items = getSeries();
    $("#seriesCount").textContent = `${formatNumber(items.length)} series`;
    if (!grid) return;
    if (!items.length) {
      grid.innerHTML = '<div class="loading-line">No source series are available.</div>';
      return;
    }
    const preview = items.slice(0, 18);
    grid.innerHTML = preview.map((item) => {
      const location = item.market || item.geography_name || "";
      const flag = item.price_flag ? ` · ${item.price_flag}` : "";
      return `<button class="series-tile" type="button" data-series-id="${escapeHTML(item.series_id)}"><span class="series-tile-heading"><strong>${escapeHTML(item.commodity)}</strong><span class="series-type">${escapeHTML(item.source)}</span></span><small>${escapeHTML(item.unit || "Unit not stated")} · ${escapeHTML(item.price_type)} · ${escapeHTML(location)}${escapeHTML(flag)}</small><small>${formatNumber(item.observed_months)} reported months · ${escapeHTML(item.first_observed_month)} — ${escapeHTML(item.last_observed_month)}</small></button>`;
    }).join("") + (items.length > preview.length ? `<div class="loading-line">Showing the first ${preview.length} series. Use Price analysis to access all ${formatNumber(items.length)} series.</div>` : "");
    grid.querySelectorAll("button.series-tile").forEach((button) => {
      const item = getSeriesItem(state.dataset, button.dataset.seriesId);
      const status = document.createElement("small");
      status.textContent = item && item.forecast_available ? "Historical forecast available" : "Forecast unavailable";
      button.append(status);
    });
    grid.querySelectorAll("[data-series-id]").forEach((button) => button.addEventListener("click", () => {
      const item = getSeriesItem(state.dataset, button.dataset.seriesId);
      if (!item) return;
      fillSeriesSelectors(state.dataset, item.series_id);
      setSection("analysis-section");
      loadHistory(state.dataset, item.series_id);
    }));
  }

  function monthRange(start, end) {
    const [startYear, startMonth] = start.split("-").map(Number);
    const [endYear, endMonth] = end.split("-").map(Number);
    const labels = [];
    let year = startYear;
    let month = startMonth;
    while (year < endYear || (year === endYear && month <= endMonth)) {
      labels.push(`${year}-${String(month).padStart(2, "0")}`);
      month += 1;
      if (month === 13) { month = 1; year += 1; }
    }
    return labels;
  }

  function clearChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      delete state.charts[key];
    }
  }

  function drawHistory(history, series, chartKey, canvasSelector, emptySelector) {
    clearChart(chartKey);
    const valid = history.filter((row) => row.month);
    const canvas = $(canvasSelector);
    const empty = $(emptySelector);
    if (!valid.length || !canvas || !window.Chart) {
      if (empty) empty.hidden = false;
      return;
    }
    const byMonth = new Map(valid.map((row) => [row.month, row.price]));
    const labels = monthRange(valid[0].month, valid[valid.length - 1].month);
    const values = labels.map((label) => byMonth.has(label) ? byMonth.get(label) : null);
    if (empty) empty.hidden = true;
    state.charts[chartKey] = new Chart(canvas, {
      type: "line",
      data: { labels, datasets: [{
        label: series.source === "NBS" ? "NBS reported average" : `WFP ${series.price_flag || "price"}`,
        data: values,
        borderColor: "#367c5b",
        backgroundColor: "rgba(54,124,91,.10)",
        borderWidth: 2,
        pointRadius: 1.8,
        fill: true,
        tension: 0.2,
        spanGaps: false,
      }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: "index" },
        plugins: { legend: { display: true }, tooltip: { callbacks: { label: (context) => ` ${formatPrice(context.parsed.y)}` } } },
        scales: {
          x: { title: { display: true, text: "Reporting month" }, ticks: { maxTicksLimit: 12 } },
          y: { title: { display: true, text: series.currency || "Price (source currency)" }, beginAtZero: false },
        },
      },
    });
  }

  function renderLatest(series) {
    if (!series) return;
    const isNbs = series.source === "NBS";
    const location = series.market || series.geography_name || "";
    const date = series.latest_observation_date || series.last_observed_month;
    $("#analysisLatestPrice").textContent = formatPrice(series.latest_price);
    $("#analysisLatestContext").textContent = `${series.commodity} · ${series.unit || "unit not stated"} · ${series.price_type}${location ? ` · ${location}` : ""}`;
    $("#analysisLatestDate").textContent = date || "—";
    $("#analysisLatestAge").textContent = Number.isFinite(Number(series.observation_age_months))
      ? `${formatNumber(series.observation_age_months)} month${Number(series.observation_age_months) === 1 ? "" : "s"} old`
      : "—";
    $("#analysisFreshness").textContent = isNbs
      ? `NBS summary · ${series.freshness || "unavailable"}`
      : (series.price_flag === "actual" ? (series.freshness || "unavailable") : `WFP ${series.price_flag || "source"} flag`);
    $("#analysisFreshnessExplanation").textContent = series.price_flag && series.price_flag !== "actual"
      ? "This WFP record is not a pure direct-actual price flag. Its date does not establish freshness of direct market observations."
      : `${series.freshness_explanation || "Observation age is not a measure of price accuracy or representativeness."} Age describes recency only.`;
    $("#forecastLatestPrice").textContent = `Latest available value: ${formatPrice(series.latest_price)}`;
    $("#forecastLatestContext").textContent = `${series.source} reporting period ${formatMonth(series.last_observed_month)}. No forecast is available.`;
    $("#forecastLatestDate").textContent = formatMonth(series.last_observed_month);
    $("#forecastLatestAge").textContent = "—";
    $("#forecastFreshness").textContent = "Not forecastable";
  }

  function renderHistoryTable(history, series) {
    const body = $("#historyTableBody");
    if (!body) return;
    body.innerHTML = history.slice().reverse().map((row) => {
      const place = series.market || series.geography_name || "—";
      const status = row.price === null ? "Price missing in source" : (row.price_flag || "Reported");
      return `<tr><td>${escapeHTML(row.month)}</td><td>${escapeHTML(series.commodity)}</td><td class="price-cell">${escapeHTML(formatPrice(row.price))}</td><td>${escapeHTML(series.unit || "Not stated")}</td><td>${escapeHTML(series.price_type)}</td><td>${escapeHTML(place)}</td><td>${escapeHTML(status)}</td></tr>`;
    }).join("");
    $("#historyCount").textContent = `${formatNumber(history.length)} source rows`;
  }

  async function loadHistory(dataset, seriesId) {
    const series = getSeriesItem(dataset, seriesId);
    if (!series) return;
    state.selected[dataset] = seriesId;
    const params = new URLSearchParams({ dataset, series_id: seriesId });
    if (dataset === $("#analysisDataset").value) {
      if ($("#analysisStartDate").value) params.set("start_date", $("#analysisStartDate").value);
      if ($("#analysisEndDate").value) params.set("end_date", $("#analysisEndDate").value);
    }
    try {
      const payload = await apiGet(`/api/history?${params.toString()}`);
      state.history = payload.history || [];
      const selectedSeries = payload.series || series;
      const location = selectedSeries.market || selectedSeries.geography_name || "";
      $("#historyTitle").textContent = `${selectedSeries.commodity} · ${selectedSeries.unit || "unit not stated"}${location ? ` · ${location}` : ""}`;
      $("#historyCoverage").textContent = `${selectedSeries.first_observed_month} — ${selectedSeries.last_observed_month}`;
      $("#overviewChartTitle").textContent = `${selectedSeries.commodity} monthly history`;
      $("#overviewChartMeta").textContent = `${formatNumber(selectedSeries.observed_months)} reported months · ${selectedSeries.price_type}`;
      drawHistory(state.history, selectedSeries, "history", "#historyChart", "#historyEmpty");
      drawHistory(state.history, selectedSeries, "overview", "#overviewChart", "#overviewChartEmpty");
      renderHistoryTable(state.history, selectedSeries);
      renderLatest(selectedSeries);
      setMessage("#forecastError", "", false);
    } catch (error) {
      showToast(error.message, true);
      setMessage("#historyEmpty", error.message, true);
    }
  }

  async function loadPerformance(dataset, seriesId) {
    if (!seriesId) return;
    try {
      const params = new URLSearchParams({ dataset, series_id: seriesId });
      const payload = await apiGet(`/api/performance?${params.toString()}`);
      const summary = payload.summary || {};
      $("#performanceSummary").textContent = payload.message || "No validated evaluation records are available.";
      $("#performanceDates").textContent = summary.test_date_count
        ? `${formatNumber(summary.test_date_count)} common test dates · ${summary.test_first_month} — ${summary.test_last_month} · ${summary.forecast_classification || "historical experiment"}`
        : "No evaluation dates";
      const labels = { naive_last_observation: "Naive last observation", random_forest: "Random Forest", xgboost: "XGBoost", gradient_boosting: "Gradient Boosting", stacking: "Hybrid stacking", lstm: "LSTM" };
      $("#performanceTableBody").innerHTML = (payload.records || []).map((row) => {
        const champion = row.champion_selected_on_validation ? " · champion by validation" : "";
        const status = row.status === "trained_and_evaluated" ? `Evaluated${champion}` : (row.unavailable_reason || "Unavailable");
        return `<tr><td>${escapeHTML(labels[row.model] || row.model)}${row.champion_selected_on_validation ? " ★" : ""}</td><td>${formatNumber(row.rmse, 2)}</td><td>${formatNumber(row.mae, 2)}</td><td>${row.mape === null ? "—" : `${formatNumber(row.mape, 2)}%`}</td><td>${formatNumber(row.r2, 3)}</td><td>${formatNumber(row.bias, 2)}</td><td>${escapeHTML(status)}</td></tr>`;
      }).join("");
      setMessage("#performanceError", payload.status === "evaluated" ? "" : (payload.message || "No test metrics are available."), payload.status !== "evaluated");
      drawPerformance(payload.predictions_by_date || []);
    } catch (error) {
      setMessage("#performanceError", error.message, true);
    }
  }

  function drawPerformance(rows) {
    clearChart("performance");
    const canvas = $("#performanceChart");
    const empty = $("#performanceChartEmpty");
    if (!rows.length || !canvas || !window.Chart) {
      if (empty) empty.hidden = false;
      return;
    }
    const models = [...new Set(rows.flatMap((row) => Object.keys(row.models || {})))];
    const actual = rows.map((row) => {
      const first = Object.values(row.models || {})[0];
      return first ? first.actual : null;
    });
    const palette = ["#a66b2c", "#2c7a56", "#4f79a7", "#8756a5", "#cf5b45", "#68727d"];
    const datasets = [{ label: "Actual", data: actual, borderColor: "#183d2e", backgroundColor: "#183d2e", borderWidth: 3, pointRadius: 2, spanGaps: false }];
    models.forEach((model, index) => {
      datasets.push({
        label: model.replaceAll("_", " "),
        data: rows.map((row) => row.models?.[model]?.predicted ?? null),
        borderColor: palette[index % palette.length], backgroundColor: palette[index % palette.length],
        borderWidth: model === "naive_last_observation" ? 2 : 1.7, pointRadius: 1.5,
        borderDash: model === "naive_last_observation" ? [5, 4] : [], spanGaps: false,
      });
    });
    if (empty) empty.hidden = true;
    state.charts.performance = new Chart(canvas, {
      type: "line", data: { labels: rows.map((row) => row.month), datasets },
      options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: "index" }, plugins: { legend: { display: true } }, scales: { x: { title: { display: true, text: "Common chronological test month" }, ticks: { maxTicksLimit: 12 } }, y: { title: { display: true, text: "Price in source NGN/unit" } } } },
    });
  }

  function renderForecast(payload) {
    const rows = payload.forecast || [];
    const historical = payload.historical || [];
    const metadata = payload.metadata || {};
    if (!rows.length) return;
    $("#forecastEmpty").hidden = true;
    $("#forecastResults").hidden = false;
    $("#forecastChartTitle").textContent = `${payload.series?.commodity || metadata.commodity} · ${payload.horizon}-month recursive forecast`;
    $("#forecastMetaLabel").textContent = `Historical only · origin ${metadata.forecast_origin_month}`;
    $("#forecastMethodText").textContent = `${metadata.method}. Generated ${metadata.forecast_generation_date} from origin ${metadata.forecast_origin_month}; latest actual in the source is ${metadata.latest_actual_month_in_source}. It is not a current market price and has no confidence interval.`;
    $("#forecastTableBody").innerHTML = rows.map((row) => `<tr><td>${escapeHTML(row.month)}</td><td class="price-cell">${escapeHTML(formatPrice(row.price))}</td><td>${formatNumber(row.step)}</td><td>${escapeHTML(payload.model.replaceAll("_", " "))}</td><td>Historical only</td></tr>`).join("");
    clearChart("forecast");
    const actualByMonth = new Map(historical.map((row) => [row.month, row.price]));
    const forecastByMonth = new Map(rows.map((row) => [row.month, row.price]));
    const allMonths = [...actualByMonth.keys(), ...forecastByMonth.keys()].sort();
    const labels = allMonths.length ? monthRange(allMonths[0], allMonths[allMonths.length - 1]) : [];
    const canvas = $("#forecastChart");
    if (canvas && window.Chart) {
      state.charts.forecast = new Chart(canvas, {
        type: "line",
        data: { labels, datasets: [
          { label: "Historical WFP actual", data: labels.map((month) => actualByMonth.get(month) ?? null), borderColor: "#2c7a56", backgroundColor: "rgba(44,122,86,.08)", borderWidth: 2, pointRadius: 2, spanGaps: false },
          { label: "Recursive forecast", data: labels.map((month) => forecastByMonth.get(month) ?? null), borderColor: "#cf5b45", backgroundColor: "rgba(207,91,69,.08)", borderWidth: 2, borderDash: [6, 4], pointRadius: 3, spanGaps: false },
        ] },
        options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: "index" }, plugins: { legend: { display: true } }, scales: { x: { title: { display: true, text: "Month" } }, y: { title: { display: true, text: `${metadata.currency || "NGN"} per ${metadata.unit || "source unit"}` } } } },
      });
    }
  }

  async function requestForecast() {
    const dataset = $("#forecastDataset").value;
    const seriesId = $("#forecastSeries").value;
    const model = $("#forecastModel").value;
    const horizon = $("#forecastHorizon").value;
    const button = $("#generateForecast");
    button.disabled = true;
    button.textContent = "Loading saved forecast…";
    setMessage("#forecastError", "", false);
    try {
      const payload = await apiGet(`/api/forecast?${new URLSearchParams({ dataset, series_id: seriesId, model, horizon }).toString()}`);
      renderForecast(payload);
    } catch (error) {
      setMessage("#forecastError", error.message, true);
    } finally {
      button.disabled = !getSeriesItem(dataset, seriesId)?.forecast_available;
      button.textContent = button.disabled ? "Forecast unavailable" : "Show saved forecast";
    }
  }

  function setSection(sectionId) {
    document.querySelectorAll(".dashboard-section").forEach((section) => {
      section.hidden = section.id !== sectionId;
      section.classList.toggle("active", section.id === sectionId);
    });
    document.querySelectorAll(".side-link").forEach((link) => link.classList.toggle("active", link.dataset.sectionTarget === sectionId));
    const names = { "overview-section": "Overview", "analysis-section": "Price analysis", "forecast-section": "Forecasting", "performance-section": "Model performance", "about-section": "Data and methodology" };
    $("#pageTitle").textContent = names[sectionId] || "Overview";
    $("#sidebar").classList.remove("open");
    $("#mobileMenu").setAttribute("aria-expanded", "false");
    if (sectionId === "performance-section") loadPerformance(state.dataset, state.selected[state.dataset]);
  }

  function switchDataset(dataset) {
    if (!state.series[dataset]) return;
    state.dataset = dataset;
    configureDatasetSelectors(dataset);
    const series = fillSeriesSelectors(dataset);
    renderOverview();
    if (series) loadHistory(dataset, series.series_id);
    if (!$("#performance-section").hidden) loadPerformance(dataset, series?.series_id);
  }

  function bindEvents() {
    document.querySelectorAll("[data-section-target]").forEach((button) => button.addEventListener("click", () => setSection(button.dataset.sectionTarget)));
    $("#mobileMenu").addEventListener("click", () => {
      const open = $("#sidebar").classList.toggle("open");
      $("#mobileMenu").setAttribute("aria-expanded", String(open));
    });
    ["#globalDataset", "#analysisDataset", "#forecastDataset"].forEach((selector) => $(selector).addEventListener("change", (event) => switchDataset(event.target.value)));
    $("#analysisSeries").addEventListener("change", (event) => {
      const item = getSeriesItem(state.dataset, event.target.value);
      if (!item) return;
      fillSeriesSelectors(state.dataset, item.series_id);
      loadHistory(state.dataset, item.series_id);
    });
    $("#forecastSeries").addEventListener("change", (event) => {
      state.selected[state.dataset] = event.target.value;
      updateForecastAvailability(state.dataset, event.target.value);
      renderLatest(getSeriesItem(state.dataset, event.target.value));
    });
    $("#performanceSeries").addEventListener("change", (event) => loadPerformance(state.dataset, event.target.value));
    $("#refreshHistory").addEventListener("click", () => loadHistory(state.dataset, $("#analysisSeries").value));
    ["#analysisStartDate", "#analysisEndDate"].forEach((selector) => $(selector).addEventListener("change", () => loadHistory(state.dataset, $("#analysisSeries").value)));
    $("#generateForecast").addEventListener("click", requestForecast);
  }

  async function init() {
    bindEvents();
    try {
      const [overview, series] = await Promise.all([apiGet("/api/overview"), apiGet("/api/series")]);
      state.overview = overview;
      state.series = { wfp: series.wfp || [], nbs: series.nbs || [] };
      configureDatasetSelectors(overview.default_dataset || "wfp");
      const current = fillSeriesSelectors(state.dataset);
      renderOverview();
      if (current) await loadHistory(state.dataset, current.series_id);
      $("#footerStatus").textContent = "WFP and NBS data ready";
    } catch (error) {
      $("#footerStatus").textContent = "Data unavailable";
      $("#overviewSeriesGrid").innerHTML = '<div class="loading-line">The source data could not be loaded. Check the prepared WFP/NBS tables.</div>';
      $("#statGrid").innerHTML = `<div class="inline-alert error-alert" style="grid-column:1/-1">${escapeHTML(error.message)}</div>`;
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
