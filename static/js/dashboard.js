(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    dataset: "wfp",
    analysisDataset: "wfp",
    forecastDataset: "wfp",
    overview: null,
    series: { wfp: [], nbs: [] },
    analysisSelected: { wfp: "", nbs: "" },
    performanceSelected: { wfp: "", nbs: "" },
    overviewSelected: { wfp: "", nbs: "" },
    forecastSelected: { wfp: "", nbs: "" },
    forecastCommodity: { wfp: "", nbs: "" },
    forecastMarket: { wfp: "", nbs: "" },
    history: [],
    historyRequestId: { analysis: 0, overview: 0 },
    performanceRequestId: 0,
    charts: {},
    forecastRequestId: 0,
  };

  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }

  function formatNumber(value, digits = 0) {
    if (value === null || value === undefined || value === "") return "n/a";
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

  function isPureActualForecastSeries(series, dataset) {
    return Boolean(
      dataset === "wfp"
      && series
      && series.source_key === "wfp"
      && series.price_flag === "actual"
      && series.forecast_request_available
    );
  }

  function configureDatasetSelectors(dataset) {
    const selected = dataset === "nbs" ? "nbs" : "wfp";
    state.dataset = selected;
    state.analysisDataset = selected;
    state.forecastDataset = selected;
    [["#globalDataset", state.dataset], ["#analysisDataset", state.analysisDataset], ["#forecastDataset", state.forecastDataset]].forEach(([selector, value]) => {
      const node = $(selector);
      if (node) node.value = value;
    });
  }

  function fillSeriesSelectors(dataset, preferredId) {
    const items = getSeries(dataset);
    const analysisNode = $("#analysisSeries");
    let analysisId = preferredId || state.analysisSelected[dataset];
    if (!items.some((item) => item.series_id === analysisId)) analysisId = items[0]?.series_id || "";
    state.analysisSelected[dataset] = analysisId;
    const optionMarkup = (item) => {
      const geo = item.market || item.geography_name || "";
      const flag = item.price_flag ? ` · ${item.price_flag}` : "";
      return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(item.commodity)} · ${escapeHTML(item.unit || "Not stated")} · ${escapeHTML(item.price_type)} · ${escapeHTML(geo)}${escapeHTML(flag)}</option>`;
    };
    if (analysisNode) {
      analysisNode.innerHTML = items.map(optionMarkup).join("") || '<option value="">No series available</option>';
      analysisNode.value = analysisId;
    }
    return getSeriesItem(dataset, analysisId);
  }

  function fillPerformanceSelector(dataset) {
    const items = getSeries(dataset);
    const performanceNode = $("#performanceSeries");
    let performanceId = state.performanceSelected[dataset];
    if (!items.some((item) => item.series_id === performanceId)) performanceId = items[0]?.series_id || "";
    state.performanceSelected[dataset] = performanceId;
    if (performanceNode) {
      performanceNode.innerHTML = items.map((item) => {
        const geo = item.market || item.geography_name || "";
        const flag = item.price_flag ? ` · ${item.price_flag}` : "";
        return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(item.commodity)} · ${escapeHTML(item.unit || "Not stated")} · ${escapeHTML(item.price_type)} · ${escapeHTML(geo)}${escapeHTML(flag)}</option>`;
      }).join("") || '<option value="">No series available</option>';
      performanceNode.querySelectorAll("option").forEach((option) => {
        const item = items.find((candidate) => candidate.series_id === option.value);
        if (item) option.textContent += item.forecast_available ? " [historical evaluation]" : " [not evaluated]";
      });
      performanceNode.value = performanceId;
    }
    return getSeriesItem(dataset, performanceId);
  }

  function forecastMarketKey(series) {
    return JSON.stringify([series.market_id || "", series.market || "", series.admin1 || ""]);
  }

  function forecastMarketLabel(series) {
    const location = [series.market, series.admin1, series.market_id ? `ID ${series.market_id}` : ""].filter(Boolean);
    return location.join(" · ") || "Market not stated";
  }

  function renderForecastLatest(series) {
    const age = series && series.observation_age_months !== null && series.observation_age_months !== undefined && Number.isFinite(Number(series.observation_age_months))
      ? Number(series.observation_age_months)
      : null;
    const freshnessLabel = series ? ({
      fresh: "Fresh · within 3 months",
      recent: "Recent · 4–6 months",
      aging: "Aging · 7–12 months",
      stale: "Stale · over 12 months",
      unavailable: "Unavailable",
    }[series.freshness] || (age === null ? "Unavailable" : `${age} month(s) old`)) : "Select a series";
    const values = series ? {
      forecastLatestPrice: `Latest actual: ${formatPrice(series.latest_price)}`,
      forecastLatestContext: `${series.commodity} · ${series.market || "market not stated"} · ${series.unit || "unit not stated"} · ${series.price_type}. This is a source observation, not a forecast.`,
      forecastLatestDate: series.latest_observation_date ? String(series.latest_observation_date).slice(0, 10) : formatMonth(series.last_observed_month),
      forecastLatestAge: age === null ? "n/a" : `${formatNumber(age)} month(s) old`,
      forecastFreshness: freshnessLabel,
    } : {
      forecastLatestPrice: "Choose an exact series",
      forecastLatestContext: "Aggregate and mixed-price-flag records are available in historical analysis only; unit and price type must also match exactly.",
      forecastLatestDate: "n/a",
      forecastLatestAge: "n/a",
      forecastFreshness: "Select a series",
    };
    Object.entries(values).forEach(([id, value]) => {
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    });
    const freshnessElement = document.getElementById("forecastFreshness");
    if (freshnessElement?.classList) {
      ["fresh", "recent", "aging", "stale", "unavailable"].forEach((name) => freshnessElement.classList.remove(`freshness-${name}`));
      const status = series && ["fresh", "recent", "aging", "stale"].includes(series.freshness) ? series.freshness : "unavailable";
      freshnessElement.classList.add(`freshness-${status}`);
    }
  }

  function showForecastPrompt(message) {
    state.forecastRequestId += 1;
    clearChart("forecast");
    const results = $("#forecastResults");
    const empty = $("#forecastEmpty");
    const heading = empty?.querySelector("h2");
    const body = empty?.querySelector("p");
    if (results) results.hidden = true;
    if (empty) empty.hidden = false;
    if (heading) heading.textContent = "Forecast not generated yet";
    if (body) body.textContent = message;
    setMessage("#forecastError", "", false);
  }

  function updateForecastAvailability(dataset, seriesId) {
    const series = getSeriesItem(dataset, seriesId);
    const button = $("#generateForecast");
    const note = $("#modelAvailabilityNote");
    const needsExactChoice = dataset === "wfp" && seriesId === "" && !$("#forecastSeries").disabled && $("#forecastSeries").options.length > 1;
    const canCheck = isPureActualForecastSeries(series, dataset);
    if (button) {
      button.disabled = !canCheck;
      button.textContent = canCheck ? "Generate forecast" : "Forecast unavailable";
    }
    if (note) note.textContent = canCheck
      ? `${formatNumber(series.observed_months)} reported actual months (${series.first_observed_month} – ${series.last_observed_month}); full calendar-history eligibility is checked when requested.`
      : (dataset === "nbs"
        ? "NBS national/zone summaries are not supported by the WFP market forecasting engine."
        : (needsExactChoice
          ? "This market has multiple exact price series. Select the required unit and retail/wholesale type before forecasting."
          : "No pure priceflag=actual WFP market series is available for this selection."));
    renderForecastLatest(canCheck ? series : null);
    showForecastPrompt(canCheck
      ? "Review the exact market, unit, price type and horizon, then generate the recursive forecast. The response will state any history eligibility limitation."
      : (dataset === "nbs"
        ? "NBS summaries are available in historical analysis but cannot be sent to the WFP forecasting engine."
        : (needsExactChoice ? "Choose the exact unit and retail/wholesale series before generating a forecast." : "Select a commodity and market with pure priceflag=actual observations.")));
  }

  function fillForecastSelectors(dataset) {
    const commodityNode = $("#forecastCommodity");
    const marketNode = $("#forecastMarket");
    const seriesNode = $("#forecastSeries");
    const candidates = getSeries(dataset).filter((item) => isPureActualForecastSeries(item, dataset));
    const emptyOption = (node, label) => {
      if (!node) return;
      node.innerHTML = `<option value="">${escapeHTML(label)}</option>`;
      node.disabled = true;
      node.value = "";
    };
    if (!candidates.length) {
      emptyOption(commodityNode, dataset === "nbs" ? "NBS forecasting unavailable" : "No actual commodities available");
      emptyOption(marketNode, dataset === "nbs" ? "No WFP markets in NBS data" : "No markets available");
      emptyOption(seriesNode, "No pure-actual series available");
      state.forecastSelected[dataset] = "";
      updateForecastAvailability(dataset, "");
      return;
    }

    const commodities = [...new Set(candidates.map((item) => item.commodity))].sort((a, b) => a.localeCompare(b));
    let commodity = state.forecastCommodity[dataset];
    const prior = getSeriesItem(dataset, state.forecastSelected[dataset]);
    if (prior && commodities.includes(prior.commodity)) commodity = prior.commodity;
    if (!commodities.includes(commodity)) commodity = commodities[0];
    state.forecastCommodity[dataset] = commodity;
    commodityNode.innerHTML = commodities.map((name) => `<option value="${escapeHTML(name)}">${escapeHTML(name)}</option>`).join("");
    commodityNode.disabled = false;
    commodityNode.value = commodity;

    const commoditySeries = candidates.filter((item) => item.commodity === commodity);
    const markets = new Map();
    commoditySeries.forEach((item) => {
      const key = forecastMarketKey(item);
      if (!markets.has(key)) markets.set(key, item);
    });
    const marketChoices = [...markets.entries()].sort((left, right) => forecastMarketLabel(left[1]).localeCompare(forecastMarketLabel(right[1])));
    let market = state.forecastMarket[dataset];
    if (prior && prior.commodity === commodity) market = forecastMarketKey(prior);
    if (!markets.has(market)) market = marketChoices[0]?.[0] || "";
    state.forecastMarket[dataset] = market;
    marketNode.innerHTML = marketChoices.map(([key, item]) => `<option value="${escapeHTML(key)}">${escapeHTML(forecastMarketLabel(item))}</option>`).join("") || '<option value="">No markets available</option>';
    marketNode.disabled = !marketChoices.length;
    marketNode.value = market;

    const variants = commoditySeries.filter((item) => forecastMarketKey(item) === market);
    let seriesId = state.forecastSelected[dataset];
    if (!variants.some((item) => item.series_id === seriesId)) seriesId = variants.length === 1 ? variants[0].series_id : "";
    state.forecastSelected[dataset] = seriesId;
    const prompt = variants.length > 1 ? '<option value="">Choose exact unit and price type</option>' : "";
    seriesNode.innerHTML = prompt + variants.map((item) => {
      const label = `${item.unit || "Unit not stated"} · ${item.price_type} · actual · ${formatNumber(item.observed_months)} months`;
      return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(label)}</option>`;
    }).join("") || '<option value="">No exact actual series available</option>';
    seriesNode.disabled = !variants.length;
    seriesNode.value = seriesId;
    updateForecastAvailability(dataset, seriesId);
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
      state.analysisDataset = state.dataset;
      $("#analysisDataset").value = state.analysisDataset;
      fillSeriesSelectors(state.dataset, item.series_id);
      setSection("analysis-section");
      loadHistory(state.analysisDataset, item.series_id, "analysis");
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
    $("#analysisLatestAge").textContent = series.observation_age_months !== null && series.observation_age_months !== undefined && Number.isFinite(Number(series.observation_age_months))
      ? `${formatNumber(series.observation_age_months)} month${Number(series.observation_age_months) === 1 ? "" : "s"} old`
      : "—";
    $("#analysisFreshness").textContent = isNbs
      ? `NBS summary · ${series.freshness || "unavailable"}`
      : (series.price_flag === "actual" ? (series.freshness || "unavailable") : `WFP ${series.price_flag || "source"} flag`);
    $("#analysisFreshnessExplanation").textContent = series.price_flag && series.price_flag !== "actual"
      ? "This WFP record is not a pure direct-actual price flag. Its date does not establish freshness of direct market observations."
      : `${series.freshness_explanation || "Observation age is not a measure of price accuracy or representativeness."} Age describes recency only.`;
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

  async function loadHistory(dataset, seriesId, view = "analysis") {
    const series = getSeriesItem(dataset, seriesId);
    if (!series) return;
    const requestId = ++state.historyRequestId[view];
    if (view === "analysis") state.analysisSelected[dataset] = seriesId;
    else state.overviewSelected[dataset] = seriesId;
    const params = new URLSearchParams({ dataset, series_id: seriesId });
    if (view === "analysis" && dataset === state.analysisDataset) {
      if ($("#analysisStartDate").value) params.set("start_date", $("#analysisStartDate").value);
      if ($("#analysisEndDate").value) params.set("end_date", $("#analysisEndDate").value);
    }
    try {
      const payload = await apiGet(`/api/history?${params.toString()}`);
      if (requestId !== state.historyRequestId[view]) return;
      if (view === "analysis" && (dataset !== state.analysisDataset || seriesId !== state.analysisSelected[dataset])) return;
      if (view === "overview" && (dataset !== state.dataset || seriesId !== state.overviewSelected[dataset])) return;
      state.history = payload.history || [];
      const selectedSeries = payload.series || series;
      const location = selectedSeries.market || selectedSeries.geography_name || "";
      if (view === "analysis") {
        $("#historyTitle").textContent = `${selectedSeries.commodity} · ${selectedSeries.unit || "unit not stated"}${location ? ` · ${location}` : ""}`;
        $("#historyCoverage").textContent = `${selectedSeries.first_observed_month} — ${selectedSeries.last_observed_month}`;
        drawHistory(state.history, selectedSeries, "history", "#historyChart", "#historyEmpty");
        renderHistoryTable(state.history, selectedSeries);
        renderLatest(selectedSeries);
      } else {
        $("#overviewChartTitle").textContent = `${selectedSeries.commodity} monthly history`;
        $("#overviewChartMeta").textContent = `${formatNumber(selectedSeries.observed_months)} reported months · ${selectedSeries.price_type}`;
        drawHistory(state.history, selectedSeries, "overview", "#overviewChart", "#overviewChartEmpty");
      }
    } catch (error) {
      if (requestId !== state.historyRequestId[view]) return;
      if (view === "analysis") {
        showToast(error.message, true);
        setMessage("#historyEmpty", error.message, true);
      } else {
        setMessage("#overviewChartEmpty", error.message, true);
      }
    }
  }

  async function loadPerformance(dataset, seriesId) {
    if (!seriesId) return;
    const requestId = ++state.performanceRequestId;
    $("#performanceSummary").textContent = "Loading model evaluation…";
    try {
      const params = new URLSearchParams({ dataset, series_id: seriesId });
      const payload = await apiGet(`/api/performance?${params.toString()}`);
      if (requestId !== state.performanceRequestId || dataset !== state.dataset || seriesId !== state.performanceSelected[dataset]) return;
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
      if (requestId !== state.performanceRequestId || dataset !== state.dataset || seriesId !== state.performanceSelected[dataset]) return;
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
    const series = payload.series || {};
    const modelLabel = String(payload.model || metadata.model || "Unavailable").replaceAll("_", " ");
    const metric = metadata.validation_metrics || {};
    setText("forecastSubject", [series.commodity || metadata.commodity, series.market || metadata.market].filter(Boolean).join(" · ") || "Selected WFP series");
    setText("forecastSelectedHorizon", `${payload.horizon || rows.length} month(s)`);
    setText("forecastSelectedModel", modelLabel);
    setText("forecastOriginGenerated", `Origin ${metadata.forecast_origin_month || "n/a"} · Generated ${metadata.forecast_generation_date || "n/a"}`);
    setText("forecastRmse", formatNumber(metric.rmse, 2));
    setText("forecastMae", formatNumber(metric.mae, 2));
    setText("forecastMape", metric.mape === null || metric.mape === undefined ? "n/a" : `${formatNumber(metric.mape, 2)}%`);
    setText("forecastR2", formatNumber(metric.r2, 3));
    setText("forecastValidationCount", formatNumber(metric.n));
    $("#forecastChartTitle").textContent = "Historical actuals and forecast";
    renderForecastTemporalStatus(payload);
    setText("forecastMethodDetails", metadata.transformation ? `Forecast target: ${metadata.transformation}.` : "Forecast target details are unavailable.");
    $("#forecastTableBody").innerHTML = rows.map((row) => {
      const future = row.temporal_status === "future_as_of_generation";
      const status = future ? "Future at generation" : "Past at generation";
      return `<tr><td>${escapeHTML(row.month)}</td><td class="price-cell">${escapeHTML(formatPrice(row.price))}</td><td>${formatNumber(row.step)}</td><td>${status}</td></tr>`;
    }).join("");
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

  function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  }

  function renderForecastTemporalStatus(payload) {
    const metadata = payload.metadata || {};
    const series = payload.series || {};
    const classification = metadata.forecast_classification || "unverified";
    const rawAge = series.observation_age_months;
    const age = rawAge !== null && rawAge !== undefined && Number.isFinite(Number(rawAge)) ? Number(rawAge) : null;
    const freshness = String(series.freshness || "unavailable").toLowerCase();
    const isFresh = age !== null ? age <= 3 : freshness === "fresh";
    let style = "stale";
    let title = "Forecast status unverified";
    let message = "The forecast dates or source recency could not be confirmed. Do not treat this as a current market forecast.";

    if (["historical_projection_from_latest_actual", "historical_only"].includes(classification)) {
      style = "historical";
      title = "Historical projection — not a current forecast";
      message = "All requested forecast dates are already in the past relative to the generation date. They are historical projections only.";
    } else if (classification === "mixed_horizon_relative_to_generation") {
      style = "mixed";
      title = "Mixed-period projection";
      message = "Some requested dates are already in the past. Those rows are historical projections; only future rows based on recent actual data can be considered current forecasts.";
      if (!isFresh) message += ` Source data is ${age === null ? freshness : `${age} month(s) old`} and should be treated as historical context.`;
    } else if (classification === "future_as_of_generation" && isFresh) {
      style = "current";
      title = "Future forecast · recent actual data";
      message = "All requested dates are future as of the generation date, based on source data within the last three months. Estimates are not guaranteed prices.";
    } else if (classification === "future_as_of_generation") {
      style = "stale";
      title = `Future projection · ${freshness === "unavailable" ? "source age unavailable" : `${freshness} source data`}`;
      message = `All requested dates are future as of generation, but the latest actual source observation is ${age === null ? "of unknown age" : `${age} month(s) old`}. This is not a current market forecast; interpret it as historical context.`;
    }

    const box = $("#forecastTemporalStatus");
    if (box) {
      ["current", "historical", "mixed", "stale"].forEach((name) => box.classList.remove(`forecast-status-${name}`));
      box.classList.add(`forecast-status-${style}`);
    }
    setText("forecastStatusTitle", title);
    setText("forecastStatusMessage", message);
  }

  function forecastUnavailableTitle(error) {
    if (error?.code === "insufficient_or_gapped_history") return "Forecast unavailable — insufficient continuous history";
    if (error?.code === "not_direct_actual") return "Forecast unavailable — not a pure actual series";
    if (error?.code === "model_unavailable") return "Forecast unavailable — model not supported for this series";
    return "Forecast unavailable for this selection";
  }

  function forecastSelectionKey() {
    const dataset = state.forecastDataset;
    return JSON.stringify([dataset, state.forecastCommodity[dataset], state.forecastMarket[dataset], state.forecastSelected[dataset], $("#forecastHorizon").value]);
  }

  async function requestForecast() {
    const dataset = state.forecastDataset;
    const seriesId = state.forecastSelected[dataset];
    const button = $("#generateForecast");
    const requestId = ++state.forecastRequestId;
    const selectedSeries = getSeriesItem(dataset, seriesId);
    if (!isPureActualForecastSeries(selectedSeries, dataset)) {
      const reason = dataset !== "wfp"
        ? "NBS national and zone series are not supported by the WFP market forecasting engine."
        : (selectedSeries && selectedSeries.price_flag !== "actual"
          ? "This selection has no pure priceflag=actual market series. Aggregate and mixed-price-flag records are excluded."
          : "Choose a WFP series with pure priceflag=actual observations.");
      button.disabled = true;
      button.textContent = "Forecast unavailable";
      $("#forecastResults").hidden = true;
      $("#forecastEmpty").hidden = false;
      $("#forecastEmpty h2").textContent = "Forecast unavailable for this selection";
      $("#forecastEmpty p").textContent = reason;
      setMessage("#forecastError", reason, true);
      return;
    }
    const horizon = $("#forecastHorizon").value;
    if (!["1", "3", "6"].includes(horizon)) {
      const reason = "Choose a forecast horizon of 1, 3, or 6 months.";
      setMessage("#forecastError", reason, true);
      showForecastPrompt(reason);
      return;
    }
    const selectionKey = forecastSelectionKey();
    button.disabled = true;
    button.textContent = "Generating forecast…";
    setMessage("#forecastError", "", false);
    $("#forecastResults").hidden = true;
    $("#forecastEmpty").hidden = false;
    $("#forecastEmpty h2").textContent = "Checking this exact series…";
    $("#forecastEmpty p").textContent = "Checking calendar history, selecting a model using chronological validation, and preparing a recursive forecast.";
    try {
      const payload = await apiGet(`/api/forecast?${new URLSearchParams({ dataset, series_id: seriesId, model: "auto", mode: "on_demand", horizon }).toString()}`);
      if (requestId !== state.forecastRequestId || selectionKey !== forecastSelectionKey()) return;
      renderForecast(payload);
    } catch (error) {
      if (requestId === state.forecastRequestId && selectionKey === forecastSelectionKey()) {
        $("#forecastResults").hidden = true;
        $("#forecastEmpty").hidden = false;
        $("#forecastEmpty h2").textContent = forecastUnavailableTitle(error);
        $("#forecastEmpty p").textContent = error.message;
        setMessage("#forecastError", error.message, true);
      }
    } finally {
      if (requestId === state.forecastRequestId) {
        button.disabled = !isPureActualForecastSeries(getSeriesItem(dataset, seriesId), dataset);
        button.textContent = button.disabled ? "Forecast unavailable" : "Generate forecast";
      }
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
    if (sectionId === "performance-section") loadPerformance(state.dataset, state.performanceSelected[state.dataset]);
  }

  function switchOverviewDataset(dataset) {
    if (!state.series[dataset]) return;
    state.dataset = dataset;
    $("#globalDataset").value = dataset;
    const performance = fillPerformanceSelector(dataset);
    renderOverview();
    let overviewId = state.overviewSelected[dataset];
    if (!getSeriesItem(dataset, overviewId)) overviewId = getSeries(dataset)[0]?.series_id || "";
    if (overviewId) loadHistory(dataset, overviewId, "overview");
    if (!$("#performance-section").hidden) loadPerformance(dataset, performance?.series_id);
  }

  function switchAnalysisDataset(dataset) {
    if (!state.series[dataset]) return;
    state.analysisDataset = dataset;
    $("#analysisDataset").value = dataset;
    const series = fillSeriesSelectors(dataset);
    if (series) loadHistory(dataset, series.series_id, "analysis");
  }

  function switchForecastDataset(dataset) {
    if (!state.series[dataset]) return;
    state.forecastDataset = dataset;
    $("#forecastDataset").value = dataset;
    fillForecastSelectors(dataset);
  }

  function bindEvents() {
    document.querySelectorAll("[data-section-target]").forEach((button) => button.addEventListener("click", () => setSection(button.dataset.sectionTarget)));
    $("#mobileMenu").addEventListener("click", () => {
      const open = $("#sidebar").classList.toggle("open");
      $("#mobileMenu").setAttribute("aria-expanded", String(open));
    });
    $("#globalDataset").addEventListener("change", (event) => switchOverviewDataset(event.target.value));
    $("#analysisDataset").addEventListener("change", (event) => switchAnalysisDataset(event.target.value));
    $("#forecastDataset").addEventListener("change", (event) => switchForecastDataset(event.target.value));
    $("#analysisSeries").addEventListener("change", (event) => {
      const dataset = state.analysisDataset;
      const item = getSeriesItem(dataset, event.target.value);
      if (!item) return;
      fillSeriesSelectors(dataset, item.series_id);
      loadHistory(dataset, item.series_id, "analysis");
    });
    $("#forecastCommodity").addEventListener("change", (event) => {
      const dataset = state.forecastDataset;
      state.forecastCommodity[dataset] = event.target.value;
      state.forecastSelected[dataset] = "";
      fillForecastSelectors(dataset);
    });
    $("#forecastMarket").addEventListener("change", (event) => {
      const dataset = state.forecastDataset;
      state.forecastMarket[dataset] = event.target.value;
      state.forecastSelected[dataset] = "";
      fillForecastSelectors(dataset);
    });
    $("#forecastSeries").addEventListener("change", (event) => {
      const dataset = state.forecastDataset;
      state.forecastSelected[dataset] = event.target.value;
      updateForecastAvailability(dataset, event.target.value);
    });
    $("#performanceSeries").addEventListener("change", (event) => {
      state.performanceSelected[state.dataset] = event.target.value;
      loadPerformance(state.dataset, event.target.value);
    });
    $("#refreshHistory").addEventListener("click", () => loadHistory(state.analysisDataset, $("#analysisSeries").value, "analysis"));
    ["#analysisStartDate", "#analysisEndDate"].forEach((selector) => $(selector).addEventListener("change", () => loadHistory(state.analysisDataset, $("#analysisSeries").value, "analysis")));
    $("#generateForecast").addEventListener("click", requestForecast);
    $("#forecastHorizon").addEventListener("change", () => {
      showForecastPrompt(`Forecast horizon set to ${$("#forecastHorizon").value} month(s). Click Generate forecast to request new results.`);
    });
  }

  async function init() {
    bindEvents();
    try {
      const [overview, series] = await Promise.all([apiGet("/api/overview"), apiGet("/api/series")]);
      state.overview = overview;
      state.series = { wfp: series.wfp || [], nbs: series.nbs || [] };
      configureDatasetSelectors(overview.default_dataset || "wfp");
      const current = fillSeriesSelectors(state.analysisDataset);
      fillPerformanceSelector(state.dataset);
      fillForecastSelectors(state.forecastDataset);
      renderOverview();
      if (current) await loadHistory(state.analysisDataset, current.series_id, "analysis");
      const overviewItem = getSeries(state.dataset)[0];
      if (overviewItem) await loadHistory(state.dataset, overviewItem.series_id, "overview");
      $("#footerStatus").textContent = "WFP and NBS data ready";
    } catch (error) {
      $("#footerStatus").textContent = "Data unavailable";
      $("#overviewSeriesGrid").innerHTML = '<div class="loading-line">The source data could not be loaded. Check the prepared WFP/NBS tables.</div>';
      $("#statGrid").innerHTML = `<div class="inline-alert error-alert" style="grid-column:1/-1">${escapeHTML(error.message)}</div>`;
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
