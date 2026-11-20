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

  function formatMonthYear(value) {
    const match = String(value || "").match(/^(\d{4})-(\d{2})/);
    if (!match) return value || "Not reported";
    return new Intl.DateTimeFormat("en-NG", { month: "long", year: "numeric", timeZone: "UTC" })
      .format(new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, 1)));
  }

  function formatStartingObservation(month, exactDate) {
    const monthYear = formatMonthYear(month || exactDate);
    const exact = String(exactDate || "").slice(0, 10);
    return /^\d{4}-\d{2}-\d{2}$/.test(exact) ? `${monthYear} (${exact})` : monthYear;
  }

  function formatLongDate(value) {
    const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return "Date unavailable";
    return new Intl.DateTimeFormat("en-NG", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })
      .format(new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]))));
  }

  function formatConfidence(value) {
    const classification = String(value || "").trim().toLowerCase();
    if (classification === "high") return "High";
    if (["medium", "limited", "moderate"].includes(classification)) return "Medium";
    if (["low", "very low"].includes(classification)) return "Low";
    return "Not rated";
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

  function isForecastRequestableSeries(series, dataset) {
    return Boolean(series && series.source_key === dataset && series.forecast_request_available && Number(series.observed_months) > 0);
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
      const regime = item.methodology_regime ? ` · ${item.methodology_regime.replaceAll("_", " ")}` : "";
      const mapping = dataset === "nbs" && item.research_commodity_mapping
        ? ` · research mapping: ${item.research_commodity_mapping}${item.mapping_status ? ` (${item.mapping_status.replaceAll("_", " ")})` : ""}`
        : "";
      return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(item.commodity)} · ${escapeHTML(item.unit || "Not stated")} · ${escapeHTML(item.price_type)} · ${escapeHTML(geo)}${escapeHTML(flag)}${escapeHTML(regime)}${escapeHTML(mapping)}</option>`;
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
        const regime = item.methodology_regime ? ` · ${item.methodology_regime.replaceAll("_", " ")}` : "";
        const mapping = dataset === "nbs" && item.research_commodity_mapping
          ? ` · research mapping: ${item.research_commodity_mapping}${item.mapping_status ? ` (${item.mapping_status.replaceAll("_", " ")})` : ""}`
          : "";
        return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(item.commodity)} · ${escapeHTML(item.unit || "Not stated")} · ${escapeHTML(item.price_type)} · ${escapeHTML(geo)}${escapeHTML(flag)}${escapeHTML(regime)}${escapeHTML(mapping)}</option>`;
      }).join("") || '<option value="">No series available</option>';
      performanceNode.value = performanceId;
    }
    return getSeriesItem(dataset, performanceId);
  }

  function forecastMarketKey(series) {
    if (series.source_key === "nbs") return JSON.stringify(["nbs", series.geography_level || "", series.geography_name || ""]);
    return JSON.stringify([series.market_id || "", series.market || "", series.admin1 || ""]);
  }

  function forecastMarketLabel(series) {
    if (series.source_key === "nbs") return `${series.geography_level || "Geography"}: ${series.geography_name || "not stated"}`;
    const location = [series.market, series.admin1, series.market_id ? `ID ${series.market_id}` : ""].filter(Boolean);
    return location.join(" · ") || "Market not stated";
  }

  function renderForecastLatest(series) {
    const rawAge = series?.forecast_source_age_months ?? series?.observation_age_months;
    const age = series && rawAge !== null && rawAge !== undefined && Number.isFinite(Number(rawAge))
      ? Number(rawAge)
      : null;
    const rawFreshnessLimit = series?.forecast_freshness_limit_months;
    const freshnessLimit = rawFreshnessLimit !== null && rawFreshnessLimit !== undefined && Number.isFinite(Number(rawFreshnessLimit))
      ? Number(series.forecast_freshness_limit_months)
      : 3;
    const forecastFreshness = series?.forecast_freshness || series?.freshness;
    const freshnessLabel = !series ? "Select a series" : age === null
      ? "Freshness unavailable"
      : (forecastFreshness === "fresh" || age <= freshnessLimit
        ? "Recent source data"
        : "Older source data");
    const values = series ? {
      forecastLatestPrice: `Latest reported: ${formatPrice(series.latest_price)}`,
      forecastLatestContext: `${series.source} · ${series.commodity}${series.research_commodity_mapping ? ` (research mapping: ${series.research_commodity_mapping}; ${String(series.mapping_status || "mapping status unavailable").replaceAll("_", " ")})` : ""} · ${series.market || series.geography_name || "geography not stated"} · ${series.methodology_regime ? String(series.methodology_regime).replaceAll("_", " ") + " · " : ""}${series.unit || "unit not stated"} · ${series.price_type}. This is a source observation, not a forecast.`,
      forecastLatestDate: formatStartingObservation(series.last_observed_month, series.latest_observation_date),
      forecastLatestAge: age === null ? "n/a" : `${formatNumber(age)} month(s) old`,
      forecastFreshness: freshnessLabel,
    } : {
      forecastLatestPrice: "Choose an exact series",
      forecastLatestContext: "Choose an exact source series. The backend will explain if its price definition or history cannot support a defensible forecast.",
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
      const status = series && age !== null ? (age <= freshnessLimit ? "fresh" : "stale") : "unavailable";
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
    const needsExactChoice = seriesId === "" && !$("#forecastSeries").disabled && $("#forecastSeries").options.length > 1;
    const canCheck = isForecastRequestableSeries(series, dataset);
    if (button) {
      button.disabled = !canCheck;
      button.textContent = canCheck ? "Generate forecast" : "Forecast unavailable";
    }
    if (note) note.textContent = canCheck
      ? (dataset === "nbs"
        ? `${formatNumber(series.observed_months)} NBS reported months (${series.first_observed_month} – ${series.last_observed_month}); the original item label, CPI regime and geography stay separate. Research mapping is descriptive only.`
        : `${formatNumber(series.observed_months)} reported WFP months (${series.first_observed_month} – ${series.last_observed_month}); priceflag=${series.price_flag || "not stated"}. Pure-actual eligibility and compatible-pool support are checked when requested.`)
      : (needsExactChoice
        ? `This ${dataset === "nbs" ? "geography" : "market"} has multiple exact price series. Select the required unit and price type before forecasting.`
        : (dataset === "nbs"
          ? "No NBS monthly observations are available for this item and geography."
          : "No WFP monthly observations are available for this market selection."));
    renderForecastLatest(canCheck ? series : null);
    showForecastPrompt(canCheck
      ? `Review the exact ${dataset === "nbs" ? "item and geography" : "market, unit and price type"} and horizon, then generate the forecast. The response will state any history eligibility limitation.`
      : (needsExactChoice
        ? "Choose the exact unit and price type before generating a forecast."
        : (dataset === "nbs"
          ? "Select an NBS item and geography with reported monthly observations; eligibility is checked without combining it with WFP."
          : "Review the exact commodity, market, unit, price type and price flag, then submit a forecast attempt.")));
  }

  function fillForecastSelectors(dataset) {
    const commodityNode = $("#forecastCommodity");
    const marketNode = $("#forecastMarket");
    const seriesNode = $("#forecastSeries");
    const candidates = getSeries(dataset).filter((item) => isForecastRequestableSeries(item, dataset));
    const emptyOption = (node, label) => {
      if (!node) return;
      node.innerHTML = `<option value="">${escapeHTML(label)}</option>`;
      node.disabled = true;
      node.value = "";
    };
    if (!candidates.length) {
      emptyOption(commodityNode, dataset === "nbs" ? "No NBS items available" : "No WFP commodities available");
      emptyOption(marketNode, dataset === "nbs" ? "No NBS geographies available" : "No markets available");
      emptyOption(seriesNode, "No reported series available");
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
    commodityNode.innerHTML = commodities.map((name) => {
      const item = candidates.find((candidate) => candidate.commodity === name);
      const mapped = dataset === "nbs" && item?.research_commodity_mapping
        ? ` — mapped research group: ${item.research_commodity_mapping}${item.mapping_status ? ` (${item.mapping_status.replaceAll("_", " ")})` : ""}`
        : "";
      return `<option value="${escapeHTML(name)}">${escapeHTML(name + mapped)}</option>`;
    }).join("");
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
      const regime = item.methodology_regime ? ` · ${item.methodology_regime.replaceAll("_", " ")}` : "";
      const mapping = dataset === "nbs" && item.research_commodity_mapping
        ? ` · mapped: ${item.research_commodity_mapping}${item.mapping_status ? ` (${item.mapping_status.replaceAll("_", " ")})` : ""}`
        : "";
      const label = `${item.unit || "Unit not stated"} · ${item.price_type} · ${dataset === "wfp" ? "actual" : "NBS report"}${regime} · ${formatNumber(item.observed_months)} months${mapping}`;
      return `<option value="${escapeHTML(item.series_id)}">${escapeHTML(label)}</option>`;
    }).join("") || '<option value="">No reported series available</option>';
    seriesNode.disabled = !variants.length;
    seriesNode.value = seriesId;
    seriesNode.querySelectorAll("option").forEach((option) => {
      const item = variants.find((candidate) => candidate.series_id === option.value);
      if (!item) return;
      const flag = item.price_flag ? `priceflag=${item.price_flag}` : "NBS reported average";
      const regime = item.methodology_regime ? ` | ${item.methodology_regime.replaceAll("_", " ")}` : "";
      const mapping = dataset === "nbs" && item.research_commodity_mapping
        ? ` | maps to ${item.research_commodity_mapping}${item.mapping_status ? ` (${item.mapping_status.replaceAll("_", " ")})` : ""}`
        : "";
      option.textContent = `${item.unit || "Unit not stated"} | ${item.price_type} | ${flag}${regime} | ${formatNumber(item.observed_months)} months | latest ${item.last_observed_month || "unknown"}${mapping}`;
    });
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

  function shiftMonth(value, offset) {
    const [year, month] = String(value).split("-").map(Number);
    const ordinal = year * 12 + month - 1 + Number(offset);
    return `${Math.floor(ordinal / 12)}-${String(ordinal % 12 + 1).padStart(2, "0")}`;
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
    const overallCards = $("#overallPerformanceCards");
    if (overallCards) overallCards.hidden = true;
    $("#performanceSummary").textContent = "Loading model evaluation…";
    try {
      const params = new URLSearchParams({ dataset, series_id: seriesId });
      const payload = await apiGet(`/api/performance?${params.toString()}`);
      if (requestId !== state.performanceRequestId || dataset !== state.dataset || seriesId !== state.performanceSelected[dataset]) return;
      const summary = payload.summary || {};
      const overall = payload.overall_evaluation === true;
      renderOverallPerformanceCards(overall ? payload.summary_cards : null);
      const viewLabel = $("#performanceViewLabel");
      if (viewLabel) {
        viewLabel.hidden = !overall;
        viewLabel.textContent = overall ? "Overall evaluation" : "";
      }
      const mapeHeading = $("#performanceMapeHeading");
      const r2Heading = $("#performanceR2Heading");
      if (mapeHeading) mapeHeading.textContent = overall ? "Median MAPE" : "MAPE";
      if (r2Heading) r2Heading.textContent = overall ? "Median R²" : "R²";
      $("#performanceSummary").textContent = payload.message || "No validated evaluation records are available.";
      const validation = summary.evaluation_period === "validation";
      const dateCount = validation ? summary.validation_date_count : summary.test_date_count;
      const firstDate = validation ? summary.validation_first_month : summary.test_first_month;
      const lastDate = validation ? summary.validation_last_month : summary.test_last_month;
      const periodLabel = validation ? "chronological validation dates" : "common historical test dates";
      const scopeLabel = summary.validation_scope ? ` · ${summary.validation_scope}` : "";
      $("#performanceDates").textContent = overall
        ? `Saved validation coverage · ${summary.evaluated_series_count || 0} exact series · ${summary.evaluated_pool_count || 0} compatible pools${summary.validation_audit_as_of ? ` · audit as of ${summary.validation_audit_as_of}` : ""} · evaluation dates differ by series`
        : dateCount
          ? `${formatNumber(dateCount)} ${periodLabel} · ${firstDate} — ${lastDate}${scopeLabel}${summary.forecast_classification ? ` · ${summary.forecast_classification}` : ""}`
          : "No evaluation dates";
      const labels = { naive_last_observation: "Naive last observation", trend_baseline: "Trend baseline", random_forest: "Random Forest", pooled_random_forest: "Pooled Random Forest", xgboost: "XGBoost", gradient_boosting: "Gradient Boosting", pooled_gradient_boosting: "Pooled Gradient Boosting", stacking: "Hybrid stacking", lstm: "LSTM" };
      $("#performanceTableBody").innerHTML = (payload.records || []).map((row) => {
        const champion = row.champion_selected_on_validation ? " · champion by validation" : "";
        const status = ["trained_and_evaluated", "validation_evaluated"].includes(row.status)
          ? `${row.status === "validation_evaluated" ? "Validation evaluated" : "Test evaluated"}${champion}${row.evaluation_scope ? ` · ${row.evaluation_scope}` : ""}`
          : ["overall_validation", "overall_coverage"].includes(row.status)
            ? `Overall ${row.status === "overall_validation" ? "validation" : "coverage"}${row.evaluation_scope ? ` · ${row.evaluation_scope}` : ""}`
          : (row.unavailable_reason || "Unavailable");
        return `<tr><td>${escapeHTML(labels[row.model] || row.model)}${row.champion_selected_on_validation ? " ★" : ""}</td><td>${formatNumber(row.rmse, 2)}</td><td>${formatNumber(row.mae, 2)}</td><td>${row.mape === null ? "—" : `${formatNumber(row.mape, 2)}%`}</td><td>${formatNumber(row.r2, 3)}</td><td>${formatNumber(row.bias, 2)}</td><td>${escapeHTML(status)}</td></tr>`;
      }).join("");
      const acceptedStatuses = ["evaluated", "validation_evaluated", "overall_evaluated", "overall_coverage"];
      setMessage("#performanceError", acceptedStatuses.includes(payload.status) ? "" : (payload.message || "No validation or test metrics are available."), !acceptedStatuses.includes(payload.status));
      const chartEmpty = $("#performanceChartEmpty");
      if (chartEmpty) chartEmpty.textContent = overall
        ? "The saved overall summary contains validation metrics only; no compatible actual-versus-predicted rows are available to chart."
        : "No actual-versus-predicted rows are available for this evaluation scope.";
      drawPerformance(payload.predictions_by_date || [], summary.evaluation_period || "test");
    } catch (error) {
      if (requestId !== state.performanceRequestId || dataset !== state.dataset || seriesId !== state.performanceSelected[dataset]) return;
      setMessage("#performanceError", error.message, true);
    }
  }

  function renderOverallPerformanceCards(cards) {
    const section = $("#overallPerformanceCards");
    if (!section) return;
    section.hidden = !cards;
    if (!cards) return;
    const setText = (selector, value) => {
      const node = $(selector);
      if (node) node.textContent = String(value ?? "");
    };
    const metricText = (value, digits, suffix = "") => {
      const number = Number(value);
      return value !== null && value !== undefined && value !== "" && Number.isFinite(number)
        ? `${formatNumber(number, digits)}${suffix}`
        : "Not available";
    };
    setText("#overallPerformanceCardsTitle", cards.label || "Overall validation performance");
    setText("#overallRmse", metricText(cards.rmse, 2));
    setText("#overallRmseNote", cards.rmse === null || cards.rmse === undefined
      ? (cards.rmse_unavailable_reason || "No defensible common-scale RMSE is available.")
      : "Overall validation RMSE.");
    setText("#overallMae", metricText(cards.mae, 2));
    setText("#overallMaeNote", cards.mae === null || cards.mae === undefined
      ? (cards.mae_unavailable_reason || "No defensible common-scale MAE is available.")
      : "Overall validation MAE.");
    setText("#overallMapeLabel", cards.mape_label || "MAPE");
    setText("#overallMape", metricText(cards.mape, 2, "%"));
    setText("#overallMapeNote", cards.mape_method || cards.mape_unavailable_reason || "No valid overall MAPE is available.");
    setText("#overallR2Label", cards.r2_label || "R²");
    setText("#overallR2", metricText(cards.r2, 3));
    setText("#overallR2Note", cards.r2_method || cards.r2_unavailable_reason || "No valid overall R² is available.");
    const observationCount = Number(cards.validation_observations);
    setText("#overallValidationObservations", Number.isFinite(observationCount) && observationCount >= 0
      ? formatNumber(observationCount) : "Not available");
    const metricCoverage = Number(cards.metric_unit_count);
    const metricCoverageNote = Number.isFinite(metricCoverage)
      ? ` Metric medians use ${formatNumber(metricCoverage)} unambiguous evaluation units.`
      : "";
    setText("#overallValidationObservationsNote", `${cards.observation_count_method || cards.evaluation_scope || "Validation observation coverage."}${metricCoverageNote}`);
  }

  function drawPerformance(rows, evaluationPeriod = "test") {
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
      options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: "index" }, plugins: { legend: { display: true } }, scales: { x: { title: { display: true, text: evaluationPeriod === "validation" ? "Chronological validation month" : "Common chronological test month" }, ticks: { maxTicksLimit: 12 } }, y: { title: { display: true, text: "Price in source NGN/unit" } } } },
    });
  }

  function renderForecast(payload) {
    const rows = payload.forecast || [];
    const historical = payload.historical || [];
    const metadata = payload.metadata || {};
    const generationDate = String(metadata.forecast_generation_date || "");
    const generationMonth = generationDate.slice(0, 7);
    const originMonth = String(metadata.forecast_origin_month || "");
    const monthPattern = /^\d{4}-(0[1-9]|1[0-2])$/;
    const datePattern = /^\d{4}-(0[1-9]|1[0-2])-\d{2}$/;
    const requestedHorizon = Number(payload.horizon || rows.length);
    const forecastBaseMonth = originMonth > generationMonth ? originMonth : generationMonth;
    const expectedMonths = monthPattern.test(originMonth) && monthPattern.test(generationMonth)
      && Number.isInteger(requestedHorizon) && requestedHorizon >= 1 && requestedHorizon <= 6
      ? monthRange(shiftMonth(forecastBaseMonth, 1), shiftMonth(forecastBaseMonth, requestedHorizon))
      : [];
    if (!rows.length || !datePattern.test(generationDate) || !monthPattern.test(originMonth)
      || rows.length !== requestedHorizon || rows.some((row, index) => {
        const month = String(row.month || "");
        return !monthPattern.test(month) || month !== expectedMonths[index]
          || !Number.isFinite(Number(row.price)) || Number(row.price) <= 0
          || (row.step !== undefined && Number(row.step) !== index + 1);
      })) {
      const error = new Error("The forecast was not displayed because its dates could not be verified as the requested future calendar months after generation.");
      error.code = "non_future_forecast_dates";
      throw error;
    }
    $("#forecastEmpty").hidden = true;
    $("#forecastResults").hidden = false;
    const series = payload.series || {};
    const rawModelLabel = metadata.model_type
      ? `${metadata.model_type}: ${String(payload.model || metadata.model || "Unavailable").replaceAll("_", " ")}`
      : String(payload.model || metadata.model || "Unavailable").replaceAll("_", " ");
    const metric = metadata.validation_metrics || {};
    const selectedModel = payload.model || metadata.model || "";
    const isBaseline = metadata.model_type === "validated baseline"
      || String(metadata.model_type || "").includes("baseline")
      || ["naive_last_observation", "trend_baseline", "robust_recent_movement", "seasonal_naive_12", "last_price_carry_forward"].includes(selectedModel);
    const isUnvalidatedFallback = String(metadata.model_type || "").includes("unvalidated");
    const modelLabel = isBaseline
      ? (isUnvalidatedFallback ? "Unvalidated baseline estimate: " : "Baseline forecast: ") + String(selectedModel || "baseline").replaceAll("_", " ")
      : rawModelLabel;
    const sourceLabel = String(series.source || metadata.source || (payload.dataset || "WFP")).toUpperCase();
    const isWorldBankEstimate = sourceLabel.includes("WORLD")
      || metadata.price_value_type === "world_bank_close_estimate"
      || metadata.is_model_estimate === true;
    const locationLabel = series.market || metadata.market || (series.geography_name ? `${series.geography_level || "Geography"}: ${series.geography_name}` : "");
    setText("forecastSubject", [series.commodity || metadata.commodity, locationLabel].filter(Boolean).join(" · ") || `Selected ${sourceLabel} series`);
    setText("forecastSource", sourceLabel);
    setText("forecastSelectedHorizon", `${payload.horizon || rows.length} month(s)`);
    setText("forecastSelectedModel", modelLabel);
    setText("forecastConfidence", formatConfidence(metadata.confidence_level));
    const latestObservation = metadata.latest_observation_date_in_source
      || metadata.latest_actual_date_in_source
      || metadata.latest_observation_month_in_source
      || metadata.latest_actual_month_in_source;
    const latestObservationMonth = metadata.latest_observation_month_in_source
      || metadata.latest_actual_month_in_source
      || latestObservation;
    setText("forecastLatestDate", formatStartingObservation(latestObservationMonth, latestObservation));
    const rawSourceAge = metadata.source_age_months;
    const freshnessLimit = Number.isFinite(Number(metadata.freshness_limit_months)) ? Number(metadata.freshness_limit_months) : 3;
    if (rawSourceAge !== null && rawSourceAge !== undefined && Number.isFinite(Number(rawSourceAge))) {
      const sourceAge = Number(rawSourceAge);
      setText("forecastLatestAge", formatNumber(sourceAge) + " month(s) old");
      setText("forecastFreshness", sourceAge <= freshnessLimit ? "Recent source data" : "Older source data");
    } else {
      setText("forecastFreshness", "Freshness unavailable");
    }
    setText("forecastOriginGenerated", formatLongDate(metadata.forecast_generation_date));
    setText("forecastRmse", formatNumber(metric.rmse, 2));
    setText("forecastMae", formatNumber(metric.mae, 2));
    setText("forecastMape", metric.mape === null || metric.mape === undefined ? "n/a" : `${formatNumber(metric.mape, 2)}%`);
    setText("forecastR2", formatNumber(metric.r2, 3));
    setText("forecastValidationCount", formatNumber(metric.n));
    const historicalLabel = metadata.historical_price_label
      || (isWorldBankEstimate ? "World Bank modeled close estimates (not direct market observations)"
        : sourceLabel === "NBS" ? "NBS reported observations" : `WFP reported priceflag=${metadata.price_flag || "actual"} observations`);
    $("#forecastChartTitle").textContent = `${historicalLabel} and estimates`;
    renderForecastTemporalStatus(payload);
    const methodology = [
      metadata.strategy_level === "exact_series_fallback"
        ? "Limited history was handled with a simple estimate from the selected series."
        : metadata.selection_basis ? "Model choice was informed by earlier observations." : "The selected model is shown above.",
      Number(metadata.forecast_bridge_months) > 0
        ? `${metadata.forecast_bridge_months} intervening month(s) were estimated to reach the forecast period; they are not actual observations.`
        : "No intervening months needed to be estimated.",
      "Confidence is a qualitative guide based on the existing classification. Validation results describe past checks and do not guarantee future prices.",
    ];
    setText("forecastMethodDetails", methodology.join(" "));
    $("#forecastTableBody").innerHTML = rows.map((row, index) => {
      const status = row.temporal_status === "historical_projection_from_latest_observation" || row.month < generationMonth
        ? "Past estimate · not actual"
        : row.temporal_status === "current_month_projection" || row.month === generationMonth
          ? "Current-month estimate"
          : "Forecast";
      const change = Number(row.percentage_change_from_origin);
      const changeLabel = Number.isFinite(change)
        ? `${change > 0 ? "+" : ""}${formatNumber(change, 1)}% from starting price`
        : "Change from starting price unavailable";
      const label = formatMonth(row.month);
      const methodLabel = row.method || metadata.method || modelLabel;
      return `<tr><td class="forecast-value-cell"><span class="forecast-month-label">${escapeHTML(label)}</span><strong class="forecast-predicted-price">${escapeHTML(formatPrice(row.price))}</strong><small class="forecast-change-label">${escapeHTML(changeLabel)}</small><small class="forecast-method-label">${escapeHTML(methodLabel)}</small></td><td>${formatNumber(row.step ?? index + 1)}</td><td>${status}</td></tr>`;
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
          { label: historicalLabel, data: labels.map((month) => actualByMonth.get(month) ?? null), borderColor: "#2c7a56", backgroundColor: "rgba(44,122,86,.08)", borderWidth: 2, pointRadius: 2, spanGaps: false },
          { label: isBaseline ? "Baseline estimate" : "Model estimate", data: labels.map((month) => forecastByMonth.get(month) ?? null), borderColor: "#cf5b45", backgroundColor: "rgba(207,91,69,.08)", borderWidth: 2, borderDash: [6, 4], pointRadius: 3, spanGaps: false },
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
    const rawAge = metadata.source_age_months ?? series.observation_age_months;
    const age = rawAge !== null && rawAge !== undefined && Number.isFinite(Number(rawAge)) ? Number(rawAge) : null;
    const freshnessLimit = Number.isFinite(Number(metadata.freshness_limit_months)) ? Number(metadata.freshness_limit_months) : 3;
    const freshness = String(series.freshness || "unavailable").toLowerCase();
    const isFresh = typeof metadata.is_current_market_forecast === "boolean"
      ? metadata.is_current_market_forecast
      : (age !== null ? age <= freshnessLimit : freshness === "fresh");
    const isRecent = age !== null ? age <= freshnessLimit : freshness === "fresh";
    let style = "stale";
    let title = "Forecast status unavailable";
    let message = "The source date or forecast period could not be confirmed. These are estimates, not reported market prices.";

    if (classification === "current_month_projection") {
      style = "mixed";
      title = "Price Forecast";
      message = "This is an estimate for the current month, not a reported market price.";
      if (!isRecent) message += " The source data are older.";
    } else if (["historical_projection_from_latest_actual", "historical_projection_from_latest_observation", "historical_only"].includes(classification)) {
      style = "historical";
      title = "Past-period estimate";
      message = "These values are estimates for past months, not reported market prices.";
    } else if (classification === "mixed_horizon_relative_to_generation") {
      style = "mixed";
      title = "Price Forecast";
      message = "Some dates are in the past or current month. Those values are estimates, not reported prices.";
      if (!isRecent) message += " The source data are older.";
    } else if (classification === "future_as_of_generation" && isFresh) {
      style = "current";
      title = "Price Forecast";
      message = "Based on recent source data.";
    } else if (classification === "future_as_of_generation") {
      style = "stale";
      title = "Price Forecast";
      message = "Based on older source data; it may not reflect current market conditions.";
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
    if (error?.code === "stale_source_data") return "Forecast unavailable — the source is too old for current conditions";
    if (error?.code === "forecast_bridge_too_long") return "Forecast unavailable — too many unobserved months to bridge";
    if (error?.code === "insufficient_history") return "Forecast unavailable — limited history did not support baseline validation";
    if (error?.code === "non_future_forecast_dates") return "Forecast unavailable — dates are not verifiably future-only";
    if (error?.code === "insufficient_local_history") return "Forecast unavailable: insufficient local history";
    if (error?.code === "pooled_validation_unavailable") return "Forecast unavailable: no validated compatible pool";
    if (error?.code === "pooled_validation_unstable") return "Forecast unavailable: pooled validation was insufficient";
    if (error?.code === "insufficient_or_gapped_history") return "Forecast unavailable — insufficient usable history";
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
    if (!isForecastRequestableSeries(selectedSeries, dataset)) {
      const reason = dataset === "nbs"
        ? "This exact NBS item/geography has no reported monthly observations available for forecasting. NBS remains separate from WFP market data."
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
    if (!/^[1-6]$/.test(horizon)) {
      const reason = "Choose a forecast horizon from 1 through 6 months.";
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
        button.disabled = !isForecastRequestableSeries(getSeriesItem(dataset, seriesId), dataset);
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
