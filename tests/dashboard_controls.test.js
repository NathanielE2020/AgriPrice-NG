const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { URL } = require("node:url");

function decode(value) {
  return value.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
}

class Option {
  constructor(value, text) { this.value = decode(value); this.textContent = decode(text); }
}

class Element {
  constructor(id) {
    this.id = id;
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.listeners = {};
    this.attributes = {};
    this.options = [];
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      toggle: (name, force) => {
        const active = force === undefined ? !classes.has(name) : force;
        if (active) classes.add(name); else classes.delete(name);
        return active;
      },
    };
    this.children = { h2: { textContent: "" }, p: { textContent: "" } };
  }
  set innerHTML(html) {
    this.html = String(html);
    this.options = [...this.html.matchAll(/<option\b[^>]*value="([^"]*)"[^>]*>([\s\S]*?)<\/option>/gi)]
      .map((match) => new Option(match[1], match[2].replace(/<[^>]*>/g, "")));
    if (this.options.length) this.value = this.options[0].value;
  }
  get innerHTML() { return this.html || ""; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelectorAll(selector) { return selector === "option" ? this.options : []; }
  querySelector(selector) { return this.children[selector] || null; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  append() {}
}

const nodes = new Map();
function getNode(selector) {
  if (!nodes.has(selector)) nodes.set(selector, new Element(selector.replace(/^#/, "")));
  return nodes.get(selector);
}

function makeSeries(id, commodity, market, flag) {
  const isFresh = id === "rice-current-fresh";
  return {
    series_id: id, source_key: "wfp", source: "WFP", commodity, market, market_id: market,
    admin1: "Kano", unit: "KG", price_type: "Retail", price_flag: flag,
    forecast_request_available: true, forecast_available: false,
    observed_months: 90, first_observed_month: "2018-01", last_observed_month: isFresh ? "2026-11" : "2026-04",
    latest_observation_date: isFresh ? "2026-11-15" : "2026-04-15", latest_price: 500, observation_age_months: isFresh ? 0 : 7,
    currency: "NGN", freshness: isFresh ? "fresh" : "aging",
  };
}

const records = [
  makeSeries("rice-actual-dawanau", "Rice (local)", "Dawanau", "actual"),
  { ...makeSeries("rice-wholesale-dawanau", "Rice (local)", "Dawanau", "actual"), unit: "50 KG", price_type: "Wholesale" },
  makeSeries("rice-actual-baga", "Rice (local)", "Baga Road", "actual"),
  makeSeries("rice-current-fresh", "Rice (local)", "Current Market", "actual"),
  { ...makeSeries("rice-13kg", "Rice (local)", "Mislabelled Bulk Market", "actual"), unit: "13 KG" },
  { ...makeSeries("rice-13decimal", "Rice (local)", "Small Pack Market", "actual"), unit: "1.3 KG" },
  { ...makeSeries("rice-500g", "Rice (local)", "Gram Unit Market", "actual"), unit: "500 G" },
  { ...makeSeries("rice-unlisted-2.3kg", "Rice (local)", "Unlisted Size Market", "actual"), unit: "2.3 KG" },
  { ...makeSeries("rice-over-limit", "Rice (local)", "Over Limit Market", "actual"), unit: "2.8001 KG" },
  { ...makeSeries("rice-liter", "Rice (local)", "Volume Unit Market", "actual"), unit: "L" },
  { ...makeSeries("rice-crate", "Rice (local)", "Package Unit Market", "actual"), unit: "30 pieces (crate)" },
  makeSeries("beans-insufficient", "Beans (red)", "Baga Road", "actual"),
  makeSeries("rice-aggregate", "Rice (local)", "Dawanau", "aggregate"),
  { ...makeSeries("millet-spaced-unit", "Millet", "Spaced Unit Market", "actual"), unit: " 2.5 Kg " },
];
const nbsRecords = [
  { ...makeSeries("nbs-semovita-post", "Semovita, Prepacked (1kg)", "National", "actual"), source_key: "nbs", source: "NBS", market: null, market_id: null, geography_level: "national", geography_name: "Nigeria", unit: "1kg", price_type: "reported_monthly_average", price_flag: "NBS reported average", methodology_regime: "post_2025_cpi_rebase", forecast_request_available: true, observed_months: 17, first_observed_month: "2025-01", last_observed_month: "2026-05", latest_observation_date: "2026-05-31", observation_age_months: 6, freshness: "recent" },
  { ...makeSeries("nbs-semovita-pre", "Semovita, Prepacked (1kg)", "National", "actual"), source_key: "nbs", source: "NBS", market: null, market_id: null, geography_level: "national", geography_name: "Nigeria", unit: "1kg", price_type: "reported_monthly_average", price_flag: "NBS reported average", methodology_regime: "pre_2025_cpi_rebase", forecast_request_available: true, observed_months: 2, first_observed_month: "2024-11", last_observed_month: "2024-12", latest_observation_date: "2024-12-31", observation_age_months: 23, freshness: "stale" },
  { ...makeSeries("nbs-unknown-unit", "Yam Tuber", "National", "actual"), source_key: "nbs", source: "NBS", geography_level: "national", geography_name: "Nigeria", unit: "Not stated in workbook", forecast_request_available: true },
];
const calls = [];
const charts = [];
const performanceResponses = new Map();
let mismatchForecastUnit = false;
function shiftMonthKey(value, offset) {
  const [year, month] = value.split("-").map(Number);
  const ordinal = year * 12 + month - 1 + offset;
  return `${Math.floor(ordinal / 12)}-${String(ordinal % 12 + 1).padStart(2, "0")}`;
}
class FakeChart {
  constructor(canvas, config) { this.canvas = canvas; this.config = config; charts.push(this); }
  destroy() { this.destroyed = true; }
}
const overview = {
  default_dataset: "wfp",
  datasets: {
    wfp: { historical_observations: 4, commodity_count: 2, market_count: 2, state_count: 1, date_start: "2018-01", date_end: "2026-04", latest_actual_observation_date: "2026-04-15", currency: "NGN" },
    nbs: { historical_observations: 1, commodity_count: 1, geography_count: 1, date_start: "2026-01", date_end: "2026-01", currency: "NGN" },
  },
  model_status: { available_model_count: 0, message: "No saved model" },
};

function apiResponse(payload, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => payload };
}

global.document = {
  ready: null,
  addEventListener(name, callback) { if (name === "DOMContentLoaded") this.ready = callback; },
  querySelector: getNode,
  getElementById(id) { return getNode("#" + id); },
  querySelectorAll() { return []; },
  createElement(id) { return new Element(id); },
};
global.Chart = FakeChart;
global.window = { Chart: FakeChart, setTimeout: () => 1 };
global.fetch = async (url) => {
  const parsed = new URL(url, "http://localhost");
  calls.push(parsed);
  if (parsed.pathname === "/api/overview") return apiResponse(overview);
  if (parsed.pathname === "/api/series") return apiResponse({
    wfp: records,
    nbs: nbsRecords,
  });
  if (parsed.pathname === "/api/history") {
    const sourceRecords = parsed.searchParams.get("dataset") === "nbs" ? nbsRecords : records;
    const selected = sourceRecords.find((record) => record.series_id === parsed.searchParams.get("series_id"));
    return apiResponse({ series: selected, history: [{ month: "2026-04", price: 500 }] });
  }
  if (parsed.pathname === "/api/performance") {
    return apiResponse(performanceResponses.get(parsed.searchParams.get("series_id"))
      || { status: "not_evaluated", message: "No evaluation", records: [], predictions_by_date: [] });
  }
  if (parsed.pathname === "/api/forecast") {
    const sourceRecords = parsed.searchParams.get("dataset") === "nbs" ? nbsRecords : records;
    const chosen = sourceRecords.find((record) => record.series_id === parsed.searchParams.get("series_id"));
    if (!chosen) return apiResponse({ error: { code: "series_not_found", message: "No exact series" } }, 404);
    const requestedHorizon = Number(parsed.searchParams.get("horizon"));
    const actualMonth = chosen.last_observed_month;
    const actualDate = chosen.latest_observation_date;
    const generated = "2026-11-19";
    const generatedMonth = generated.slice(0, 7);
    const futureBaseMonth = chosen.last_observed_month > generatedMonth ? chosen.last_observed_month : generatedMonth;
    const returnedMonths = chosen.series_id === "rice-actual-baga" && requestedHorizon === 2
      ? ["2026-11", "2026-12"]
      : Array.from({ length: requestedHorizon }, (_, index) => shiftMonthKey(futureBaseMonth, index + 1));
    const pastCount = returnedMonths.filter((month) => month <= generatedMonth).length;
    const classification = pastCount === returnedMonths.length
      ? "historical_projection_from_latest_observation"
      : pastCount ? "mixed_horizon_relative_to_generation" : "future_as_of_generation";
    const age = chosen.observation_age_months;
    const model = chosen.series_id === "rice-current-fresh" ? "naive_last_observation"
      : chosen.series_id === "beans-insufficient" ? "robust_recent_movement"
        : chosen.price_flag !== "actual" ? "robust_recent_movement" : "random_forest";
    const isFallback = ["beans-insufficient", "nbs-semovita-post", "nbs-semovita-pre"].includes(chosen.series_id) || chosen.price_flag !== "actual";
    const historicalPriceLabel = chosen.source_key === "nbs" ? "NBS reported monthly observations"
      : chosen.price_flag === "actual" ? "WFP direct actual observations" : `WFP reported priceflag=${chosen.price_flag} observations; not direct actuals`;
    return apiResponse({
      series: mismatchForecastUnit ? { ...chosen, unit: "50 KG" } : chosen, horizon: requestedHorizon, model,
      historical: [{ month: actualMonth, price: 500 }],
      forecast: returnedMonths.map((month, index) => ({ month, price: 510 + index, step: index + 1, percentage_change_from_origin: 2 + index, temporal_status: month <= generatedMonth ? "historical_projection_from_latest_observation" : "future_as_of_generation" })),
      metadata: {
        method: "Recursive monthly forecast", selection_basis: "Validation RMSE", forecast_origin_month: actualMonth,
        latest_actual_date_in_source: chosen.price_flag === "actual" ? actualDate : null, latest_actual_month_in_source: chosen.price_flag === "actual" ? actualMonth : null,
        latest_observation_date_in_source: chosen.series_id === "rice-actual-dawanau" ? "2026-04-14" : actualDate,
        source_age_months: age,
        freshness_limit_months: 3,
        source_freshness_classification: age <= 3 ? "fresh_within_3_months" : "stale_for_current_market",
        is_current_market_forecast: age <= 3 && classification === "future_as_of_generation",
        model_type: isFallback ? "unvalidated conservative baseline" : model === "naive_last_observation" ? "validated baseline" : "individual-series model",
        forecast_generation_date: generated,
        forecast_classification: classification, forecast_bridge_months: Math.max(0, Number(generatedMonth.slice(0, 4)) * 12 + Number(generatedMonth.slice(5, 7)) - Number(actualMonth.slice(0, 4)) * 12 - Number(actualMonth.slice(5, 7))),
        transformation: "source-reported monthly price", currency: "NGN", unit: chosen.unit,
        historical_price_label: historicalPriceLabel, price_flag: chosen.price_flag || null,
        confidence_level: isFallback || age > 3 ? "low" : "limited",
        validation_metrics: isFallback ? { rmse: null, mae: null, mape: null, r2: null, n: 0 } : { rmse: 4, mae: 3, mape: 1, r2: 0.8, n: 12 }, currency: "NGN", unit: chosen.unit,
      },
      warning: "Future estimates based on the exact selected source series.",
    });
  }
  return apiResponse({ error: { message: "Unexpected request" } }, 404);
};

async function settle() {
  for (let index = 0; index < 5; index += 1) await new Promise((resolve) => setImmediate(resolve));
}

async function trigger(id, event) {
  const element = getNode("#" + id);
  assert.equal(typeof element.listeners[event], "function", id + " must have a " + event + " handler");
  await element.listeners[event]({ target: element });
  await settle();
}

async function main() {
  vm.runInThisContext(fs.readFileSync("static/js/dashboard.js", "utf8"), { filename: "dashboard.js" });
  await document.ready();

  const commodity = getNode("#forecastCommodity");
  const market = getNode("#forecastMarket");
  const exactSeries = getNode("#forecastSeries");
  const horizon = getNode("#forecastHorizon");
  assert.deepEqual(commodity.options.map((option) => option.value), ["Beans (red)", "Millet", "Rice (local)"]);
  assert.equal(getNode("#latestDate").textContent, "15th April 2026");
  assert.equal(getNode("#overallDateRange").textContent, "All WFP price flags: January 2018 – April 2026");
  assert.equal((getNode("#statGrid").innerHTML.match(/class="stat-card"/g) || []).length, 5);
  assert.match(getNode("#statGrid").innerHTML, /class="stat-label"[\s\S]*class="stat-value"[\s\S]*class="stat-sub"/);
  const dashboardTemplate = fs.readFileSync("templates/dashboard.html", "utf8");
  const dashboardCss = fs.readFileSync("static/css/dashboard.css", "utf8");
  assert.match(dashboardTemplate, /About the Project/);
  assert.doesNotMatch(dashboardTemplate, /CURRENT STATUS|What is available now|Source freshness limits interpretation|Current-market forecasting is unavailable/);
  assert.doesNotMatch(dashboardTemplate, /modelAvailabilityNote|reported observations \(/i);
  assert.match(dashboardCss, /\.stat-grid \{ display: grid; gap: 14px;/);
  assert.match(dashboardCss, /@media \(max-width: 380px\) \{[\s\S]*\.stat-grid \{ grid-template-columns: 1fr;/);
  assert.ok(!getNode("#analysisSeries").options.some((option) => option.value === "rice-wholesale-dawanau"), "bulk units stay out of historical selection");
  assert.ok(!getNode("#analysisSeries").options.some((option) => option.value === "rice-13kg"), "13 KG is not treated as 1.3 KG");
  assert.ok(getNode("#analysisSeries").options.some((option) => option.value === "rice-13decimal"), "an actual 1.3 KG series remains selectable");
  assert.ok(getNode("#analysisSeries").options.some((option) => option.value === "rice-500g"), "a source-recorded 500 G series remains selectable");
  assert.ok(getNode("#analysisSeries").options.some((option) => option.value === "rice-unlisted-2.3kg"), "an unlisted but valid 2.3 KG size remains selectable without an allowlist");
  assert.ok(!getNode("#analysisSeries").options.some((option) => option.value === "rice-over-limit"), "a weight just above 2.8 KG is excluded");
  assert.ok(!getNode("#analysisSeries").options.some((option) => option.value === "rice-liter"), "volume units are excluded from history selection");
  assert.ok(!getNode("#analysisSeries").options.some((option) => option.value === "rice-crate"), "count/package units are excluded from history selection");
  assert.ok(getNode("#analysisSeries").options.some((option) => option.value === "rice-aggregate"), "small-unit aggregate history remains inspectable");
  assert.ok(getNode("#analysisSeries").options.some((option) => option.value === "millet-spaced-unit"), "unit matching ignores capitalization and spacing while retaining source text");
  getNode("#analysisSeries").value = "rice-unlisted-2.3kg";
  const historyChartsBeforeUnlistedSize = charts.length;
  await trigger("analysisSeries", "change");
  assert.equal(calls.filter((url) => url.pathname === "/api/history").at(-1).searchParams.get("series_id"), "rice-unlisted-2.3kg", "the unlisted valid weight resolves to its exact history series");
  assert.equal(charts.length, historyChartsBeforeUnlistedSize + 1, "the selected unlisted size is rendered in the history chart");
  assert.deepEqual(charts.at(-1).config.data.datasets[0].data, [500]);
  assert.doesNotMatch(getNode("#overviewSeriesGrid").innerHTML, /50 KG/);
  assert.doesNotMatch(getNode("#overviewSeriesGrid").innerHTML, /13 KG/);
  assert.match(getNode("#overviewSeriesGrid").innerHTML, /January 2018 – April 2026/);
  commodity.value = "Rice (local)";
  await trigger("forecastCommodity", "change");
  const baga = market.options.find((option) => option.textContent.includes("Baga Road"));
  assert.ok(baga, "commodity selection should populate its available markets");
  market.value = baga.value;
  await trigger("forecastMarket", "change");
  assert.equal(exactSeries.value, "rice-actual-baga", "market selection must resolve to the exact actual series id");
  const dawanau = market.options.find((option) => option.textContent.includes("Dawanau"));
  market.value = dawanau.value;
  await trigger("forecastMarket", "change");
  assert.equal(exactSeries.options.length, 1, "only the exact small-unit pure-actual series is offered for WFP forecasting");
  assert.ok(!market.options.some((option) => option.textContent.includes("Over Limit Market")), "over-limit weights are omitted from forecast market choices");
  assert.equal(exactSeries.value, "rice-actual-dawanau");
  assert.ok(!exactSeries.options.some((option) => option.value === "rice-aggregate"), "aggregate flags remain history-only");
  assert.ok(!exactSeries.options.some((option) => option.value === "rice-wholesale-dawanau"), "bulk prices cannot appear under a small-unit label");
  assert.ok(!exactSeries.options.some((option) => option.value === "rice-13kg"), "13 KG cannot appear as a small-unit forecast option");
  assert.equal(exactSeries.options.find((option) => option.value === "rice-actual-dawanau").textContent, "KG · Retail · Actual", "exact-series selector is concise and readable");
  assert.equal(getNode("#generateForecast").disabled, false);

  const gramMarket = market.options.find((option) => option.textContent.includes("Gram Unit Market"));
  market.value = gramMarket.value;
  await trigger("forecastMarket", "change");
  assert.equal(exactSeries.value, "rice-500g", "forecast selection resolves the exact gram-based series");
  assert.equal(exactSeries.options[0].textContent, "500 G · Retail · Actual", "the recorded gram label is not converted for display");
  horizon.value = "1";
  await trigger("generateForecast", "click");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("series_id"), "rice-500g");

  const unlistedSizeMarket = market.options.find((option) => option.textContent.includes("Unlisted Size Market"));
  market.value = unlistedSizeMarket.value;
  await trigger("forecastMarket", "change");
  assert.equal(exactSeries.value, "rice-unlisted-2.3kg", "an unlisted valid quantity reaches the exact forecast selector");
  assert.ok(exactSeries.options[0].textContent.startsWith("2.3 KG") && exactSeries.options[0].textContent.includes("Retail"), "the original unit text remains visible");
  await trigger("generateForecast", "click");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("series_id"), "rice-unlisted-2.3kg");

  market.value = dawanau.value;
  await trigger("forecastMarket", "change");
  assert.equal(exactSeries.value, "rice-actual-dawanau", "returning to another market restores only that market's exact series");

  horizon.value = "1";
  await trigger("generateForecast", "click");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("series_id"), "rice-actual-dawanau");
  assert.equal(getNode("#forecastResults").hidden, false, "a selected small-unit pure-actual series can use its existing estimate path");
  assert.match(getNode("#forecastTableBody").innerHTML, /Dec 2026[\s\S]*₦510\.00[\s\S]*\+2\.0%/);

  const historyForecastContext = getNode("#forecastLatestContext").textContent;
  getNode("#analysisSeries").value = "rice-aggregate";
  await trigger("analysisSeries", "change");
  assert.equal(exactSeries.value, "rice-actual-dawanau");
  assert.equal(getNode("#forecastLatestContext").textContent, historyForecastContext, "historical selection must not rewrite forecast summary");
  getNode("#analysisStartDate").value = "2026-02";
  getNode("#analysisEndDate").value = "2026-04";
  await trigger("analysisStartDate", "change");
  await trigger("refreshHistory", "click");
  const filteredHistory = calls.filter((url) => url.pathname === "/api/history").at(-1);
  assert.equal(filteredHistory.searchParams.get("start_date"), "2026-02");
  assert.equal(filteredHistory.searchParams.get("end_date"), "2026-04");
  getNode("#performanceSeries").value = "rice-actual-baga";
  performanceResponses.set("rice-actual-baga", {
    status: "validation_evaluated", message: "Individual validation results for the selected series.",
    series: records.find((record) => record.series_id === "rice-actual-baga"),
    summary: { evaluation_period: "validation", validation_scope: "exact selected series", validation_date_count: 1, validation_first_month: "2026-03", validation_last_month: "2026-03", validation_observations: 1 },
    records: [{ model: "naive_last_observation", status: "validation_evaluated", rmse: 3, mae: 2, mape: 1, r2: 0.5, bias: 0, n: 1, evaluation_scope: "exact selected series" }],
    predictions_by_date: [{ month: "2026-03", models: { naive_last_observation: { actual: 100, predicted: 98 } } }],
  });
  await trigger("performanceSeries", "change");
  assert.equal(calls.filter((url) => url.pathname === "/api/performance").at(-1).searchParams.get("series_id"), "rice-actual-baga");
  assert.equal(getNode("#performanceViewLabel").hidden, true, "validated selected-series results do not switch to the overall view");
  assert.equal(getNode("#overallPerformanceCards").hidden, true, "overall summary cards stay hidden for individual results");
  assert.equal(getNode("#performanceRmseHeading").hidden, false, "individual metrics retain their original RMSE column");
  assert.equal(getNode("#performanceMaeHeading").hidden, false, "individual metrics retain their original MAE column");
  assert.equal(getNode("#performanceBiasHeading").hidden, false, "individual metrics retain their original bias column");
  assert.equal(getNode("#performanceMapeHeading").textContent, "MAPE");
  assert.equal((getNode("#performanceTableBody").innerHTML.match(/<td>/g) || []).length, 7, "individual rows retain all seven original columns");
  const individualTable = getNode("#performanceTableBody").innerHTML;
  assert.match(individualTable, /3\.00/);
  assert.match(individualTable, /2\.00/);
  assert.match(individualTable, /1\.00%/);
  assert.match(individualTable, /0\.500/);
  assert.match(individualTable, /Validation evaluated/);
  assert.equal(getNode("#performanceChartEmpty").hidden, true, "actual-versus-predicted rows remain charted when available");

  const chartsBeforeOverallWithoutPredictions = charts.length;
  performanceResponses.set("rice-aggregate", {
    status: "overall_evaluated", overall_evaluation: true,
    message: "No individual validation results are available for Rice. This WFP summary is not a result for the selected commodity.",
    series: records.find((record) => record.series_id === "rice-aggregate"),
    summary: { evaluation_period: "validation", evaluated_series_count: 744, evaluated_pool_count: 38, validation_audit_as_of: "2026-11-19" },
    summary_cards: {
      label: "Overall validation performance", source: "WFP", evaluation_scope: "exact-series chronological validation",
      rmse: null, mae: null, rmse_unavailable_reason: "Absolute errors cannot be combined across different price scales.",
      mae_unavailable_reason: "Absolute errors cannot be combined across different price scales.",
      mape: 9.96, mape_label: "Median per-series MAPE", mape_method: "Median of genuine per-series MAPE values.",
      r2: 0.303, r2_label: "Median per-series R²", r2_method: "Median of genuine per-series R² values.",
      validation_observations: 9520, evaluation_unit_count: 744, metric_unit_count: 744,
      observation_count_method: "Sum across unique exact-series validation observations.",
    },
    records: [
      { model: "naive_last_observation", status: "overall_validation", rmse: null, mae: null, mape: 9.96, r2: 0.303, bias: null, evaluation_unit_count: 744, evaluation_scope: "Exact-series chronological validation; median per-series MAPE/R²" },
      { model: "pooled_gradient_boosting", status: "overall_validation", rmse: null, mae: null, mape: 8.4, r2: 0.41, bias: null, evaluation_unit_count: 38, evaluation_scope: "Compatible-pool group-held-out validation; median per-pool MAPE/R²" },
    ],
    predictions_by_date: [],
  });
  getNode("#performanceSeries").value = "rice-aggregate";
  await trigger("performanceSeries", "change");
  assert.equal(getNode("#performanceViewLabel").textContent, "Overall evaluation");
  assert.equal(getNode("#performanceViewLabel").hidden, false);
  assert.equal(getNode("#overallPerformanceCards").hidden, false);
  assert.equal(getNode("#overallPerformanceCardsTitle").textContent, "Overall validation performance");
  assert.equal(getNode("#overallPerformanceScope").textContent, "Evaluation scope: exact-series chronological validation.");
  assert.equal(getNode("#overallRmse").textContent, "Not comparable");
  assert.equal(getNode("#overallRmseNote").textContent, "Not comparable across different commodities and price scales.");
  assert.equal(getNode("#overallMae").textContent, "Not comparable");
  assert.equal(getNode("#overallMaeNote").textContent, "Not comparable across different commodities and price scales.");
  assert.equal(getNode("#overallMapeLabel").textContent, "Median per-series MAPE");
  assert.equal(getNode("#overallMape").textContent, "9.96%");
  assert.equal(getNode("#overallR2").textContent, "0.303");
  assert.equal(getNode("#overallValidationObservations").textContent, "9,520");
  assert.match(getNode("#overallValidationObservationsNote").textContent, /744 unambiguous evaluation units/);
  assert.equal(getNode("#performanceRmseHeading").hidden, true);
  assert.equal(getNode("#performanceMaeHeading").hidden, true);
  assert.equal(getNode("#performanceBiasHeading").hidden, true);
  assert.equal(getNode("#performanceMapeHeading").textContent, "Median MAPE");
  assert.match(getNode("#performanceDates").textContent, /744 exact series.*38 compatible pools.*19th November 2026.*dates differ/);
  assert.match(getNode("#performanceSummary").textContent, /Overall WFP evaluation.*not a result for the selected commodity.*medians.*not a head-to-head comparison/);
  const overallTable = getNode("#performanceTableBody").innerHTML;
  assert.equal((overallTable.match(/<td>/g) || []).length, 8, "overall rows contain only Model, Median MAPE, Median R² and scope");
  assert.match(overallTable, /9\.96%/);
  assert.match(overallTable, /0\.303/);
  assert.match(overallTable, /Exact-series chronological validation/);
  assert.match(overallTable, /Compatible-pool group-held-out validation/);
  assert.doesNotMatch(overallTable, /n\/a|999999/i, "overall table omits unavailable errors and research test metrics");
  assert.equal(getNode("#performanceChartEmpty").hidden, false);
  assert.equal(getNode("#performanceChartEmpty").textContent, "The overall summary contains evaluation scores; this chart appears when compatible prediction rows are saved.");
  assert.equal(charts.length, chartsBeforeOverallWithoutPredictions, "missing overall predictions do not fabricate chart points or suppress metrics");

  const chartCountBeforeOverallPredictions = charts.length;
  performanceResponses.set("rice-aggregate", {
    ...performanceResponses.get("rice-aggregate"),
    predictions_by_date: [{ month: "2026-03", models: { naive_last_observation: { actual: 100, predicted: 98 } } }],
  });
  await trigger("performanceSeries", "change");
  assert.equal(getNode("#performanceChartEmpty").hidden, true, "genuine compatible prediction rows are charted when supplied");
  assert.equal(charts.length, chartCountBeforeOverallPredictions + 1);
  assert.deepEqual(charts.at(-1).config.data.datasets[0].data, [100]);
  assert.deepEqual(charts.at(-1).config.data.datasets[1].data, [98]);

  performanceResponses.set("rice-aggregate", {
    status: "overall_coverage", overall_evaluation: true, message: "No valid overall scores are available.",
    summary: { evaluation_period: "validation", evaluated_series_count: 1, evaluated_pool_count: 0 },
    summary_cards: {
      label: "Overall validation performance", rmse: null, mae: null, mape: null, r2: null,
      rmse_unavailable_reason: "No compatible scale.", mae_unavailable_reason: "No compatible scale.",
      mape_unavailable_reason: "No valid MAPE.", r2_unavailable_reason: "No valid R².",
      validation_observations: 0, metric_unit_count: 0, observation_count_method: "No eligible observations.",
    },
    records: [], predictions_by_date: [],
  });
  await trigger("performanceSeries", "change");
  assert.equal(getNode("#overallPerformanceCards").hidden, false);
  assert.equal(getNode("#overallRmse").textContent, "Not comparable");
  assert.equal(getNode("#overallMape").textContent, "Not available");
  assert.equal(getNode("#overallR2").textContent, "Not available");
  assert.equal(getNode("#overallValidationObservations").textContent, "0");
  performanceResponses.set("rice-actual-baga", {
    status: "validation_evaluated", overall_evaluation: false,
    summary: { evaluation_period: "validation", validation_observations: 1 },
    records: [{ model: "naive_last_observation", status: "validation_evaluated", rmse: 3, mae: 2, mape: 1, r2: 0.5, bias: 0, n: 1 }],
    predictions_by_date: [{ month: "2026-03", models: { naive_last_observation: { actual: 100, predicted: 98 } } }],
  });
  getNode("#performanceSeries").value = "rice-actual-baga";
  await trigger("performanceSeries", "change");
  assert.equal(getNode("#overallPerformanceCards").hidden, true, "returning to individual validation hides overall cards");
  assert.equal(getNode("#performanceRmseHeading").hidden, false, "returning to individual evaluation restores raw-error columns");

  const forecastCallsBeforeHorizon = calls.filter((url) => url.pathname === "/api/forecast").length;
  horizon.value = "6";
  await trigger("forecastHorizon", "change");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").length, forecastCallsBeforeHorizon, "changing horizon must wait for Generate forecast");
  const generateButton = getNode("#generateForecast");
  const pendingForecast = generateButton.listeners.click({ target: generateButton });
  assert.equal(generateButton.disabled, true, "forecast action enters a loading state");
  assert.match(generateButton.textContent, /Generating forecast/);
  await pendingForecast;
  await settle();
  const successCall = calls.filter((url) => url.pathname === "/api/forecast").at(-1);
  assert.equal(successCall.searchParams.get("series_id"), "rice-actual-dawanau");
  assert.equal(successCall.searchParams.get("horizon"), "6");
  assert.equal(successCall.searchParams.get("model"), "auto");
  assert.match(getNode("#forecastTableBody").innerHTML, /Dec 2026[\s\S]*May 2027/);
  assert.equal((getNode("#forecastTableBody").innerHTML.match(/<tr>/g) || []).length, 6, "the six-month action renders six forecast dates");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Price Forecast");
  assert.equal(getNode("#forecastStatusMessage").textContent, "Future-month prices are estimates, not reported prices.");
  assert.doesNotMatch(getNode("#forecastStatusMessage").textContent, /market conditions|older source data/i);
  assert.equal(getNode("#forecastConfidence").textContent, "Low", "the existing low-confidence classification is presented in plain language");
  assert.match(getNode("#forecastTableBody").innerHTML, /<td>Forecast<\/td>/);
  assert.equal(getNode("#forecastLatestDate").textContent, "14th April 2026", "the exact starting observation date is readable without an ISO duplicate");
  assert.equal(getNode("#forecastFreshness").textContent, "Older data", "the result clearly identifies older data");
  assert.equal(getNode("#forecastOriginGenerated").textContent, "19th November 2026");
  assert.equal(getNode("#forecastSource").textContent, "WFP");
  assert.match(getNode("#forecastSelectedModel").textContent, /random forest/);
  assert.equal(getNode("#forecastSelectedHorizon").textContent, "6 month(s)");
  assert.equal(getNode("#forecastRmse").textContent, "4.00");
  assert.equal(getNode("#forecastMae").textContent, "3.00");
  assert.equal(getNode("#forecastMape").textContent, "1.00%");
  assert.equal(getNode("#forecastR2").textContent, "0.800");
  assert.equal(getNode("#forecastValidationCount").textContent, "12");
  assert.match(getNode("#forecastMethodDetails").textContent, /Model choice was informed by earlier observations/);
  assert.doesNotMatch(getNode("#forecastMethodDetails").textContent, /Generated 2026-11-19/);
  assert.deepEqual(charts.at(-1).config.data.datasets.map((dataset) => dataset.label), ["WFP direct actual observations", "Model estimate"]);

  const bagaMarket = market.options.find((option) => option.textContent.includes("Baga Road"));
  market.value = bagaMarket.value;
  await trigger("forecastMarket", "change");
  horizon.value = "2";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastResults").hidden, true, "a response outside the generation-relative requested future calendar months is not rendered");
  assert.equal(getNode("#forecastEmpty h2").textContent, "Price Forecast");
  assert.match(getNode("#forecastEmpty p").textContent, /follows the selected series' latest observation/);
  horizon.value = "3";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Price Forecast");
  assert.match(getNode("#forecastTableBody").innerHTML, /<td>Forecast<\/td>/);

  const currentMarket = market.options.find((option) => option.textContent.includes("Current Market"));
  market.value = currentMarket.value;
  await trigger("forecastMarket", "change");
  horizon.value = "1";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Price Forecast");
  assert.equal(getNode("#forecastStatusMessage").textContent, "Future-month prices are estimates, not reported prices.");
  assert.equal(getNode("#forecastFreshness").textContent, "Recent data");
  assert.equal(getNode("#forecastConfidence").textContent, "Medium", "the existing limited-confidence classification is presented in plain language");
  assert.match(getNode("#forecastSelectedModel").textContent, /^Baseline forecast: naive last observation$/);
  assert.deepEqual(charts.at(-1).config.data.datasets.map((dataset) => dataset.label), ["WFP direct actual observations", "Baseline estimate"]);

  for (const requested of ["2", "4", "5"]) {
    horizon.value = requested;
    await trigger("forecastHorizon", "change");
    await trigger("generateForecast", "click");
    const request = calls.filter((url) => url.pathname === "/api/forecast").at(-1);
    assert.equal(request.searchParams.get("horizon"), requested, "every supported horizon reaches the forecast API");
    assert.equal((getNode("#forecastTableBody").innerHTML.match(/<tr>/g) || []).length, Number(requested));
    assert.match(getNode("#forecastTableBody").innerHTML, /<td>Forecast<\/td>/);
  }

  commodity.value = "Beans (red)";
  await trigger("forecastCommodity", "change");
  await trigger("generateForecast", "click");
  assert.equal(exactSeries.value, "beans-insufficient");
  assert.equal(getNode("#forecastResults").hidden, false, "limited history receives a qualified exact-series fallback");
  assert.match(getNode("#forecastSelectedModel").textContent, /robust recent movement/);
  assert.match(getNode("#forecastMethodDetails").textContent, /confidence is a qualitative guide/i);

  getNode("#globalDataset").value = "nbs";
  await trigger("globalDataset", "change");
  assert.equal(getNode("#forecastDataset").value, "wfp", "overview source changes must not alter forecast source");
  assert.equal(exactSeries.value, "beans-insufficient");
  getNode("#analysisDataset").value = "nbs";
  await trigger("analysisDataset", "change");
  assert.equal(getNode("#forecastDataset").value, "wfp", "historical source must remain independent from forecast source");
  getNode("#forecastDataset").value = "nbs";
  await trigger("forecastDataset", "change");
  assert.equal(commodity.disabled, false);
  assert.equal(commodity.value, "Semovita, Prepacked (1kg)");
  assert.equal(market.options[0].textContent, "national: Nigeria");
  assert.equal(exactSeries.value, "", "the pre/post-rebase NBS observations require an explicit methodology choice");
  assert.ok(!exactSeries.options.some((option) => option.value === "nbs-unknown-unit"), "unitless NBS records are not selectable");
  const preNbsOption = exactSeries.options.find((option) => option.value === "nbs-semovita-pre");
  const postNbsOption = exactSeries.options.find((option) => option.value === "nbs-semovita-post");
  assert.ok(postNbsOption.textContent.includes("Post-2025 CPI base"));
  assert.ok(preNbsOption.textContent.includes("Pre-2025 CPI base"));
  assert.ok(preNbsOption.textContent.includes("1kg"));
  assert.ok(preNbsOption.textContent.includes("Monthly reported average"));
  assert.ok(preNbsOption.textContent.includes("NBS reported average"));
  assert.doesNotMatch(preNbsOption.textContent, /reported_monthly_average|priceflag=|\d+ reported observation|latest 202\d|pre_2025/i);
  assert.equal(getNode("#generateForecast").disabled, true);
  exactSeries.value = "nbs-semovita-post";
  await trigger("forecastSeries", "change");
  assert.equal(getNode("#forecastLatestPrice").textContent, "Latest reported: ₦500.00");
  assert.equal(getNode("#forecastLatestDate").textContent, "31st May 2026");
  assert.equal(getNode("#forecastDetailSource").textContent, "NBS");
  assert.equal(getNode("#forecastDetailItem").textContent, "Semovita, Prepacked (1kg)");
  assert.equal(getNode("#forecastDetailGeography").textContent, "national: Nigeria");
  assert.equal(getNode("#forecastDetailUnit").textContent, "1kg");
  assert.equal(getNode("#forecastDetailPriceType").textContent, "Monthly reported average");
  assert.equal(getNode("#forecastDetailPriceFlag").textContent, "NBS reported average");
  assert.equal(getNode("#forecastDetailRegime").textContent, "Post-2025 CPI base");
  assert.equal(getNode("#forecastDetailSeriesId").textContent, "nbs-semovita-post");
  assert.equal(getNode("#generateForecast").disabled, false, "reported NBS history is sent to exact-series eligibility validation");
  await trigger("generateForecast", "click");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("dataset"), "nbs");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("series_id"), "nbs-semovita-post");
  assert.equal(getNode("#forecastResults").hidden, false, "an exact NBS item/geography/regime with positive history receives an estimate");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Price Forecast");
  assert.equal(getNode("#forecastSource").textContent, "NBS");
  assert.match(getNode("#forecastSubject").textContent, /Semovita, Prepacked \(1kg\).*per 1kg.*Nigeria/);

  mismatchForecastUnit = true;
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastResults").hidden, true, "a response for another unit is rejected rather than shown under the selected series");
  assert.equal(getNode("#forecastEmpty h2").textContent, "Price Forecast");
  assert.match(getNode("#forecastEmpty p").textContent, /No other unit or location was substituted/);
  mismatchForecastUnit = false;
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastResults").hidden, false, "the exact-unit response remains available after rejecting a mismatch");

  console.log("Frontend workflow passed: simple forecast wording, visible starting date and freshness, confidence labels, generation-relative dates, and limited-history estimates.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
