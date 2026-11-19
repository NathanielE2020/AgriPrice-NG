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
  makeSeries("beans-insufficient", "Beans (red)", "Baga Road", "actual"),
  makeSeries("rice-aggregate", "Rice (local)", "Dawanau", "aggregate"),
];
const calls = [];
const charts = [];
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
    nbs: [
      { ...makeSeries("nbs-yam-post", "Yam Tuber", "National", "actual"), source_key: "nbs", source: "NBS", market: null, market_id: null, geography_level: "national", geography_name: "Nigeria", methodology_regime: "post_2025_cpi_rebase", forecast_request_available: true, observed_months: 17, first_observed_month: "2025-01", last_observed_month: "2026-05", observation_age_months: 6, freshness: "recent" },
      { ...makeSeries("nbs-yam-pre", "Yam Tuber", "National", "actual"), source_key: "nbs", source: "NBS", market: null, market_id: null, geography_level: "national", geography_name: "Nigeria", methodology_regime: "pre_2025_cpi_rebase", forecast_request_available: true, observed_months: 2, first_observed_month: "2024-11", last_observed_month: "2024-12", observation_age_months: 23, freshness: "stale" },
    ],
  });
  if (parsed.pathname === "/api/history") {
    const selected = records.find((record) => record.series_id === parsed.searchParams.get("series_id"));
    return apiResponse({ series: selected, history: [{ month: "2026-04", price: 500 }] });
  }
  if (parsed.pathname === "/api/performance") return apiResponse({ status: "not_evaluated", message: "No evaluation", records: [], predictions_by_date: [] });
  if (parsed.pathname === "/api/forecast") {
    if (parsed.searchParams.get("dataset") === "nbs") {
      const nbsId = parsed.searchParams.get("series_id");
      const message = nbsId === "nbs-yam-pre"
        ? "Only 0 gap-aware supervised rows are usable from 2 NBS observations; at least 30 are required."
        : "Only 14 gap-aware supervised rows are usable from 17 NBS observations; at least 30 are required.";
      return apiResponse({ error: { code: "insufficient_or_gapped_history", message } }, 422);
    }
    if (parsed.searchParams.get("series_id") === "beans-insufficient") {
      return apiResponse({ error: { code: "insufficient_or_gapped_history", message: "Only 17 gap-aware supervised rows are usable; insufficient history." } }, 422);
    }
    const selected = records.find((record) => record.series_id === parsed.searchParams.get("series_id"));
    if (selected && selected.price_flag !== "actual") {
      return apiResponse({ error: { code: "not_direct_actual", message: "WFP aggregate and mixed-price-flag records cannot be forecast as direct actual prices." } }, 422);
    }
    const requestedHorizon = Number(parsed.searchParams.get("horizon"));
    const futureMonths = ["2026-12", "2027-01", "2027-02", "2027-03", "2027-04", "2027-05"];
    const returnedMonths = selected.series_id === "rice-actual-baga" && requestedHorizon === 2
      ? ["2026-11", "2026-12"]
      : futureMonths.slice(0, requestedHorizon);
    const actualMonth = selected.last_observed_month;
    const actualDate = selected.latest_observation_date;
    return apiResponse({
      series: selected, horizon: requestedHorizon, model: selected.series_id === "rice-current-fresh" ? "naive_last_observation" : "random_forest",
      historical: [{ month: actualMonth, price: 500 }],
      forecast: returnedMonths.map((month, index) => ({ month, price: 510 + index, step: index + 1, temporal_status: "future_as_of_generation" })),
      metadata: {
        method: "Recursive monthly forecast", selection_basis: "Validation RMSE", forecast_origin_month: actualMonth,
        latest_actual_date_in_source: actualDate, latest_actual_month_in_source: actualMonth,
        latest_observation_date_in_source: selected.series_id === "rice-actual-dawanau" ? "2026-04-14" : actualDate,
        source_age_months: selected.observation_age_months,
        source_freshness_classification: selected.observation_age_months <= 3 ? "fresh_within_3_months" : "stale_for_current_market",
        model_type: selected.series_id === "rice-current-fresh" ? "validated baseline" : "individual-series model",
        forecast_generation_date: "2026-11-19",
        forecast_classification: "future_as_of_generation", transformation: "original WFP actual price", currency: "NGN", unit: "KG",
        validation_metrics: { rmse: 4, mae: 3, mape: 1, r2: 0.8, n: 12 }, currency: "NGN", unit: "KG",
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
  assert.deepEqual(commodity.options.map((option) => option.value), ["Beans (red)", "Rice (local)"]);
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
  assert.equal(exactSeries.options.length, 4, "same-market flag, unit and price-type variants require an explicit choice");
  assert.equal(exactSeries.options[0].value, "", "no exact unit/type is silently selected when variants are ambiguous");
  assert.ok(exactSeries.options.some((option) => option.value === "rice-aggregate"), "aggregate selections remain visible for an explicit eligibility attempt");
  assert.match(exactSeries.options.find((option) => option.value === "rice-aggregate").textContent, /priceflag=aggregate/);
  exactSeries.value = "rice-aggregate";
  await trigger("forecastSeries", "change");
  assert.equal(getNode("#generateForecast").disabled, false, "non-actual WFP records can be explicitly checked rather than silently omitted");
  horizon.value = "1";
  await trigger("generateForecast", "click");
  const rejectedAggregateRequest = calls.filter((url) => url.pathname === "/api/forecast").at(-1);
  assert.equal(rejectedAggregateRequest.searchParams.get("series_id"), "rice-aggregate");
  assert.match(getNode("#forecastEmpty p").textContent, /aggregate and mixed-price-flag/);
  exactSeries.value = "rice-actual-dawanau";
  await trigger("forecastSeries", "change");

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
  await trigger("performanceSeries", "change");
  assert.equal(calls.filter((url) => url.pathname === "/api/performance").at(-1).searchParams.get("series_id"), "rice-actual-baga");

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
  assert.match(getNode("#forecastTableBody").innerHTML, /2026-12/);
  assert.equal((getNode("#forecastTableBody").innerHTML.match(/<tr>/g) || []).length, 6, "the six-month action renders six forecast dates");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Future projection · aging source data");
  assert.match(getNode("#forecastStatusMessage").textContent, /not a current market forecast/);
  assert.doesNotMatch(getNode("#forecastTableBody").innerHTML, /Past at generation/);
  assert.equal(getNode("#forecastLatestDate").textContent, "2026-04-14", "the forecast response's exact underlying observation date is displayed");
  assert.equal(getNode("#forecastFreshness").textContent, "Stale · 7 month(s)", "the result displays freshness from forecast-origin metadata");
  assert.equal(getNode("#forecastOriginGenerated").textContent, "Origin 2026-04 · Generated 2026-11-19");
  assert.equal(getNode("#forecastSource").textContent, "WFP");
  assert.match(getNode("#forecastSelectedModel").textContent, /random forest/);
  assert.equal(getNode("#forecastSelectedHorizon").textContent, "6 month(s)");
  assert.equal(getNode("#forecastRmse").textContent, "4.00");
  assert.equal(getNode("#forecastMae").textContent, "3.00");
  assert.equal(getNode("#forecastMape").textContent, "1.00%");
  assert.equal(getNode("#forecastR2").textContent, "0.800");
  assert.equal(getNode("#forecastValidationCount").textContent, "12");
  assert.match(getNode("#forecastMethodDetails").textContent, /Forecast target: original WFP actual price/);
  assert.doesNotMatch(getNode("#forecastMethodDetails").textContent, /Generated 2026-11-19/);
  assert.deepEqual(charts.at(-1).config.data.datasets.map((dataset) => dataset.label), ["Historical WFP actual", "Future model estimate"]);

  const bagaMarket = market.options.find((option) => option.textContent.includes("Baga Road"));
  market.value = bagaMarket.value;
  await trigger("forecastMarket", "change");
  horizon.value = "2";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastResults").hidden, true, "a response containing a non-future month is not rendered");
  assert.match(getNode("#forecastEmpty h2").textContent, /not verifiably future-only/);
  assert.match(getNode("#forecastEmpty p").textContent, /strictly later than the generation month/);
  horizon.value = "3";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Future projection · aging source data");
  assert.match(getNode("#forecastStatusMessage").textContent, /not a current market forecast/);
  assert.match(getNode("#forecastStatusMessage").textContent, /7 month\(s\) old/);
  assert.match(getNode("#forecastTableBody").innerHTML, /Future at generation/);

  const currentMarket = market.options.find((option) => option.textContent.includes("Current Market"));
  market.value = currentMarket.value;
  await trigger("forecastMarket", "change");
  horizon.value = "1";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Future estimate · recent source data");
  assert.match(getNode("#forecastStatusMessage").textContent, /within the last three months/);
  assert.equal(getNode("#forecastFreshness").textContent, "Fresh · within 3 months");
  assert.match(getNode("#forecastSelectedModel").textContent, /^Baseline forecast: naive last observation$/);
  assert.deepEqual(charts.at(-1).config.data.datasets.map((dataset) => dataset.label), ["Historical WFP actual", "Future baseline estimate"]);

  for (const requested of ["2", "4", "5"]) {
    horizon.value = requested;
    await trigger("forecastHorizon", "change");
    await trigger("generateForecast", "click");
    const request = calls.filter((url) => url.pathname === "/api/forecast").at(-1);
    assert.equal(request.searchParams.get("horizon"), requested, "every supported horizon reaches the forecast API");
    assert.equal((getNode("#forecastTableBody").innerHTML.match(/<tr>/g) || []).length, Number(requested));
    assert.doesNotMatch(getNode("#forecastTableBody").innerHTML, /Past at generation/);
  }

  commodity.value = "Beans (red)";
  await trigger("forecastCommodity", "change");
  await trigger("generateForecast", "click");
  assert.equal(exactSeries.value, "beans-insufficient");
  assert.match(getNode("#forecastEmpty p").textContent, /17 gap-aware supervised rows are usable/);
  assert.match(getNode("#forecastEmpty h2").textContent, /insufficient usable history/);

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
  assert.equal(commodity.value, "Yam Tuber");
  assert.equal(market.options[0].textContent, "national: Nigeria");
  assert.equal(exactSeries.value, "", "the pre/post-rebase NBS observations require an explicit methodology choice");
  assert.ok(exactSeries.options.some((option) => option.textContent.includes("post 2025 cpi rebase")));
  assert.ok(exactSeries.options.some((option) => option.textContent.includes("pre 2025 cpi rebase")));
  assert.equal(getNode("#generateForecast").disabled, true);
  exactSeries.value = "nbs-yam-post";
  await trigger("forecastSeries", "change");
  assert.equal(getNode("#generateForecast").disabled, false, "reported NBS history is sent to exact-series eligibility validation");
  await trigger("generateForecast", "click");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("dataset"), "nbs");
  assert.equal(calls.filter((url) => url.pathname === "/api/forecast").at(-1).searchParams.get("series_id"), "nbs-yam-post");
  assert.match(getNode("#forecastEmpty h2").textContent, /insufficient usable history/);
  assert.match(getNode("#forecastEmpty p").textContent, /17 NBS observations/);

  console.log("Frontend workflow passed: independent selectors, exact WFP/NBS series routing, 1-6 month horizons, future-only rows, stale-data warnings, compact metrics, and source-specific insufficient-history response.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
