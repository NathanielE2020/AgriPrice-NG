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
    forecast_request_available: flag === "actual", forecast_available: false,
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
    nbs: [{ ...makeSeries("nbs-yam", "Yam Tuber", "National", "actual"), source_key: "nbs", source: "NBS", forecast_request_available: false }],
  });
  if (parsed.pathname === "/api/history") {
    const selected = records.find((record) => record.series_id === parsed.searchParams.get("series_id"));
    return apiResponse({ series: selected, history: [{ month: "2026-04", price: 500 }] });
  }
  if (parsed.pathname === "/api/performance") return apiResponse({ status: "not_evaluated", message: "No evaluation", records: [], predictions_by_date: [] });
  if (parsed.pathname === "/api/forecast") {
    if (parsed.searchParams.get("series_id") === "beans-insufficient") {
      return apiResponse({ error: { code: "insufficient_or_gapped_history", message: "Only 17 calendar rows are usable; insufficient history." } }, 422);
    }
    const selected = records.find((record) => record.series_id === parsed.searchParams.get("series_id"));
    const requestedHorizon = Number(parsed.searchParams.get("horizon"));
    const futureMonths = selected.series_id === "rice-current-fresh"
      ? ["2026-12", "2027-01", "2027-02", "2027-03", "2027-04", "2027-05"]
      : selected.series_id === "rice-actual-baga"
        ? ["2026-10", "2026-11", "2026-12", "2027-01", "2027-02", "2027-03"]
        : ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"];
    const classification = selected.series_id === "rice-current-fresh"
      ? "future_as_of_generation"
      : selected.series_id === "rice-actual-baga"
        ? "mixed_horizon_relative_to_generation"
        : "historical_projection_from_latest_actual";
    const actualMonth = selected.last_observed_month;
    const actualDate = selected.latest_observation_date;
    return apiResponse({
      series: selected, horizon: requestedHorizon, model: "random_forest",
      historical: [{ month: actualMonth, price: 500 }],
      forecast: futureMonths.slice(0, requestedHorizon).map((month, index) => ({ month, price: 510 + index, step: index + 1, temporal_status: month > "2026-11" ? "future_as_of_generation" : "after_origin_not_current" })),
      metadata: {
        method: "Recursive monthly forecast", selection_basis: "Validation RMSE", forecast_origin_month: actualMonth,
        latest_actual_date_in_source: actualDate, latest_actual_month_in_source: actualMonth, forecast_generation_date: "2026-11-19",
        forecast_classification: classification, transformation: "original WFP actual price", currency: "NGN", unit: "KG",
        validation_metrics: { rmse: 4, mae: 3, mape: 1, r2: 0.8, n: 12 }, currency: "NGN", unit: "KG",
      },
      warning: "Historical evidence only.",
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
  const forecastCallsBeforeHorizon = calls.filter((url) => url.pathname === "/api/forecast").length;
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
  assert.equal(exactSeries.options.length, 3, "same-market unit and price-type variants require an explicit choice");
  assert.equal(exactSeries.options[0].value, "", "no exact unit/type is silently selected when variants are ambiguous");
  assert.ok(exactSeries.options.every((option) => option.value !== "rice-aggregate"), "aggregate IDs cannot enter forecast options");
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
  assert.match(getNode("#forecastTableBody").innerHTML, /2026-05/);
  assert.equal((getNode("#forecastTableBody").innerHTML.match(/<tr>/g) || []).length, 6, "the six-month action renders six forecast dates");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Historical projection — not a current forecast");
  assert.equal(getNode("#forecastStatusMessage").textContent.includes("already in the past"), true);
  assert.equal(getNode("#forecastLatestDate").textContent, "2026-04-15", "the exact latest actual date remains visible");
  assert.equal(getNode("#forecastFreshness").textContent, "Aging · 7–12 months", "the result does not overwrite source freshness");
  assert.equal(getNode("#forecastOriginGenerated").textContent, "Origin 2026-04 · Generated 2026-11-19");
  assert.match(getNode("#forecastSelectedModel").textContent, /random forest/);
  assert.equal(getNode("#forecastSelectedHorizon").textContent, "6 month(s)");
  assert.equal(getNode("#forecastRmse").textContent, "4.00");
  assert.equal(getNode("#forecastMae").textContent, "3.00");
  assert.equal(getNode("#forecastMape").textContent, "1.00%");
  assert.equal(getNode("#forecastR2").textContent, "0.800");
  assert.equal(getNode("#forecastValidationCount").textContent, "12");
  assert.match(getNode("#forecastMethodDetails").textContent, /Forecast target: original WFP actual price/);
  assert.doesNotMatch(getNode("#forecastMethodDetails").textContent, /Generated 2026-11-19/);
  assert.deepEqual(charts.at(-1).config.data.datasets.map((dataset) => dataset.label), ["Historical WFP actual", "Recursive forecast"]);

  const bagaMarket = market.options.find((option) => option.textContent.includes("Baga Road"));
  market.value = bagaMarket.value;
  await trigger("forecastMarket", "change");
  horizon.value = "3";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Mixed-period projection");
  assert.match(getNode("#forecastStatusMessage").textContent, /Some requested dates are already in the past/);
  assert.match(getNode("#forecastStatusMessage").textContent, /only future rows based on recent actual data/);
  assert.match(getNode("#forecastStatusMessage").textContent, /7 month\(s\) old/);
  assert.match(getNode("#forecastTableBody").innerHTML, /Past at generation/);
  assert.match(getNode("#forecastTableBody").innerHTML, /Future at generation/);

  const currentMarket = market.options.find((option) => option.textContent.includes("Current Market"));
  market.value = currentMarket.value;
  await trigger("forecastMarket", "change");
  horizon.value = "1";
  await trigger("forecastHorizon", "change");
  await trigger("generateForecast", "click");
  assert.equal(getNode("#forecastStatusTitle").textContent, "Future forecast · recent actual data");
  assert.match(getNode("#forecastStatusMessage").textContent, /within the last three months/);
  assert.equal(getNode("#forecastFreshness").textContent, "Fresh · within 3 months");

  commodity.value = "Beans (red)";
  await trigger("forecastCommodity", "change");
  await trigger("generateForecast", "click");
  assert.equal(exactSeries.value, "beans-insufficient");
  assert.match(getNode("#forecastEmpty p").textContent, /17 calendar rows are usable/);
  assert.match(getNode("#forecastEmpty h2").textContent, /insufficient continuous history/);

  getNode("#globalDataset").value = "nbs";
  await trigger("globalDataset", "change");
  assert.equal(getNode("#forecastDataset").value, "wfp", "overview source changes must not alter forecast source");
  assert.equal(exactSeries.value, "beans-insufficient");
  getNode("#analysisDataset").value = "nbs";
  await trigger("analysisDataset", "change");
  assert.equal(getNode("#forecastDataset").value, "wfp", "historical source must remain independent from forecast source");
  getNode("#forecastDataset").value = "nbs";
  await trigger("forecastDataset", "change");
  assert.equal(commodity.disabled, true);
  assert.equal(getNode("#generateForecast").disabled, true);
  assert.match(getNode("#modelAvailabilityNote").textContent, /NBS national\/zone summaries/);

  console.log("Frontend workflow passed: selectors, exact series routing, independent history, explicit horizon action, historical/mixed/current freshness states, compact validation metrics, insufficient-history explanation, and NBS unavailability.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
