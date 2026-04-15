async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function getApiBaseUrl() {
  const stored = await chrome.storage.sync.get(["apiBaseUrl"]);
  return stored.apiBaseUrl || "http://127.0.0.1:8000";
}

function renderList(elementId, items) {
  const target = document.getElementById(elementId);
  target.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function formatModelBadge(modelStatus) {
  if (modelStatus === "hybrid_price_model") {
    return "Price Model Live";
  }
  if (modelStatus === "trained_model") {
    return "Full Model Live";
  }
  return "Heuristic Mode";
}

function buildPricingSummary(payload) {
  const low = payload.suggested_price_range.low;
  const high = payload.suggested_price_range.high;
  if (payload.model_status === "hybrid_price_model" || payload.model_status === "trained_model") {
    return `The strongest signal here is pricing: the trained model currently likes this listing around $${low}-$${high}. Sell-speed fields are still softer estimates.`;
  }
  return `This listing is currently using fallback estimates. Treat the suggested range of $${low}-$${high} as a rough starting point, not a trained recommendation.`;
}

function buildPricingRead(payload) {
  const spread = payload.suggested_price_range.high - payload.suggested_price_range.low;
  if (spread <= 8) {
    return "Tight range";
  }
  if (spread <= 16) {
    return "Balanced";
  }
  return "Wide range";
}

function buildInsightBanner(payload) {
  if (payload.model_status === "hybrid_price_model" || payload.model_status === "trained_model") {
    return "Use the price range as the anchor. Treat sell-speed as directional unless we add outcome-labeled marketplace data.";
  }
  return "This analysis is still in fallback mode, so use it as a directional read instead of a final pricing decision.";
}

function renderResults(payload) {
  document.getElementById("results").classList.remove("hidden");
  document.getElementById("fastSellScore").textContent = `${payload.fast_sell_score}/100`;
  document.getElementById("modelBadge").textContent = formatModelBadge(payload.model_status);
  document.getElementById("probabilityText").textContent =
    `${Math.round(payload.sold_within_30_days_probability * 100)}% estimated chance of selling within 30 days`;
  document.getElementById("daysToSell").textContent =
    `${payload.estimated_days_to_sell.low}-${payload.estimated_days_to_sell.high} days`;
  document.getElementById("priceRange").textContent =
    `$${payload.suggested_price_range.low}-$${payload.suggested_price_range.high}`;
  document.getElementById("pricingRead").textContent = buildPricingRead(payload);
  document.getElementById("pricingSummary").textContent = buildPricingSummary(payload);
  document.getElementById("insightBannerText").textContent = buildInsightBanner(payload);
  document.getElementById("scoreFill").style.width = `${Math.max(6, Math.min(100, payload.fast_sell_score))}%`;

  renderList(
    "factors",
    (payload.top_factors || []).map((factor) => `${factor.label}: ${factor.detail}`)
  );
  renderList("strengths", payload.top_strengths);
  renderList("weaknesses", payload.top_weaknesses);
  renderList("suggestions", payload.actionable_recommendations);
  renderList("notes", [
    `Status: ${payload.model_status}`,
    `Source: ${payload.model_source}`,
    ...(payload.notes || [])
  ]);
}

async function saveApiBaseUrl() {
  const input = document.getElementById("apiBaseUrl");
  await chrome.storage.sync.set({ apiBaseUrl: input.value.trim() });
}

async function requestListingFromTab(tabId) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "SCRAPE_LISTING" });
  } catch (error) {
    throw new Error(
      "Could not reach the page scraper. Reload the Depop tab after reloading the extension, then try again."
    );
  }
}

async function analyzeCurrentListing() {
  const status = document.getElementById("statusMessage");
  status.textContent = "Scraping listing details from the current page...";
  await saveApiBaseUrl();

  const tab = await getActiveTab();
  if (!tab?.id || !tab.url?.includes("depop.com")) {
    status.textContent = "Open a Depop listing page before running analysis.";
    return;
  }

  let response;
  try {
    response = await requestListingFromTab(tab.id);
  } catch (error) {
    status.textContent = error.message;
    return;
  }

  if (!response?.ok) {
    status.textContent = "Could not scrape the current page. Reload the tab and try again.";
    return;
  }

  const apiBaseUrl = await getApiBaseUrl();
  status.textContent = "Calling the backend for pricing and sell-speed analysis...";

  try {
    const result = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(response.listing)
    });
    if (!result.ok) {
      throw new Error(`Backend returned ${result.status}`);
    }

    const payload = await result.json();
    renderResults(payload);
    status.textContent = "Analysis complete.";
  } catch (error) {
    status.textContent = `Backend request failed: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("apiBaseUrl").value = await getApiBaseUrl();
  document.getElementById("analyzeButton").addEventListener("click", analyzeCurrentListing);
});
