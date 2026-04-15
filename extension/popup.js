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

function bestItems(items, limit = 2, fallback = []) {
  const source = items && items.length ? items : fallback;
  return source.slice(0, limit);
}

function setModeLabel(payload) {
  const modeNode = document.getElementById("modeLabel");
  if (payload.model_status === "heuristic_fallback") {
    modeNode.textContent = "Heuristic mode";
  } else if (payload.model_status === "hybrid_price_model") {
    modeNode.textContent = "Hybrid mode";
  } else {
    modeNode.textContent = "Model mode";
  }
}

function renderResults(payload) {
  document.getElementById("results").classList.remove("hidden");
  document.getElementById("fastSellScore").textContent = `${payload.fast_sell_score}/100`;
  document.getElementById("probabilityText").textContent =
    `${Math.round(payload.sold_within_30_days_probability * 100)}% est. sell chance`;
  document.getElementById("daysToSell").textContent =
    `${payload.estimated_days_to_sell.low}-${payload.estimated_days_to_sell.high} days`;
  document.getElementById("priceRange").textContent =
    `$${payload.suggested_price_range.low}-$${payload.suggested_price_range.high}`;
  document.getElementById("topSuggestion").textContent =
    bestItems(payload.actionable_recommendations, 1, ["Stay inside the suggested price range"])[0];
  document.getElementById("scoreFill").style.width = `${Math.max(6, Math.min(100, payload.fast_sell_score))}%`;
  setModeLabel(payload);

  renderList(
    "strengths",
    bestItems(payload.top_strengths, 2, ["pricing looks aligned with the model range"])
  );
  renderList(
    "weaknesses",
    bestItems(payload.top_weaknesses, 2, ["no major watchouts surfaced from the current read"])
  );
}

async function saveApiBaseUrl() {
  const input = document.getElementById("apiBaseUrl");
  await chrome.storage.sync.set({ apiBaseUrl: input.value.trim() });
}

async function requestListingFromTab(tabId) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "SCRAPE_LISTING" });
  } catch (_error) {
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
  const payloadToSend = response.listing;
  console.log("[Depop Insights] JSON sent to /predict", payloadToSend);
  status.textContent = "Calling the backend for pricing and sell-speed analysis...";

  try {
    const result = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadToSend)
    });
    if (!result.ok) {
      throw new Error(`Backend returned ${result.status}`);
    }

    const payload = await result.json();
    console.log("[Depop Insights] Backend /predict response", payload);
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
