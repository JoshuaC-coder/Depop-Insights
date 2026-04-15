const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

chrome.runtime.onInstalled.addListener(async () => {
  const stored = await chrome.storage.sync.get(["apiBaseUrl"]);
  if (!stored.apiBaseUrl) {
    await chrome.storage.sync.set({ apiBaseUrl: DEFAULT_API_BASE_URL });
  }
});
