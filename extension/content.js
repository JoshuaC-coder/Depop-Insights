function textFromSelectors(selectors) {
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    if (node && node.textContent) {
      return node.textContent.trim();
    }
  }
  return "";
}

function parsePrice(rawText) {
  const match = rawText.replace(/,/g, "").match(/(\d+(\.\d+)?)/);
  return match ? Number(match[1]) : null;
}

function detectBrand(title, description) {
  const combined = `${title} ${description}`.toLowerCase();
  const brands = ["nike", "adidas", "carhartt", "stussy", "supreme", "champion", "essentials", "gap", "hollister", "abercrombie"];
  return brands.find((brand) => combined.includes(brand)) || "";
}

function inferItemType(title, description, categoryGuess) {
  const combined = `${title} ${description}`.toLowerCase();
  if (combined.includes("zip")) return "zip-up";
  if (combined.includes("crew")) return "crewneck";
  if (combined.includes("oversized")) return "oversized sweatshirt";
  if (combined.includes("hoodie")) return "hoodie";
  return categoryGuess || "general";
}

function inferCategory(title, description) {
  const combined = `${title} ${description}`.toLowerCase();
  if (/hoodie|crewneck|crew neck|zip[- ]?up|sweatshirt/.test(combined)) return "hoodies";
  if (/jacket|coat|outerwear/.test(combined)) return "jackets";
  if (/jean|denim/.test(combined)) return "jeans";
  if (/dress/.test(combined)) return "dresses";
  if (/shoe|sneaker|boot|loafer|sandal/.test(combined)) return "shoes";
  if (/shirt|tee|t-shirt|top|blouse/.test(combined)) return "tops";
  if (/pant|trouser|jogger|sweatpant/.test(combined)) return "bottoms";
  if (/bag|purse|tote|wallet/.test(combined)) return "bags";
  return "general";
}

function collectImageUrls() {
  return Array.from(document.images)
    .map((img) => img.src)
    .filter((src) => src && src.startsWith("http"))
    .slice(0, 6);
}

function scrapeListing() {
  const title = textFromSelectors(["h1", "[data-testid='listing-title']", "[class*='Title']"]);
  const description = textFromSelectors([
    "[data-testid='listing-description']",
    "[class*='description']",
    "section p"
  ]);
  const priceText = textFromSelectors(["[data-testid='listing-price']", "[class*='price']", "main span"]);
  const shippingText = textFromSelectors(["[data-testid='shipping-price']", "[class*='shipping']"]);
  const likesText = textFromSelectors(["[data-testid='likes-count']", "[class*='likes']"]);
  const sellerText = textFromSelectors(["[data-testid='shop-name']", "[class*='seller']"]);

  const listing = {
    title,
    description,
    price: parsePrice(priceText),
    shipping_price: parsePrice(shippingText) || 0,
    total_buyer_cost: (parsePrice(priceText) || 0) + (parsePrice(shippingText) || 0),
    brand: detectBrand(title, description),
    category: inferCategory(title, description),
    hoodie_type: inferItemType(title, description, inferCategory(title, description)),
    size: textFromSelectors(["[data-testid='listing-size']", "[class*='size']"]),
    condition: textFromSelectors(["[data-testid='listing-condition']", "[class*='condition']"]),
    color: textFromSelectors(["[data-testid='listing-color']", "[class*='colour']", "[class*='color']"]),
    likes: parsePrice(likesText),
    comments_count: 0,
    seller_rating: null,
    seller_followers: null,
    discounted: /sale|discount|reduced/i.test(document.body.innerText),
    listed_date: null,
    sold_date: null,
    image_urls: collectImageUrls(),
    sold_status: /sold/i.test(document.body.innerText),
    source_url: window.location.href,
    seller_name: sellerText
  };
  return listing;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "SCRAPE_LISTING") {
    sendResponse({ ok: true, listing: scrapeListing() });
  }
  return true;
});
