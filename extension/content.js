function cleanText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .replace(/\u00a0/g, " ")
    .trim();
}

function textFromSelectors(selectors) {
  for (const selector of selectors) {
    const nodes = document.querySelectorAll(selector);
    for (const node of nodes) {
      const text = cleanText(node.textContent);
      if (text) {
        return text;
      }
    }
  }
  return "";
}

function allTextsFromSelectors(selectors, maxItems = 10) {
  const seen = new Set();
  const values = [];
  for (const selector of selectors) {
    const nodes = document.querySelectorAll(selector);
    for (const node of nodes) {
      const text = cleanText(node.textContent);
      if (text && !seen.has(text)) {
        seen.add(text);
        values.push(text);
        if (values.length >= maxItems) return values;
      }
    }
  }
  return values;
}

function parsePrice(rawText) {
  const text = cleanText(rawText).replace(/,/g, "");
  const match = text.match(/(\d+(\.\d+)?)/);
  return match ? Number(match[1]) : null;
}

function normalizeBrand(rawBrand) {
  if (!rawBrand) return "";
  const cleaned = cleanText(rawBrand).replace(/^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$/g, "");
  if (!cleaned) return "";
  return cleaned
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function findLabeledValue(labels, options = {}) {
  const root = options.root || document.body;
  const maxLength = options.maxLength || 60;
  const bodyText = cleanText(root.innerText || "");
  for (const label of labels) {
    const regex = new RegExp(`${label}\\s*[:\\-]?\\s*([^\\n|•]+)`, "i");
    const match = bodyText.match(regex);
    if (match) {
      const candidate = cleanText(match[1]);
      if (candidate && candidate.length <= maxLength) {
        return candidate;
      }
    }
  }
  return "";
}

function detectBrandFromPage() {
  const directBrand = textFromSelectors([
    "[data-testid='listing-brand']",
    "[data-testid='brand']",
    "[href*='/brand/']",
    "[href*='/designer/']",
    "[class*='brand'] a",
    "[class*='Brand'] a",
    "[class*='brand']",
    "[class*='Brand']"
  ]);
  if (directBrand && directBrand.length <= 40) {
    return normalizeBrand(directBrand);
  }

  const labeledBrand = findLabeledValue(["brand", "designer", "label"]);
  return normalizeBrand(labeledBrand);
}

function detectBrand(title, description) {
  const pageBrand = detectBrandFromPage();
  if (pageBrand) return pageBrand;

  const combined = `${title} ${description}`.toLowerCase();
  const brands = [
    "nike",
    "adidas",
    "carhartt",
    "stussy",
    "supreme",
    "champion",
    "essentials",
    "fear of god",
    "lululemon",
    "aritzia",
    "patagonia",
    "the north face",
    "brandy melville",
    "levi's",
    "levis",
    "dickies",
    "gap",
    "hollister",
    "abercrombie",
    "zara",
    "uniqlo",
    "diesel",
    "ed hardy",
    "bape",
    "essentials",
    "juicy couture",
    "true religion"
  ];

  for (const brand of brands) {
    if (combined.includes(brand)) {
      return normalizeBrand(brand === "levis" ? "Levi's" : brand);
    }
  }
  return "";
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

function extractTitle() {
  return textFromSelectors([
    "[data-testid='listing-title']",
    "main h1",
    "h1",
    "[class*='title']",
    "[class*='Title']"
  ]);
}

function extractDescription() {
  const direct = textFromSelectors([
    "[data-testid='listing-description']",
    "[class*='description'] p",
    "[class*='Description'] p",
    "main section p"
  ]);
  if (direct) return direct;

  const paragraphs = allTextsFromSelectors(["main p", "article p", "section p"], 8)
    .filter((text) => text.length > 20 && !text.match(/^\$\d/));
  return paragraphs.join(" ").slice(0, 1200);
}

function extractPriceText() {
  const candidates = allTextsFromSelectors([
    "[data-testid='listing-price']",
    "[class*='price']",
    "[class*='Price']",
    "main span",
    "main div"
  ], 40);
  return candidates.find((text) => /[$£€]\s?\d|\d+(\.\d{2})?/.test(text)) || "";
}

function extractShippingText() {
  const direct = textFromSelectors([
    "[data-testid='shipping-price']",
    "[class*='shipping']",
    "[class*='Shipping']"
  ]);
  if (direct) return direct;
  return findLabeledValue(["shipping", "delivery", "postage"]);
}

function extractSize() {
  const direct = textFromSelectors([
    "[data-testid='listing-size']",
    "[class*='size']",
    "[class*='Size']"
  ]);
  if (direct && direct.length <= 30) return direct;
  return findLabeledValue(["size"]);
}

function extractCondition() {
  const direct = textFromSelectors([
    "[data-testid='listing-condition']",
    "[class*='condition']",
    "[class*='Condition']"
  ]);
  if (direct && direct.length <= 50) return direct;
  return findLabeledValue(["condition"]);
}

function extractColor() {
  const direct = textFromSelectors([
    "[data-testid='listing-color']",
    "[class*='colour']",
    "[class*='color']",
    "[class*='Colour']",
    "[class*='Color']"
  ]);
  if (direct && direct.length <= 40) return direct;
  return findLabeledValue(["color", "colour"]);
}

function extractLikesText() {
  const direct = textFromSelectors([
    "[data-testid='likes-count']",
    "[class*='likes']",
    "[class*='Likes']"
  ]);
  if (direct) return direct;
  return findLabeledValue(["likes", "saves", "favorites"], { maxLength: 12 });
}

function scrapeListing() {
  const title = extractTitle();
  const description = extractDescription();
  const priceText = extractPriceText();
  const shippingText = extractShippingText();
  const likesText = extractLikesText();
  const brand = detectBrand(title, description);
  const category = inferCategory(title, description);

  const listing = {
    title,
    description,
    price: parsePrice(priceText),
    shipping_price: parsePrice(shippingText) || 0,
    total_buyer_cost: (parsePrice(priceText) || 0) + (parsePrice(shippingText) || 0),
    brand,
    category,
    hoodie_type: inferItemType(title, description, category),
    size: extractSize(),
    condition: extractCondition(),
    color: extractColor(),
    likes: parsePrice(likesText),
    comments_count: null,
    seller_rating: null,
    seller_followers: null,
    discounted: /sale|discount|reduced/i.test(document.body.innerText),
    listed_date: null,
    sold_date: null,
    image_urls: collectImageUrls(),
    sold_status: /sold/i.test(document.body.innerText),
    source_url: window.location.href
  };

  const debug = {
    title,
    description_preview: description.slice(0, 180),
    priceText,
    shippingText,
    likesText,
    extracted_brand: brand,
    extracted_size: listing.size,
    extracted_condition: listing.condition,
    extracted_color: listing.color,
    category_guess: category
  };

  console.log("[Depop Insights] Scraped listing payload", listing);
  console.log("[Depop Insights] Scrape debug", debug);
  return { listing, debug };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "SCRAPE_LISTING") {
    const { listing, debug } = scrapeListing();
    sendResponse({ ok: true, listing, debug });
  }
  return true;
});
