from __future__ import annotations

from typing import Iterable

from backend.app.schemas import Factor, ListingInput


TREND_KEYWORDS = {
    "vintage",
    "y2k",
    "oversized",
    "faded",
    "distressed",
    "rare",
    "streetwear",
    "heavyweight",
    "essential",
    "graphic",
}

CONDITION_TERMS = {"flaw", "flaws", "stain", "faded", "worn", "cracking", "excellent", "good"}


def _push_unique(target: list[str], items: Iterable[str]) -> None:
    for item in items:
        if item and item not in target:
            target.append(item)


def build_factors(listing: ListingInput, probability: float, expected_days: float) -> tuple[list[Factor], list[str], list[str], list[str]]:
    title = listing.title.lower()
    description = listing.description.lower()
    total_cost = listing.total_buyer_cost or (listing.price or 0) + (listing.shipping_price or 0)
    title_keywords = sum(keyword in title for keyword in TREND_KEYWORDS)
    description_keywords = sum(keyword in description for keyword in TREND_KEYWORDS)

    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    factors: list[Factor] = []

    if listing.brand:
        strengths.append("brand is explicitly identified")
        factors.append(Factor(label="Brand clarity", impact="positive", detail=f"Listing names the brand as {listing.brand}."))
    else:
        weaknesses.append("brand is missing or unclear")
        suggestions.append("Add the brand name to the title and details so buyers can filter for it.")
        factors.append(Factor(label="Brand clarity", impact="negative", detail="Brand is missing, which weakens buyer trust and search relevance."))

    if title_keywords >= 2:
        strengths.append("title contains strong resale search keywords")
        factors.append(Factor(label="Keyword coverage", impact="positive", detail="Title includes multiple resale/trend keywords that improve discoverability."))
    else:
        weaknesses.append("title could use stronger search keywords")
        suggestions.append("Add accurate category and style keywords like vintage, oversized, graphic, or material details when relevant.")
        factors.append(Factor(label="Keyword coverage", impact="negative", detail="Title is likely under-optimized for common resale search patterns."))

    if len(listing.description.split()) >= 18 or description_keywords >= 2:
        strengths.append("description provides useful buyer context")
        factors.append(Factor(label="Description quality", impact="positive", detail="Description is detailed enough to reduce buyer uncertainty."))
    else:
        weaknesses.append("description is relatively thin")
        suggestions.append("Expand the description with fit, measurements, material weight, and condition details.")
        factors.append(Factor(label="Description quality", impact="negative", detail="Short descriptions make it harder to justify price and condition."))

    if listing.shipping_price is not None and listing.shipping_price > 10:
        weaknesses.append("shipping cost is on the high side")
        suggestions.append("Consider lowering shipping or bundling more value into the listing if possible.")
        factors.append(Factor(label="Shipping cost", impact="negative", detail=f"Shipping is ${listing.shipping_price:.2f}, which may suppress conversion."))
    elif listing.shipping_price is not None:
        strengths.append("shipping cost looks reasonable")
        factors.append(Factor(label="Shipping cost", impact="positive", detail=f"Shipping is ${listing.shipping_price:.2f}, which is not unusually high for this listing type."))

    if listing.likes is not None and listing.likes >= 10:
        strengths.append("listing already shows buyer interest")
        factors.append(Factor(label="Buyer interest", impact="positive", detail=f"Listing has {listing.likes} likes, signaling early market interest."))
    elif listing.likes is not None and listing.likes <= 2:
        weaknesses.append("listing has limited engagement so far")
        suggestions.append("Refresh the cover image and title to improve click-through and save rate.")
        factors.append(Factor(label="Buyer interest", impact="negative", detail=f"Listing currently has only {listing.likes} likes."))

    if total_cost > 80:
        weaknesses.append("total buyer cost may be above impulse-buy territory")
        suggestions.append("Test a slightly lower total price if you want to optimize for faster sell-through.")
        factors.append(Factor(label="Total buyer cost", impact="negative", detail=f"Estimated total buyer cost is ${total_cost:.2f}."))
    else:
        strengths.append("total buyer cost stays in a more approachable range")
        factors.append(Factor(label="Total buyer cost", impact="positive", detail=f"Estimated total buyer cost is ${total_cost:.2f}."))

    if any(term in description for term in CONDITION_TERMS):
        strengths.append("condition details are at least partially addressed")
    else:
        weaknesses.append("condition transparency could be stronger")
        suggestions.append("Call out any flaws, wear, fading, or excellent condition details explicitly.")

    if probability < 0.5:
        suggestions.append("Consider a more competitive price or a sharper title to improve the 30-day sell probability.")
    if expected_days > 30:
        suggestions.append("Revisit photos, title keywords, and pricing if fast sell-through is the goal.")

    return factors[:6], strengths[:4], weaknesses[:4], suggestions[:4]
