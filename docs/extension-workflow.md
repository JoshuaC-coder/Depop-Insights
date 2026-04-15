# Extension Workflow

1. The user opens a Depop listing page.
2. The popup asks the content script to scrape visible listing data from the active tab.
3. The popup sends the extracted JSON to the FastAPI backend.
4. The backend validates the payload and returns:
   - model price range
   - sell-speed beta score
   - estimated days-to-sell range
   - suggested price range
   - top strengths
   - top weaknesses
   - actionable recommendations
5. The popup renders the response in a compact seller-facing view.

Current limitations:

- The page scraper uses resilient selector heuristics, but Depop DOM changes will require maintenance.
- Seller metrics such as followers and rating are only populated when they are available on-page.
- The extension is currently optimized for local backend development at `http://127.0.0.1:8000`.
- Sell-speed outputs are still heuristic when the loaded model bundle is price-only.
