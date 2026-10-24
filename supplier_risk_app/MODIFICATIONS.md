# Project integration changes

## What changed

- Added `supplier_data.py` to load and search the 1,000-row supplier CSV.
- Added `/api/suppliers` and `/api/supplier/<Supplier_ID>` endpoints.
- Updated `/analyze` so the LLM receives both:
  - the selected supplier's structured CSV profile;
  - retrieved RAG document evidence.
- Added explicit evidence/data-source instructions to the LLM prompt.
- Added risk score, risk level, procurement action, key risks, evidence, and data-source output fields.
- Updated the Assistant page to display the integrated result.
- Updated the Suppliers page to load the actual CSV rather than hard-coded suppliers.
- Updated Dashboard supplier counts to use the CSV.
- Added `PyMuPDF` and `python-dotenv` to `requirements.txt`.
- Added environment-variable configuration for `OPENAI_MODEL`.

## Flow

CSV -> supplier lookup -> supplier profile -> RAG retrieval -> LLM -> procurement decision

## Important

The existing 1,000-row CSV is the structured project dataset. It should not be described as live/current external supplier data unless you replace or augment it with a verified live data feed.
