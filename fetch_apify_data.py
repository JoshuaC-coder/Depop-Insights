from __future__ import annotations

from ml.data_ingestion.apify_ingestion import fetch_and_store_apify_data


def main() -> None:
    raw_path, csv_path = fetch_and_store_apify_data()
    print(f"Saved raw Apify payload to {raw_path}")
    print(f"Saved normalized training CSV to {csv_path}")


if __name__ == "__main__":
    main()
