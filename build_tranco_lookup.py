"""
build_tranco_lookup.py
-----------------------
Run this ONCE from inside the `api` folder, after placing top-1m.csv there.

What it does:
  1. Reads top-1m.csv (format: rank,domain — no header)
  2. Strips it down to just "domain,rank" pairs
  3. Sorts by domain name (alphabetically) so main.py can binary-search it instantly
  4. Writes tranco_lookup.txt

After this runs successfully, top-1m.csv is no longer needed by the running app —
only tranco_lookup.txt matters. You can delete the original csv/zip if you want,
or just leave top-1m.csv out of git (it's large and easy to re-generate).

Usage:
    cd C:\\Users\\MITHUN\\phishing-detection-extension\\api
    python build_tranco_lookup.py
"""

import csv
import sys
import os

INPUT_FILE = "top-1m.csv"
OUTPUT_FILE = "tranco_lookup.txt"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Could not find {INPUT_FILE} in the current folder.")
        print("   Make sure you're running this from inside the `api` folder,")
        print("   and that top-1m.csv has been downloaded and placed there.")
        sys.exit(1)

    print(f"Reading {INPUT_FILE} ...")
    entries = []

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue  # skip any malformed/blank lines
            rank_str, domain = row
            try:
                rank = int(rank_str)
            except ValueError:
                continue  # skip header or junk rows, if any sneak in
            entries.append((domain.strip().lower(), rank))

    print(f"Loaded {len(entries):,} domain entries.")

    print("Sorting by domain name...")
    entries.sort(key=lambda pair: pair[0])

    print(f"Writing {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for domain, rank in entries:
            f.write(f"{domain},{rank}\n")

    print(f"✅ Done. {OUTPUT_FILE} created with {len(entries):,} entries.")
    print("   main.py will load this file at startup for instant lookups.")
    print(f"   You can now delete {INPUT_FILE} and the original .zip if you like.")


if __name__ == "__main__":
    main()
