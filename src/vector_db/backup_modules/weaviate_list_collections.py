#!/usr/bin/env python3
"""List Weaviate collections and their object counts. Requires Weaviate running."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.backup_modules.weaviate_client import get_client


def main():
    try:
        client = get_client()
    except Exception as e:
        print("Failed to connect to Weaviate: %s. Is it running?" % e, file=sys.stderr)
        sys.exit(1)
    try:
        collections = client.collections.list_all()
        if not collections:
            print("No collections. Weaviate is empty.")
            return
        print("Collections (%s):\n" % len(collections))
        total = 0
        unknown_count = 0
        for name in sorted(collections.keys()):
            coll = client.collections.get(name)
            try:
                count = coll.aggregate.over_all(total_count=True).total_count
            except Exception:
                count = None
            if count is not None:
                total += count
            else:
                unknown_count += 1
            print("  %s: %s" % (name, count if count is not None else "?"))
        total_str = "?" if unknown_count > 0 else str(total)
        print("\nTotal objects: %s" % total_str)
    finally:
        client.close()


if __name__ == "__main__":
    main()
