#!/usr/bin/env python3
"""Delete all Weaviate collections (and their data). Requires Weaviate running.
Set confirm_delete=True in main() to actually delete; otherwise dry run."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.backup_modules.weaviate_client import get_client


def main():
    confirm_delete = False  
    try:
        client = get_client()
    except Exception as e:
        print("Failed to connect to Weaviate: %s. Is it running?" % e, file=sys.stderr)
        sys.exit(1)
    try:
        collections = client.collections.list_all()
        names = sorted(collections.keys())
        if not names:
            print("No collections. Weaviate is already empty.")
            return
        print("Collections:", names)
        if not confirm_delete:
            print("Dry run. Set confirm_delete=True in main() to delete.")
            return
        for name in names:
            client.collections.delete(name)
            print("Deleted:", name)
        print("All collections deleted.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
