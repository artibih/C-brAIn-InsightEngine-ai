## Create and backup (tabular) a knowledge base from JSON files

If you have JSON files with the content of papers in it, you can process them and create the vector DB with them.
After that, you can back it up into .parquet files. Here is the step by step process:

1. Put JSON files under `data/input` (or set `folder_path` in `batch_process_documents.main()`).
2. Start Weaviate (e.g. `docker compose up -d weaviate`).
3. Run `uv run python3 src/vector_db/backup_modules/batch_process_documents.py` (this will ingest the JSON files into the Weaviate vector db)
4. Run `uv run python3 src/vector_db/backup_modules/backup_vector_db_to_tabular.py` (this will take the just-created Weaviate vector db an back it up using .parquet files)

## Restore a pre-existing tabular backup into the vector DB

If you have a back up ready (.parquet files), then you can re-create the vector db from that. 
Here is the step by step:

1. Start Weaviate (e.g. `docker compose up -d weaviate`).
2. Set `backup_dir` in `restore_from_tabular.main()` to your backup dir (where the .parquets are located).
3. Set `skip_checkpoint = True` in `restore_from_tabular.main()` to skip checkpoint DB (optional).
4. Run `uv run python3 src/vector_db/backup_modules/restore_from_tabular.py` (this will read the .parquet files and re-create the vector db)

## Helper scripts

- **List collections and object counts:** `uv run python3 src/vector_db/backup_modules/weaviate_list_collections.py`
- **Wipe all Weaviate collections:** `uv run python3 src/vector_db/backup_modules/weaviate_reset.py` (dry run by default; set `confirm_delete = True` in `weaviate_reset.main()` to actually delete).
