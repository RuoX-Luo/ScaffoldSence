# Build ChromaDB from Excel

This project provides a script to build a local ChromaDB collection from `整理后数据库.xlsx` using `BAAI/bge-large-zh-v1.5`.

## What it does

- Reads the xlsx file directly (without `openpyxl` or `pandas`)
- Uses row 1 as headers
- Skips row 2 (description row)
- Uses `chunk_id` as Chroma `id`
- Uses `text` as Chroma `document`
- Writes metadata:
  - `doc_id`
  - `doc_name`
  - `section_path`
  - `page_start`, `page_end` (parsed from `page_start/page_end`)
  - `para_idx`
  - `len_chars`
  - `sheet_name`
  - `source_row`
- Embedding model: `BAAI/bge-large-zh-v1.5`
- Embeddings are normalized for cosine similarity

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (CPU)

```bash
python3 build_chromadb_from_excel.py \
  --excel-path "整理后数据库.xlsx" \
  --sheet-name "Sheet1" \
  --persist-dir "./chroma_db" \
  --collection-name "regulation_chunks_v1" \
  --device "cpu" \
  --batch-size 16 \
  --recreate
```

## Notes

- `--recreate` will delete existing collection with the same name before writing.
- If you omit `--recreate`, script uses `upsert` and is idempotent on same `chunk_id`.
