#!/usr/bin/env python3
"""
Build a ChromaDB collection from an xlsx file using BAAI/bge-large-zh-v1.5.

Assumptions for the given workbook:
- Row 1: English header names
- Row 2: Description row (must be skipped)
- Row 3+: Data rows
"""

import argparse
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

NS = {"m": MAIN_NS, "r": REL_NS, "pr": PKG_REL_NS}


@dataclass
class ChunkRecord:
    row_number: int
    chunk_id: str
    text: str
    metadata: Dict[str, object]


def col_letters_to_index(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx


def read_cell_value(cell: ET.Element, shared_strings: List[str]) -> Optional[str]:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("m:v", NS)

    if cell_type == "s" and value_node is not None and value_node.text is not None:
        s_idx = int(value_node.text)
        if 0 <= s_idx < len(shared_strings):
            return shared_strings[s_idx]
        return None

    if cell_type == "inlineStr":
        t_node = cell.find("m:is/m:t", NS)
        return t_node.text if t_node is not None else None

    if value_node is not None:
        return value_node.text

    return None


def parse_shared_strings(xlsx: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []

    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    shared: List[str] = []
    for si in root.findall("m:si", NS):
        text_nodes = si.findall(".//m:t", NS)
        shared.append("".join((node.text or "") for node in text_nodes))
    return shared


def resolve_sheet_path(
    xlsx: zipfile.ZipFile, sheet_name: Optional[str] = None
) -> Tuple[str, str]:
    workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))

    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pr:Relationship", NS)
    }

    sheets = workbook.findall("m:sheets/m:sheet", NS)
    if not sheets:
        raise ValueError("No sheet found in workbook.")

    selected = None
    if sheet_name:
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                selected = sheet
                break
        if selected is None:
            available = ", ".join(s.attrib.get("name", "") for s in sheets)
            raise ValueError(
                f"Sheet '{sheet_name}' not found. Available sheets: {available}"
            )
    else:
        selected = sheets[0]

    selected_name = selected.attrib.get("name", "Sheet1")
    rid = selected.attrib.get(f"{{{REL_NS}}}id")
    if rid not in rel_map:
        raise ValueError(f"Cannot resolve sheet relationship id: {rid}")

    target = rel_map[rid]
    target = target.lstrip("/")
    if not target.startswith("xl/"):
        if target.startswith("worksheets/"):
            target = f"xl/{target}"
        else:
            target = f"xl/worksheets/{target.split('/')[-1]}"

    return selected_name, target


def parse_sheet_rows(
    xlsx: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]
) -> Dict[int, Dict[int, Optional[str]]]:
    root = ET.fromstring(xlsx.read(sheet_path))
    sheet_data = root.find("m:sheetData", NS)
    if sheet_data is None:
        return {}

    rows: Dict[int, Dict[int, Optional[str]]] = {}
    for row in sheet_data.findall("m:row", NS):
        row_number = int(row.attrib.get("r", "0"))
        parsed_row: Dict[int, Optional[str]] = {}
        for cell in row.findall("m:c", NS):
            ref = cell.attrib.get("r", "")
            m = re.match(r"([A-Z]+)(\d+)$", ref)
            if not m:
                continue
            col_idx = col_letters_to_index(m.group(1))
            parsed_row[col_idx] = read_cell_value(cell, shared_strings)
        rows[row_number] = parsed_row
    return rows


def as_int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(float(stripped))
    except ValueError:
        return None


def parse_page_range(raw: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if raw is None:
        return None, None
    text = raw.strip()
    if not text:
        return None, None
    parts = [p.strip() for p in text.split("/", 1)]
    if len(parts) == 1:
        page = as_int_or_none(parts[0])
        return page, page
    return as_int_or_none(parts[0]), as_int_or_none(parts[1])


def load_records_from_excel(excel_path: str, sheet_name: Optional[str]) -> List[ChunkRecord]:
    with zipfile.ZipFile(excel_path) as xlsx:
        shared_strings = parse_shared_strings(xlsx)
        actual_sheet_name, sheet_path = resolve_sheet_path(xlsx, sheet_name=sheet_name)
        rows = parse_sheet_rows(xlsx, sheet_path, shared_strings)

    if 1 not in rows:
        raise ValueError("Header row (row 1) is missing.")

    header_row = rows[1]
    if not header_row:
        raise ValueError("Header row is empty.")

    max_col = max(header_row.keys())
    headers: Dict[int, str] = {}
    for col_idx in range(1, max_col + 1):
        header = header_row.get(col_idx)
        if header is None or not str(header).strip():
            headers[col_idx] = f"col_{col_idx}"
        else:
            headers[col_idx] = str(header).strip()

    records: List[ChunkRecord] = []
    for row_num in sorted(rows.keys()):
        if row_num <= 2:
            # Skip header row (1) and description row (2).
            continue

        row_cells = rows[row_num]
        mapped = {headers[col]: row_cells.get(col) for col in headers}

        chunk_id = (mapped.get("chunk_id") or "").strip()
        text = (mapped.get("text") or "").strip()
        if not chunk_id or not text:
            continue

        page_start, page_end = parse_page_range(mapped.get("page_start/page_end"))
        para_idx = as_int_or_none(mapped.get("para_idx"))
        len_chars = as_int_or_none(mapped.get("len_chars"))

        metadata: Dict[str, object] = {
            "doc_id": (mapped.get("doc_id") or "").strip(),
            "doc_name": (mapped.get("doc_name") or "").strip(),
            "section_path": (mapped.get("section_path") or "").strip(),
            "page_start": page_start,
            "page_end": page_end,
            "para_idx": para_idx,
            "len_chars": len_chars,
            "sheet_name": actual_sheet_name,
            "source_row": row_num,
        }

        # Chroma metadata values must be scalar and not None.
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

        records.append(
            ChunkRecord(
                row_number=row_num,
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
            )
        )

    return records


def batch_iter(items: List[ChunkRecord], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_chromadb(
    excel_path: str,
    sheet_name: Optional[str],
    persist_dir: str,
    collection_name: str,
    model_name: str,
    batch_size: int,
    device: str,
    recreate: bool,
) -> None:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: pip install -r requirements.txt"
        ) from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    records = load_records_from_excel(excel_path=excel_path, sheet_name=sheet_name)
    if not records:
        raise ValueError("No valid records found in excel.")

    ids = [rec.chunk_id for rec in records]
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count > 0:
        raise ValueError(f"Found duplicate chunk_id count: {duplicate_count}")

    print(f"Loaded records: {len(records)}")
    print(f"Embedding model: {model_name} (device={device})")

    model = SentenceTransformer(model_name, device=device)
    client = chromadb.PersistentClient(path=persist_dir)

    if recreate:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    for idx, batch in enumerate(batch_iter(records, batch_size), start=1):
        batch_ids = [r.chunk_id for r in batch]
        batch_docs = [r.text for r in batch]
        batch_metas = [r.metadata for r in batch]
        batch_embeddings = model.encode(
            batch_docs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # upsert makes rerun idempotent for same chunk_id.
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embeddings,
        )
        print(f"Upserted batch {idx}: {len(batch)} records")

    total = collection.count()
    print(f"Done. Collection '{collection_name}' now has {total} records.")
    print(f"Persist directory: {persist_dir}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB collection from an xlsx source file."
    )
    parser.add_argument("--excel-path", default="整理后数据库.xlsx", help="Path to xlsx file")
    parser.add_argument("--sheet-name", default="Sheet1", help="Sheet name")
    parser.add_argument(
        "--persist-dir", default="./chroma_db", help="Chroma persistent directory"
    )
    parser.add_argument(
        "--collection-name", default="regulation_chunks_v1", help="Chroma collection name"
    )
    parser.add_argument(
        "--model-name", default="BAAI/bge-large-zh-v1.5", help="Embedding model name"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--device", default="cpu", help="Embedding device, e.g. cpu/cuda/mps")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection before writing",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    build_chromadb(
        excel_path=args.excel_path,
        sheet_name=args.sheet_name,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
