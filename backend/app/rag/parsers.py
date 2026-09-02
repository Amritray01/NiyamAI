"""
Parser Modules for FSSAI PDFs (Footnotes, Highlights, Tables, Clauses)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import pymupdf as fitz  # PyMuPDF
from bs4 import BeautifulSoup



class FootnoteRegistry:
    """Resolves superscript footnote references (<sup>11</sup>) to footnote text."""

    def __init__(self) -> None:
        self._registry: Dict[str, str] = {}

    def ingest_html(self, html: str) -> None:
        """Scan HTML for inline footnote definitions."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["p", "td", "li", "div"]):
            sup = tag.find("sup")
            if sup and re.fullmatch(r"\d+", sup.get_text(strip=True)):
                key = sup.get_text(strip=True)
                sup.decompose()
                body = tag.get_text(separator=" ", strip=True)
                if body and key not in self._registry:
                    self._registry[key] = body

    def resolve(self, sup_id: str) -> Optional[str]:
        return self._registry.get(sup_id)

    def resolve_all_in_html(self, html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        resolved = []
        seen = set()
        for sup in soup.find_all("sup"):
            key = sup.get_text(strip=True)
            if key.isdigit() and key not in seen:
                note = self.resolve(key)
                if note:
                    resolved.append(f"[{key}] {note}")
                    seen.add(key)
        return resolved


class HighlightAnnotationExtractor:
    """Extracts text from PDF Highlight annotations using PyMuPDF."""
    EPSILON: float = 2.0

    def extract_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            annots = page.annots()
            if not annots:
                return []
            words = page.get_text("words")
            for annot in annots:
                if annot.type[0] != 8:
                    continue
                r = annot.rect
                expanded = fitz.Rect(
                    r.x0 - self.EPSILON,
                    r.y0 - self.EPSILON,
                    r.x1 + self.EPSILON,
                    r.y1 + self.EPSILON,
                )
                captured = [
                    w[4] for w in words
                    if expanded.intersects(fitz.Rect(w[0], w[1], w[2], w[3]))
                ]
                if captured:
                    results.append({
                        "page": page_num,
                        "text": " ".join(captured),
                        "rect": [r.x0, r.y0, r.x1, r.y1],
                    })
        except Exception:
            pass
        return results

    def extract(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        if not os.path.exists(pdf_path):
            return []
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            return []
        page = doc.load_page(page_num)
        res = self.extract_from_page(page, page_num + 1)
        doc.close()
        return res


class DoclingFSSAIParser:
    """Robust document parser for FSSAI PDFs using PyMuPDF & BeautifulSoup."""

    def __init__(self) -> None:
        self._footnote_registry = FootnoteRegistry()
        self._highlight_extractor = HighlightAnnotationExtractor()

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parses a PDF into structured elements:
        - tables: HTML table representations
        - clauses: split numbered regulation sections & contextual page blocks
        - footnote_registry: resolved footnotes
        - highlights: PDF highlight annotations
        """
        doc = fitz.open(pdf_path)
        tables: List[Dict[str, Any]] = []
        clauses: List[Dict[str, Any]] = []
        highlights: List[Dict[str, Any]] = []

        full_text_by_page: List[Tuple[int, str]] = []
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            page_num = page_idx + 1
            text = page.get_text("text")
            full_text_by_page.append((page_num, text))

            # Extract table structures if present via PyMuPDF table finder
            tabs = page.find_tables()
            for t_idx, tab in enumerate(tabs):
                df = tab.extract()
                if not df or len(df) < 2:
                    continue
                # Build HTML representation
                header_row = df[0]
                body_rows = df[1:]
                html_parts = ["<table><thead><tr>"]
                for h in header_row:
                    html_parts.append(f"<th>{h or ''}</th>")
                html_parts.append("</tr></thead><tbody>")
                for r in body_rows:
                    html_parts.append("<tr>")
                    for cell in r:
                        html_parts.append(f"<td>{cell or ''}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</tbody></table>")
                table_html = "".join(html_parts)
                self._footnote_registry.ingest_html(table_html)
                tables.append({
                    "html": table_html,
                    "page": page_num,
                    "table_idx": t_idx
                })

            # Extract highlights directly from open page
            highlights.extend(self._highlight_extractor.extract_from_page(page, page_num))

        doc.close()

        # Extract structured clauses and continuous section blocks
        clauses = self._extract_clauses(full_text_by_page, os.path.basename(pdf_path))

        return {
            "tables": tables,
            "clauses": clauses,
            "footnote_registry": self._footnote_registry,
            "highlights": highlights,
            "total_pages": len(full_text_by_page),
        }

    @staticmethod
    def _is_regulation_clause_header(line: str) -> Optional[Tuple[str, str]]:
        """
        Distinguishes real statutory clause headers from inline decimal measurements.
        Valid: '2.1 : METAL CONTAMINANTS', '2.2.1', '2.8: Sweetening agents', '1. Lead', '3. Food business operator'
        Invalid: '5.0 on dry matter', '0.05 (on fat basis)', '10.0 on the dry'
        """
        line_clean = line.strip()
        if not line_clean:
            return None

        # Ignore obvious numeric measurement / table line patterns
        if re.search(r"\b(on\s+(the\s+)?(dry|fat|whole|shell)|ppm|mg/kg|ppb|percent|per cent|\(on\s+|\(carcass|\(fat|\(whole|\(shell)\b", line_clean, re.I):
            return None

        # Pattern A: Chapter/Section/Regulation X.Y or X.Y.Z
        m_reg = re.match(r"^(\d+(?:\.\d+)+)[:\s.-]+([A-Za-z].*)", line_clean)
        if m_reg:
            clause_id = m_reg.group(1).strip()
            title = m_reg.group(2).strip()
            # Verify clause_id is not a decimal like 0.05 or 10.0 with short trailing text
            if len(clause_id.split(".")) == 2 and clause_id.split(".")[1] in ("0", "00", "05", "10", "15", "20", "25", "50"):
                if not re.match(r"^[A-Z\s]{4,}|^(METAL|CROP|RESTRICTION|RESIDUE|SWEETENING|PROHIBITION|FOOD|DEFINITIONS|NUTRITION)", title, re.I):
                    return None
            return clause_id, title

        # Pattern B: Specific major numbered sections in regulations (e.g. '1. Lead', '2. Copper', '3. Arsenic', '1. Short title', '4. Nutrition claims')
        m_sec = re.match(r"^(\d+)\.\s+([A-Z][A-Za-z\s,\-–/()]{2,})", line_clean)
        if m_sec:
            sec_num = m_sec.group(1).strip()
            sec_title = m_sec.group(2).strip()
            return f"Section_{sec_num}", sec_title

        # Pattern C: CHAPTER headers
        m_chap = re.match(r"^(CHAPTER\s+\d+|PART\s+[I|V|X]+)[:\s.-]*(.*)", line_clean, re.I)
        if m_chap:
            return m_chap.group(1).strip(), m_chap.group(2).strip()

        return None

    def _extract_clauses(self, full_text_by_page: List[Tuple[int, str]], filename: str = "") -> List[Dict[str, Any]]:
        """
        Splits document text into numbered regulation clauses and coherent page/section blocks.
        Preserves contextual headers so that table contents (e.g., Lead limits) are not lost.
        """
        clauses: List[Dict[str, Any]] = []

        current_clause_id = "General"
        current_title = ""
        current_page = 1
        current_lines: List[str] = []

        for page_num, text in full_text_by_page:
            lines = text.split("\n")

            for line in lines:
                header_match = self._is_regulation_clause_header(line)
                if header_match:
                    new_id, new_title = header_match
                    body = "\n".join(current_lines).strip()
                    if body and len(body) >= 35:
                        clauses.append({
                            "clause_id": current_clause_id,
                            "title": current_title,
                            "text": body,
                            "page": current_page
                        })
                    current_clause_id = new_id
                    current_title = new_title
                    current_page = page_num
                    current_lines = [line]
                else:
                    if line.strip():
                        current_lines.append(line)

            # Preserve page-level structured context for rich table pages
            page_text_clean = text.strip()
            if len(page_text_clean) > 80:
                doc_title = filename.replace(".pdf", "").replace("_", " ")
                enhanced_page_text = f"[{doc_title} | Page {page_num}]\n{page_text_clean}"
                clauses.append({
                    "clause_id": f"Page_{page_num}",
                    "title": f"{doc_title} - Page {page_num}",
                    "text": enhanced_page_text,
                    "page": page_num
                })

        # Flush final clause
        body = "\n".join(current_lines).strip()
        if body and len(body) >= 35:
            clauses.append({
                "clause_id": current_clause_id,
                "title": current_title,
                "text": body,
                "page": current_page
            })

        return clauses
