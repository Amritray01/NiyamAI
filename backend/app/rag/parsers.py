from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
import pymupdf as fitz


class FootnoteRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, str] = {}

    def ingest_html(self, html: str) -> None:
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
        resolved: List[str] = []
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


try:
    import pymupdf4llm
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False


class DoclingFSSAIParser:
    def __init__(self) -> None:
        self._footnote_registry = FootnoteRegistry()
        self._highlight_extractor = HighlightAnnotationExtractor()

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        tables: List[Dict[str, Any]] = []
        highlights: List[Dict[str, Any]] = []
        full_text_by_page: List[Tuple[int, str]] = []
        last_known_headers: List[str] = ["Contaminant / Parameter", "Category / Class", "Article of Food", "Maximum Permitted Limit"]

        # Layout-Aware Markdown Extraction via PyMuPDF4LLM
        layout_md_by_page = {}
        if _HAS_PYMUPDF4LLM:
            try:
                for page_idx in range(len(doc)):
                    md_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_idx])
                    layout_md_by_page[page_idx + 1] = md_text
            except Exception:
                pass

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            page_num = page_idx + 1
            text = page.get_text("text")
            layout_text = layout_md_by_page.get(page_num, text)
            full_text_by_page.append((page_num, layout_text if len(layout_text) > len(text) else text))

            # 1. First attempt native PyMuPDF vector table detection
            found_tables = page.find_tables().tables if hasattr(page.find_tables(), "tables") else []
            if found_tables:
                for t_idx, tab in enumerate(found_tables):
                    df = tab.extract()
                    if not df or len(df) < 1:
                        continue

                    first_row = df[0]
                    has_header_keywords = any(
                        re.search(r"\b(name|article|food|parameter|limit|sampling|standard|category|class|method|value|tolerance)\b", str(cell), re.I)
                        for cell in first_row if cell
                    )

                    if has_header_keywords and len(df) >= 2:
                        header_row = [str(h).strip() if h else f"Col_{i+1}" for i, h in enumerate(first_row)]
                        body_rows = df[1:]
                        last_known_headers = header_row
                    elif last_known_headers and len(first_row) == len(last_known_headers):
                        header_row = last_known_headers
                        body_rows = df
                    else:
                        header_row = [str(h).strip() if h else f"Col_{i+1}" for i, h in enumerate(first_row)]
                        body_rows = df[1:] if len(df) > 1 else []
                        if has_header_keywords:
                            last_known_headers = header_row

                    html_parts = ["<table><thead><tr>"]
                    for h in header_row:
                        html_parts.append(f"<th>{h}</th>")
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
                        "table_idx": t_idx,
                    })
            else:
                # 2. Borderless Gazette Layout Parser (uses pymupdf4llm text & column regex)
                borderless_html = self._extract_borderless_table(layout_text, page_num, last_known_headers) or self._extract_borderless_table(text, page_num, last_known_headers)
                if borderless_html:
                    self._footnote_registry.ingest_html(borderless_html)
                    tables.append({
                        "html": borderless_html,
                        "page": page_num,
                        "table_idx": 0,
                    })

            highlights.extend(self._highlight_extractor.extract_from_page(page, page_num))

        doc.close()
        clauses = self._extract_clauses(full_text_by_page, os.path.basename(pdf_path))

        return {
            "tables": tables,
            "clauses": clauses,
            "footnote_registry": self._footnote_registry,
            "highlights": highlights,
            "total_pages": len(full_text_by_page),
        }

    def _extract_borderless_table(self, text: str, page_num: int, default_headers: List[str]) -> Optional[str]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        has_table_marker = any(
            re.search(r"\b(parts per million|ppm|mg/kg|ppb|µg/kg|article of food|name of food|tolerance limit|maximum limit)\b", l, re.I)
            for l in lines
        )
        if not has_table_marker:
            return None

        rows: List[Dict[str, str]] = []
        current_metal = "Contaminant / Standard"
        current_subcat = "General"
        current_food: List[str] = []

        skip_words = {
            "table", "name of the article of food", "parts per million", "by weight",
            "(1)", "(2)", "(3)", "(4)", "metal contaminants", "article of food"
        }

        for line in lines:
            if re.match(r"^\d+\.\s+([A-Za-z\s]+)$", line):
                current_metal = line.strip()
                continue
            if re.match(r"^\([i|v|x|a-z]+\)\s+(.*)", line, re.I):
                current_subcat = line.strip()
                continue

            m_num = re.match(r"^(\d+(?:\.\d+)?)(?:\s+(.*))?$", line)
            if m_num and (float(m_num.group(1)) < 10000 or "dry" in line.lower() or "percent" in line.lower()):
                limit_val = line.strip()
                food_text = " ".join(current_food).strip()
                if food_text and len(food_text) > 3:
                    rows.append({
                        "Parameter / Contaminant": current_metal,
                        "Category": current_subcat,
                        "Article of Food": food_text,
                        "Maximum Permitted Limit (ppm / mg/kg)": limit_val,
                    })
                current_food = []
            else:
                if line.lower() not in skip_words and not any(line.lower().startswith(sw) for sw in ["table", "name of the article", "(1)"]):
                    current_food.append(line)

        if not rows:
            return None

        headers = ["Parameter / Contaminant", "Category", "Article of Food", "Maximum Permitted Limit (ppm / mg/kg)"]
        html_parts = ["<table><thead><tr>"]
        for h in headers:
            html_parts.append(f"<th>{h}</th>")
        html_parts.append("</tr></thead><tbody>")
        for r in rows:
            html_parts.append("<tr>")
            for h in headers:
                html_parts.append(f"<td>{r.get(h, '')}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
        return "".join(html_parts)

    @staticmethod
    def _is_regulation_clause_header(line: str) -> Optional[Tuple[str, str]]:
        line_clean = line.strip()
        if not line_clean:
            return None

        if re.search(r"\b(on\s+(the\s+)?(dry|fat|whole|shell)|ppm|mg/kg|ppb|percent|per cent|\(on\s+|\(carcass|\(fat|\(whole|\(shell)\b", line_clean, re.I):
            return None

        m_reg = re.match(r"^(\d+(?:\.\d+)+)[:\s.-]+([A-Za-z].*)", line_clean)
        if m_reg:
            clause_id = m_reg.group(1).strip()
            title = m_reg.group(2).strip()
            if len(clause_id.split(".")) == 2 and clause_id.split(".")[1] in ("0", "00", "05", "10", "15", "20", "25", "50"):
                if not re.match(r"^[A-Z\s]{4,}|^(METAL|CROP|RESTRICTION|RESIDUE|SWEETENING|PROHIBITION|FOOD|DEFINITIONS|NUTRITION)", title, re.I):
                    return None
            return clause_id, title

        m_sec = re.match(r"^(\d+)\.\s+([A-Z][A-Za-z\s,\-–/()]{2,})", line_clean)
        if m_sec:
            return f"Section_{m_sec.group(1).strip()}", m_sec.group(2).strip()

        m_chap = re.match(r"^(CHAPTER\s+\d+|PART\s+[I|V|X]+)[:\s.-]*(.*)", line_clean, re.I)
        if m_chap:
            return m_chap.group(1).strip(), m_chap.group(2).strip()

        return None

    def _extract_clauses(self, full_text_by_page: List[Tuple[int, str]], filename: str = "") -> List[Dict[str, Any]]:
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
                            "page": current_page,
                        })
                    current_clause_id = new_id
                    current_title = new_title
                    current_page = page_num
                    current_lines = [line]
                else:
                    if line.strip():
                        current_lines.append(line)

            page_text_clean = text.strip()
            if len(page_text_clean) > 80:
                doc_title = filename.replace(".pdf", "").replace("_", " ")
                enhanced_page_text = f"[{doc_title} | Page {page_num}]\n{page_text_clean}"
                clauses.append({
                    "clause_id": f"Page_{page_num}",
                    "title": f"{doc_title} - Page {page_num}",
                    "text": enhanced_page_text,
                    "page": page_num,
                })

        body = "\n".join(current_lines).strip()
        if body and len(body) >= 35:
            clauses.append({
                "clause_id": current_clause_id,
                "title": current_title,
                "text": body,
                "page": current_page,
            })

        return clauses
