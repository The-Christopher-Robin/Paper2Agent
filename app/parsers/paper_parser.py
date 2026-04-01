"""Parsers for extracting structured sections from research papers.

Supported sources:
  - arXiv IDs  (e.g. ``2301.12345``)
  - arXiv URLs (``https://arxiv.org/abs/...``)
  - GitHub repository URLs  (fetches README)
  - Local Markdown / text files
  - Raw text strings
"""

import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PaperParser:

    def parse(self, source: str) -> dict[str, Any]:
        source = source.strip()

        if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", source):
            return self._from_arxiv(source)
        if "arxiv.org" in source:
            return self._from_arxiv_url(source)
        if "github.com" in source:
            return self._from_github(source)
        if Path(source).is_file():
            return self._from_file(source)
        return self._from_text(source)

    # ── arXiv ────────────────────────────────────────────────────────

    def _from_arxiv(self, arxiv_id: str) -> dict:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                xml = resp.read().decode()
        except Exception as exc:
            logger.error("arXiv fetch failed for %s: %s", arxiv_id, exc)
            return {"title": arxiv_id, "sections": {}, "error": str(exc)}

        title = self._xml_tag(xml, "title") or arxiv_id
        abstract = self._xml_tag(xml, "summary") or ""
        authors = re.findall(r"<name>(.*?)</name>", xml)

        return {
            "title": title.strip(),
            "authors": authors,
            "source": f"arxiv:{arxiv_id}",
            "sections": {
                "abstract": abstract.strip(),
                "methodology": "",
                "results": "",
            },
        }

    def _from_arxiv_url(self, url: str) -> dict:
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        if m:
            return self._from_arxiv(m.group(1))
        return {"title": url, "sections": {}, "error": "Cannot extract arXiv ID"}

    # ── GitHub README ────────────────────────────────────────────────

    def _from_github(self, url: str) -> dict:
        parts = url.rstrip("/").split("/")
        if len(parts) < 5:
            return {"title": url, "sections": {}, "error": "Invalid GitHub URL"}

        owner, repo = parts[3], parts[4]
        readme = None
        for branch in ("main", "master"):
            raw = (f"https://raw.githubusercontent.com/"
                   f"{owner}/{repo}/{branch}/README.md")
            try:
                with urllib.request.urlopen(raw, timeout=15) as resp:
                    readme = resp.read().decode()
                break
            except Exception:
                continue

        if readme is None:
            return {"title": f"{owner}/{repo}", "sections": {},
                    "error": "README not found"}

        return {
            "title": f"{owner}/{repo}",
            "source": url,
            "sections": self._md_sections(readme),
        }

    # ── Local files ──────────────────────────────────────────────────

    def _from_file(self, path: str) -> dict:
        fp = Path(path)
        text = fp.read_text(encoding="utf-8", errors="replace")
        sections = (self._md_sections(text)
                     if fp.suffix == ".md"
                     else {"full_text": text})
        return {"title": fp.stem, "source": str(fp), "sections": sections}

    # ── Raw text ─────────────────────────────────────────────────────

    @staticmethod
    def _from_text(text: str) -> dict:
        return {
            "title": text[:60].strip(),
            "source": "raw_input",
            "sections": {"full_text": text},
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _md_sections(md: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        heading = "introduction"
        buf: list[str] = []

        for line in md.split("\n"):
            m = re.match(r"^(#{1,3})\s+(.*)", line)
            if m:
                if buf:
                    key = re.sub(r"[^\w\s]", "", heading).strip().lower()
                    key = key.replace(" ", "_")[:50]
                    sections[key] = "\n".join(buf).strip()
                heading = m.group(2)
                buf = []
            else:
                buf.append(line)

        if buf:
            key = re.sub(r"[^\w\s]", "", heading).strip().lower()
            key = key.replace(" ", "_")[:50]
            sections[key] = "\n".join(buf).strip()

        return sections

    @staticmethod
    def _xml_tag(data: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", data, re.DOTALL)
        return m.group(1) if m else None
