"""Export the case study's EDA notebooks to well-formatted, shareable PDFs.

Plain `jupyter nbconvert --to webpdf` clips long code lines at the page edge
instead of wrapping them. This script exports to HTML first, injects CSS that
wraps code/output text and caps image width, then prints that HTML to PDF
with Playwright (Chromium) for a layout that reads cleanly page to page.

Usage:
    pip install nbconvert playwright
    playwright install chromium
    python export_notebooks_to_pdf.py case_study_A_eda.ipynb case_study_A_eda_outliers.ipynb
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

EXTRA_CSS = """
<style>
@media print {
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
pre, code {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
}
.highlight pre, div.input_area pre, .jp-OutputArea-output pre {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
}
img, svg {
  max-width: 100% !important;
  height: auto !important;
}
div.output_subarea, .jp-OutputArea-output {
  max-width: 100% !important;
}
div.cell, .jp-Cell {
  page-break-inside: avoid;
  break-inside: avoid;
}
table {
  font-size: 0.82em;
}
body {
  max-width: 900px;
  margin: 0 auto;
}
</style>
"""


def notebook_to_pdf(notebook_path: Path) -> Path:
    pdf_path = notebook_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / (notebook_path.stem + ".html")
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "html", "--embed-images",
                str(notebook_path), "--output", str(html_path),
            ],
            check=True,
        )
        html = html_path.read_text(encoding="utf-8")
        html = html.replace("</head>", EXTRA_CSS + "</head>", 1) if "</head>" in html else EXTRA_CSS + html
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path}")
            page.wait_for_timeout(300)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "18mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
            )
            browser.close()
    print(f"wrote {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    notebooks = [Path(arg) for arg in sys.argv[1:]] or [
        Path("case_study_A_eda.ipynb"),
        Path("case_study_A_eda_outliers.ipynb"),
    ]
    for nb in notebooks:
        notebook_to_pdf(nb)
