import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")


def main() -> None:
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = [Path(value) for value in sys.argv[2:]]
    records = []

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        page_text = []
        for page in reader.pages:
            page_text.append(page.extract_text() or "")

        full_text = "\n\n".join(
            f"===== PAGE {index + 1} =====\n{text}"
            for index, text in enumerate(page_text)
        )
        output_path = output_dir / f"{safe_stem(pdf_path)}.txt"
        output_path.write_text(full_text, encoding="utf-8")

        metadata = {
            str(key).lstrip("/"): str(value)
            for key, value in (reader.metadata or {}).items()
        }
        doi_candidates = sorted(
            set(
                re.findall(
                    r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",
                    full_text,
                    flags=re.IGNORECASE,
                )
            )
        )
        records.append(
            {
                "pdf": str(pdf_path),
                "text": str(output_path),
                "pages": len(reader.pages),
                "metadata": metadata,
                "doi_candidates": doi_candidates[:20],
                "first_pages": "\n\n".join(page_text[:3]),
            }
        )

    (output_dir / "paper_metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
