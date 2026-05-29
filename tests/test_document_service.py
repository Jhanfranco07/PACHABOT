from pathlib import Path

from app.config import Settings
from app.core.logger import setup_logging
from app.services.document_service import DocumentService


def test_document_service_builds_processed_chunks_from_raw_text(tmp_path: Path) -> None:
    settings = Settings()
    settings.raw_data_dir = tmp_path / "raw"
    settings.cleaned_data_dir = tmp_path / "cleaned"
    settings.consolidated_data_dir = tmp_path / "consolidated"
    settings.processed_data_dir = tmp_path / "processed"
    settings.processed_chunks_file = settings.processed_data_dir / "chunks.json"
    settings.raw_data_dir.mkdir()
    (settings.raw_data_dir / "ordenanza_108_2012.txt").write_text(
        "TITULO I DEFINICIONES\nARTICULO 2.- Se entiende por comercio ambulatorio la actividad temporal.",
        encoding="utf-8",
    )
    (settings.raw_data_dir / "ordenanza_227_2019.txt").write_text(
        "TITULO I DEFINICIONES\nARTICULO 2.- Se entiende por comercio ambulatorio la actividad autorizada.",
        encoding="utf-8",
    )

    chunks = DocumentService(settings, setup_logging("INFO")).build_chunks()

    assert chunks
    assert settings.processed_chunks_file.exists()
    assert (settings.cleaned_data_dir / "ordenanza_108_2012.cleaned.txt").exists()
    assert (settings.consolidated_data_dir / "norma_consolidada.json").exists()
