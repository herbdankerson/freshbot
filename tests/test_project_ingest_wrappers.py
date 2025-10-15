import base64

from src.freshbot.pipeline import project_ingest


def test_ingest_project_code_wraps_flow(tmp_path, monkeypatch):
    sample = tmp_path / "example.py"
    sample.write_text("print('hello world')", encoding="utf-8")

    captured = {}

    def fake_flow(**kwargs):
        captured.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(project_ingest, "freshbot_document_ingest", fake_flow)

    result = project_ingest.ingest_project_code(
        sample,
        extra_metadata={"tags": ["demo"]},
        source_root=tmp_path,
    )

    assert result == {"status": "succeeded"}

    assert captured["target_namespace"] == "kb"
    assert captured["target_entries"] == "kb.entries"
    assert captured["display_name"] == "example.py"

    payload = base64.b64decode(captured["content_b64"].encode("ascii"))
    assert payload == sample.read_bytes()

    metadata = captured["extra_metadata"]
    assert metadata["source"]["relative_path"] == "example.py"
    assert metadata["source"]["category"] == "code"
    assert metadata["source"]["language"] == "python"
    assert metadata["freshbot"]["namespace"] == "kb"
    assert metadata["freshbot"]["is_dev"] is True
    assert metadata["is_dev"] is True
    assert metadata["tags"] == ["demo"]


def test_ingest_project_docs_sets_docs_category(tmp_path, monkeypatch):
    sample = tmp_path / "overview.md"
    sample.write_text("# Heading", encoding="utf-8")

    captured = {}

    def fake_flow(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(project_ingest, "freshbot_document_ingest", fake_flow)

    project_ingest.ingest_project_docs(sample)

    metadata = captured["extra_metadata"]
    assert metadata["source"]["category"] == "docs"
    assert metadata["freshbot"]["category"] == "docs"
    assert metadata["freshbot"]["namespace"] == "kb"
    assert captured["target_namespace"] == "kb"
    assert captured["target_entries"] == "kb.entries"
    assert metadata["freshbot"]["is_dev"] is True
