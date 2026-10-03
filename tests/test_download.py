import http.server
import io
import json
import threading
import zipfile

import pytest

from data import download as dl
from data.registry import BY_NAME, REGISTRY, Dataset

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class RangeHandler(http.server.BaseHTTPRequestHandler):
    fail_first = {"n": 0}

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        if self.path == "/flaky" and RangeHandler.fail_first["n"] == 0:
            # send half the payload then drop the connection
            RangeHandler.fail_first["n"] = 1
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD[: len(PAYLOAD) // 2])
            self.wfile.flush()
            self.connection.close()
            return
        start = 0
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_download_and_checksum(server, tmp_path):
    p = dl.download_url(server + "/file.bin", tmp_path / "file.bin")
    assert p.read_bytes() == PAYLOAD
    assert (tmp_path / "file.bin.sha256").read_text() == dl.sha256_file(p)
    # second call is a verified no-op
    dl.download_url(server + "/file.bin", tmp_path / "file.bin")


def test_resume_from_partial(server, tmp_path):
    part = tmp_path / "r.bin.part"
    part.write_bytes(PAYLOAD[:1000])
    p = dl.download_url(server + "/r.bin", tmp_path / "r.bin")
    assert p.read_bytes() == PAYLOAD


def test_resume_after_dropped_connection(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    RangeHandler.fail_first["n"] = 0
    p = dl.download_url(server + "/flaky", tmp_path / "f.bin")
    assert p.read_bytes() == PAYLOAD


def test_checksum_mismatch(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError):
        dl.verify_or_record(f, "0" * 64)


def test_convert_formats(tmp_path):
    ds = Dataset("t", "business", "url", "CC0", "x", text_fields=["title", "objective"], fmt="zip-csv")
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("p.csv", "title;objective\nSolar;Build a solar park\nWind;Repower turbines\n")
        z.writestr("q.xml", "<root><title>Grid</title><objective>Model the grid</objective></root>")
    jl = tmp_path / "b.jsonl"
    jl.write_text(json.dumps({"title": "T", "objective": "O"}) + "\n")
    import pyarrow as pa
    import pyarrow.parquet as pq
    pqf = tmp_path / "c.parquet"
    pq.write_table(pa.table({"title": ["P"], "objective": ["Q"]}), pqf)
    out = tmp_path / "docs.jsonl"
    n = dl.convert(ds, [zp, jl, pqf], out)
    texts = [json.loads(line)["text"] for line in out.read_text().splitlines()]
    assert n == 5
    assert "Solar\n\nBuild a solar park" in texts and "Grid\n\nModel the grid" in texts and "P\n\nQ" in texts


def test_registry_has_licenses_and_groups():
    groups = {d.group for d in REGISTRY}
    assert groups == {"general", "business", "agentic"}
    assert all(d.license for d in REGISTRY)
    assert "synthetic-srlm-traces" in BY_NAME


def test_manifest(tmp_path):
    dl.save_state(tmp_path, {"fineweb-edu": {"status": "ok", "docs": 3, "bytes": 2048,
                                              "files": [{"path": "a", "sha256": "ab" * 32}]}})
    p = dl.write_manifest(tmp_path, tmp_path / "MANIFEST.md")
    text = p.read_text()
    assert "ODC-By 1.0" in text and "| ok |" in text and "2.0 KB" in text


def test_sample_corpus_has_defects():
    from data.sample.build_sample import build_corpus
    docs = build_corpus()
    joined = "\n".join(d["text"] for d in docs)
    assert "<html>" in joined and "@mail.example" in joined and "Ã©" in joined
    assert len(docs) > 3000
