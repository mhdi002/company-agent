"""Run the UI: python -m app [--host 127.0.0.1] [--port 8000]"""
import argparse

import uvicorn

from core.config import load_config

if __name__ == "__main__":
    cfg = load_config()["app"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=cfg["host"])
    ap.add_argument("--port", type=int, default=cfg["port"])
    a = ap.parse_args()
    uvicorn.run("app.server:app", host=a.host, port=a.port, log_level="info")
