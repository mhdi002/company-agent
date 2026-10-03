"""The agent loop: Start → gather companies → per company (isolated, concurrent) run the to-do items.

Decisions (to-do items 2, 3, 4) go through the SRLM engine. Everything is persisted after each step,
so a crash or Stop can be resumed: finished steps are loaded from the company's work directory.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.memory import StateStore
from agent.planner import plan_run
from agent.proposal import ModelWriter, SourceBook, TemplateWriter, build_record
from agent.srlm.engine import NOT_VERIFIED, SRLMEngine
from agent.srlm.tasks import (FACT_SHEET_SECTIONS, build_corpus, evidence_task, fact_sheet_task, field_task,
                              project_task)
from agent.todo import TodoList
from core import logging as plog
from core.config import load_config, resolve
from tools.docx_writer import read_docx_skill, write_proposal
from tools.fetch import fetch_site
from tools.http import HttpClient
from tools.research import make_research_provider, research_field
from tools.search import gather_companies, make_provider
from tools.telegram import TelegramClient

log = plog.get_logger("agent")


class Stopped(Exception):
    """Raised inside a company worker when Stop was requested."""


def safe_name(domain: str) -> str:
    return re.sub(r"[^\w.-]+", "_", domain)


def build_policy(cfg: dict):
    """Our trained model when a checkpoint exists, else the deterministic template policy (logged)."""
    from agent.policy import ModelPolicy, TemplatePolicy
    ck = resolve(cfg["agent"]["checkpoint"])
    tok_dir = resolve(cfg["paths"]["tokenizer"])
    count = None
    if (tok_dir / "tokenizer.json").exists():
        from tokenizer.tok import Tok
        count = Tok.load(tok_dir).count
    if cfg["agent"]["policy"] == "model":
        if ck.exists() and count:
            pol = ModelPolicy.from_checkpoint(ck, tok_dir, max_new_tokens=int(cfg["agent"]["max_new_tokens"]))
            return pol, pol.tok.count, ModelWriter(pol)
        log.event("policy", "warn", details=f"agent.policy=model but no checkpoint at {ck}; using template policy")
    return TemplatePolicy(), count or (lambda s: len(re.findall(r"\w+|[^\w\s]", s))), TemplateWriter()


class Agent:
    def __init__(self, cfg: dict | None = None, *, http: HttpClient | None = None, telegram: TelegramClient | None = None,
                 policy=None, count_tokens=None, writer=None):
        self.cfg = cfg or load_config()
        p = self.cfg["paths"]
        self.store = StateStore(resolve(p["state"]))
        self.todo = TodoList(self.store)
        self.http = http or HttpClient(self.cfg["fetch"], cache_dir=resolve(p["cache"]) / "http")
        self.telegram = telegram or TelegramClient(self.cfg)
        if policy is None:
            policy, count_tokens, writer = build_policy(self.cfg)
        self.policy, self.count = policy, count_tokens or (lambda s: len(s.split()))
        self.writer = writer or TemplateWriter()
        self.engine = SRLMEngine(self.policy, self.count, self.cfg["srlm"])
        self.search = make_provider(self.cfg, self.http)
        self.research = make_research_provider(self.cfg, self.http)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._skill: dict | None = None
        self._docs_since_logs = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, background: bool = True) -> str:
        if self.running:
            return self.store.data["run_id"]
        self._stop.clear()
        resume = self.store.data["status"] in ("running", "stopping", "stopped") and self.store.data["run_id"]
        run_id = self.store.data["run_id"] if resume else dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        if not resume:
            keep = {k: self.store.data[k] for k in ("processed_domains", "daily")}
            self.store.data = StateStore.fresh() | keep
        self.store.update(run_id=run_id, status="running", started_at=self.store.data.get("started_at") or
                          dt.datetime.now(dt.timezone.utc).isoformat(), finished_at=None)
        plog.configure(self.cfg["paths"]["logs"], run_id)
        log.event("run", "start", details=f"{'resume' if resume else 'new'} run {run_id}; policy={self.policy.name}")
        if background:
            self._thread = threading.Thread(target=self._run_safe, name="agent-run", daemon=True)
            self._thread.start()
        else:
            self._run_safe()
        return run_id

    def stop(self) -> None:
        if self.running:
            self._stop.set()
            self.store.update(status="stopping")
            log.event("run", "stopping", details="stop requested; finishing current step")

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _check(self) -> None:
        if self._stop.is_set():
            raise Stopped()

    # ------------------------------------------------------------------ run
    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as e:  # never die silently
            log.error("run", e)
            self.store.update(status="failed")

    def _run(self) -> None:
        cfg = self.cfg
        self.todo.ensure_run_item()
        gather = self.todo.get("gather")
        if gather["status"] != "done":
            self.todo.set("gather", "in_progress")
            plan = plan_run(cfg, self.store.processed_today())
            if plan["limit"] == 0:
                self.todo.set("gather", "skipped", "daily limit reached")
                self._finish("done")
                return
            t0 = time.time()
            try:
                found = gather_companies(self.search, plan["countries"], plan["industries"], plan["limit"],
                                         set(self.store.data["processed_domains"]),
                                         int(cfg["search"]["results_per_query"]))
            except Exception as e:
                self.todo.set("gather", "failed", str(e))
                log.error("gather", e, tool=self.search.name)
                self._finish("failed")
                return
            for c in found:
                self.store.set_company(c["domain"], name=c["name"], url=c["url"], status="pending", source=c["source"])
                self.todo.add_company(c["domain"])
            self.store.bump("found", len(found))
            self.todo.set("gather", "done" if found else "failed", None if found else "no companies found")
            log.event("gather", "done" if found else "failed", tool=self.search.name, duration=time.time() - t0,
                      details=f"{len(found)} companies: {[c['domain'] for c in found]}")
            if not found:
                self._send_logs(final=True)
                self._finish("failed")
                return
        pending = [d for d, c in self.store.data["companies"].items() if c["status"] not in ("done", "failed")]
        conc = max(1, int(cfg["agent"]["concurrency"]))
        with ThreadPoolExecutor(max_workers=conc, thread_name_prefix="company") as ex:
            list(ex.map(self._process_company_safe, pending))
        if self._stop.is_set():
            self.store.update(status="stopped", current_company=None)
            log.event("run", "stopped", details="run stopped; Start again to resume")
            return
        self._send_logs(final=True)
        self._finish("done")

    def _finish(self, status: str) -> None:
        self.store.update(status=status, finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), current_company=None)
        c = self.store.data["counters"]
        log.event("run", status, details=f"found={c['found']} processed={c['processed']} failed={c['failed']} sent={c['sent']}")

    # ------------------------------------------------------------------ per company
    def _process_company_safe(self, domain: str) -> None:
        if self._stop.is_set():
            return
        self.store.update(current_company=domain)
        self.store.set_company(domain, status="in_progress")
        t0 = time.time()
        try:
            result = self.process_company(domain)
            self.store.set_company(domain, status="done", finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                                   files=result.get("files"))
            self.store.mark_processed(domain)
            self.store.bump("processed")
            log.event("company", "done", company=domain, duration=time.time() - t0, details=result.get("summary", ""))
        except Stopped:
            self.store.set_company(domain, status="pending")
            for it in self.todo.company_items(domain):
                if it["status"] == "in_progress":
                    self.todo.set(it["id"], "pending", "stopped")
            return
        except Exception as e:  # per-company isolation: one failure never stops the run
            log.error("company", e, company=domain)
            self.store.set_company(domain, status="failed", error=f"{type(e).__name__}: {e}")
            self.store.mark_processed(domain)
            self.store.bump("failed")
            for it in self.todo.company_items(domain):
                if it["status"] in ("pending", "in_progress"):
                    self.todo.set(it["id"], "failed", f"{type(e).__name__}: {e}")
            self._notify(f"❌ {domain}: failed — {type(e).__name__}: {str(e)[:200]}", domain)
        with self._lock:
            self._docs_since_logs += 1
            if self._docs_since_logs >= int(self.cfg["agent"]["send_logs_every"]):
                self._docs_since_logs = 0
                self._send_logs(final=False)

    def _work(self, domain: str) -> Path:
        d = resolve(self.cfg["paths"]["cache"]) / "work" / safe_name(domain)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _step(self, domain: str, key: str, fn) -> Any:
        """Run a to-do item once; reuse its saved output when resuming."""
        item = f"{domain}:{key}"
        path = self._work(domain) / f"{key}.json"
        it = self.todo.get(item)
        if path.exists() and it and it["status"] == "done":
            return json.loads(path.read_text(encoding="utf-8"))
        self._check()
        self.todo.set(item, "in_progress")
        t0 = time.time()
        try:
            out = fn()
        except Stopped:
            raise
        except Exception as e:
            self.todo.set(item, "failed", f"{type(e).__name__}: {e}")
            raise
        path.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
        self.todo.set(item, "done")
        log.event(key, "done", company=domain, duration=time.time() - t0)
        return out

    def _decide(self, task) -> dict:
        self._check()
        d = self.engine.run(task)
        return {"output": d.output, "verified": d.verified, "status": d.status, "summary": d.summary()}

    def process_company(self, domain: str) -> dict:
        comp = self.store.company(domain)
        url = comp.get("url") or f"https://{domain}/"
        cfg = self.cfg

        # [2] fetch working field
        def fields() -> dict:
            site = fetch_site(self.http, url, int(cfg["fetch"]["max_pages_per_site"]), company=domain)
            dec = self._decide(field_task(domain, site["pages"]))
            out = dec["output"] if dec["verified"] else {}
            profile = {"name": out.get("name") if out.get("name") not in (None, "unknown") else comp.get("name", domain),
                       "field": out.get("field", NOT_VERIFIED) if dec["verified"] else NOT_VERIFIED,
                       "country": out.get("country", NOT_VERIFIED) if dec["verified"] else NOT_VERIFIED,
                       "services": out.get("services", []), "size": out.get("size", "not stated")}
            return {"pages": site["pages"], "site_errors": site["errors"], "profile": profile, "decision": dec["summary"]}
        f = self._step(domain, "fields", fields)
        profile, pages = f["profile"], f["pages"]

        # [3] research the field
        def research() -> dict:
            if profile["field"] in (NOT_VERIFIED, "unknown"):
                return {"evidence": [], "facts": NOT_VERIFIED, "projects": NOT_VERIFIED, "decisions": {}}
            ev = research_field(self.research, profile["field"], profile["services"],
                                int(cfg["research"]["max_sources"]), company=domain)
            if not ev:
                return {"evidence": [], "facts": NOT_VERIFIED, "projects": NOT_VERIFIED, "decisions": {}}
            d_ev = self._decide(evidence_task(domain, profile, ev))
            facts = d_ev["output"]
            d_pr = {"output": NOT_VERIFIED, "summary": {}}
            if d_ev["verified"]:
                d_pr = self._decide(project_task(domain, profile, facts))
            return {"evidence": ev, "facts": facts, "projects": d_pr["output"],
                    "decisions": {"evidence": d_ev["summary"], "projects": d_pr["summary"]}}
        r = self._step(domain, "research", research)
        facts = r["facts"] if isinstance(r["facts"], list) else []
        projects = r["projects"] if isinstance(r["projects"], list) else []

        # [4] find the text needed for the proposal (fact sheet per section)
        def text() -> dict:
            corpus = build_corpus(pages, facts, projects)
            sheets, decisions = {}, {}
            for section in FACT_SHEET_SECTIONS:
                d = self._decide(fact_sheet_task(domain, section, profile, corpus))
                sheets[section] = d["output"]
                decisions[section] = d["summary"]
            return {"corpus_urls": {c["id"]: c["url"] for c in corpus}, "fact_sheets": sheets, "decisions": decisions}
        t = self._step(domain, "text", text)

        # [5] fetch and read the Word docx skill (once per run)
        def skill() -> dict:
            if self._skill is None:
                self._skill = read_docx_skill(cfg["paths"]["docx_skill"])
            return self._skill
        sk = self._step(domain, "docx_skill", skill)

        # [6] finalize the proposal
        def finalize() -> dict:
            book = SourceBook()
            titles = {e["id"]: e.get("title", "") for e in r["evidence"]}
            ctx = {"profile": profile, "projects": projects, "fact_sheets": t["fact_sheets"], "book": book,
                   "corpus_urls": t["corpus_urls"], "domain": domain}
            sections = self.writer.write(ctx)
            for s in book.items.values():
                if s["id"].startswith("page:"):
                    s["title"] = f"{profile['name']} website ({urlparse(s['url']).path or '/'})"
                else:
                    s["title"] = titles.get(s["id"]) or s["title"] or s["id"]
            # refresh the Sources section with titles
            sections[-1]["bullets"] = [f"[{s['label']}] {s['title']} — {s['url'] or 'company website'}"
                                       for s in book.as_list()] or [NOT_VERIFIED]
            stamp = dt.datetime.now().strftime("%Y%m%d")
            base = f"{stamp}_{safe_name(domain)}"
            docx_path = resolve(cfg["paths"]["proposals"]) / f"{base}.docx"
            info = write_proposal(sections, docx_path, f"Project Proposal — {profile['name']}",
                                  cfg["docx"]["backend"] if sk.get("available") else "python-docx",
                                  cfg["docx"]["page_size"], company=domain)
            record = build_record(domain, url, profile, r["projects"], facts, sections,
                                  {"fields": f["decision"], **r.get("decisions", {}), "fact_sheets": t["decisions"]}, book)
            record["docx"] = info
            rec_path = resolve(cfg["paths"]["records"]) / f"{base}.json"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            rec_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            verified = sum(1 for s in sections if s.get("verified"))
            return {"docx": str(docx_path), "record": str(rec_path), "sections": len(sections),
                    "verified_sections": verified, "backend": info["backend"]}
        fin = self._step(domain, "finalize", finalize)

        # [7] save everything and send via Telegram
        def save_send() -> dict:
            receipts = [self.telegram.send_document(fin["docx"], f"Proposal: {profile['name']} ({domain})", company=domain),
                        self.telegram.send_document(fin["record"], f"Record: {domain}", company=domain)]
            top = projects[0]["title"] if projects else NOT_VERIFIED
            msg = (f"✅ {profile['name']} ({domain})\nField: {profile['field']} · Country: {profile['country']}\n"
                   f"Top project: {top}\nVerified sections: {fin['verified_sections']}/{fin['sections']}")
            receipts += self._notify(msg, domain)
            sent = sum(1 for x in receipts if x.get("delivered"))
            if any(x.get("delivered") for x in receipts[:1]):
                self.store.bump("sent")
            return {"receipts": receipts, "delivered": sent}
        ss = self._step(domain, "save_send", save_send)
        return {"files": {"docx": fin["docx"], "record": fin["record"]},
                "summary": f"field={profile['field']} projects={len(projects)} verified={fin['verified_sections']}/"
                           f"{fin['sections']} telegram_delivered={ss['delivered']}"}

    # ------------------------------------------------------------------ telegram helpers
    def _notify(self, text: str, company: str | None = None) -> list[dict]:
        try:
            return self.telegram.send_message(text, company=company)
        except Exception as e:  # Telegram problems never fail a company
            log.error("telegram", e, company=company)
            return [{"delivered": False, "error": str(e)}]

    def _send_logs(self, final: bool) -> None:
        d = plog.log_dir()
        c = self.store.data["counters"]
        self._notify(f"{'🏁 Run finished' if final else '📋 Progress'} {self.store.data['run_id']}: found={c['found']} "
                     f"processed={c['processed']} failed={c['failed']} sent={c['sent']}")
        for name in ("agent.log", "tools.log", "errors.log", "srlm.jsonl"):
            p = d / name
            if p.exists() and 0 < p.stat().st_size < 45_000_000:
                try:
                    self.telegram.send_document(p, f"{name} ({'final' if final else 'progress'})")
                except Exception as e:
                    log.error("telegram.logs", e)

    def snapshot(self) -> dict:
        s = self.store.snapshot()
        s["running"] = self.running
        s["policy"] = self.policy.name
        return s
