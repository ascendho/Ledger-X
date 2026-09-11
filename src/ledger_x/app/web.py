"""Loopback-only demo UI. Merchant scope is fixed by the operator, not the browser."""
import json
from pathlib import Path
import secrets
import threading

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from ledger_x.app.agent import ReconciliationAgent
from ledger_x.app.data import DEFAULT_DB
from ledger_x.paths import PROMOTION_REPORT


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


def create_app(service, model=None):
    app = FastAPI(title="Ledger-X", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    token, lock = secrets.token_urlsafe(32), threading.Lock()
    agent = ReconciliationAgent(service, model=model, trace_dir=DEFAULT_DB.parent / "traces")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = Path(__file__).with_name("workbench.html").read_text()
        return HTMLResponse(html.replace("__TOKEN__", token).replace("__SCOPE__", ", ".join(sorted(service.allowed))),
                            headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"})

    @app.post("/api/chat")
    def chat(question: Question, x_ledger_x_token: str = Header(default="")):
        if not secrets.compare_digest(token, x_ledger_x_token):
            raise HTTPException(403, "Invalid local-session token")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A request is already running")
        try:
            return agent.run(question.question)
        finally:
            lock.release()

    @app.get("/api/health")
    def health():
        return {"synthetic": True, "scope": sorted(service.allowed), "database_exists": Path(service.db).is_file()}

    @app.get("/api/benchmark")
    def benchmark():
        path = PROMOTION_REPORT
        if not path.is_file():
            return {"available": False, "message": "尚未生成通过正确性门禁的性能报告"}
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            raise HTTPException(500, "Benchmark artifact is invalid")
        return {"available": True, "result": result}

    return app
