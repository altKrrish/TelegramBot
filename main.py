"""
Flask web server for the Telegram data-analyst bot.

Endpoints:
  POST /webhook        – Telegram Bot API webhook
  GET  /logs/<run_id>  – Publicly wget-able JSONL run log
  GET  /health         – Health-check
  POST /set-webhook    – Convenience: register the webhook with Telegram
"""

import json
import logging
import os
import re
import threading
import time
import uuid

import httpx
from flask import Flask, Response, request

from agent import run_agent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
BOT_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
BASE_URL = os.environ.get("BASE_URL", "")  # e.g. https://my-app.up.railway.app

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ---------------------------------------------------------------------------
# In-memory stores (persist as long as container lives)
# ---------------------------------------------------------------------------
# Logs: run_id -> list[dict]
_logs: dict[str, list[dict]] = {}

# Conversation history per chat (for multi-turn): chat_id -> list[str]
_conversations: dict[int, list[str]] = {}

# Dedup: set of already-processed update_ids
_seen_updates: set[int] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _send_telegram(chat_id: int, text: str):
    """Send a message via the Telegram Bot API."""
    try:
        r = httpx.post(
            f"{BOT_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
        log.info("Telegram sendMessage status=%s", r.status_code)
    except Exception as exc:
        log.error("Failed to send Telegram message: %s", exc)


def _format_reply(agent_response: str, log_url: str) -> str:
    """
    Ensure the agent's output is a valid JSON string with answer + log_url.
    Handles cases where the LLM wraps in markdown fences or forgets log_url.
    """
    text = agent_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try to parse as JSON
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON object from the text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None
        else:
            obj = None

    if obj is None:
        # Last resort: wrap raw text as answer
        return json.dumps({"answer": text, "log_url": log_url})

    if isinstance(obj, dict):
        # Always overwrite log_url with the correct one
        if "log_url" in obj:
            obj["log_url"] = log_url
            return json.dumps(obj)
        # If the LLM returned {"answer": ..., ...} but forgot log_url
        if "answer" in obj:
            obj["log_url"] = log_url
            return json.dumps(obj)
        # LLM returned just the answer object (e.g. {"state": "Assam"})
        return json.dumps({"answer": obj, "log_url": log_url})

    # Scalar or list answer
    return json.dumps({"answer": obj, "log_url": log_url})


def _process_message(chat_id: int, text: str, update_id: int):
    """Background worker: run agent and reply."""
    run_id = uuid.uuid4().hex[:12]
    log_url = f"{BASE_URL}/logs/{run_id}"
    log_entries: list[dict] = []

    log.info("[%s] Processing chat=%s update=%s", run_id, chat_id, update_id)

    try:
        # Conversation history (multi-turn)
        history = _conversations.get(chat_id, [])
        history.append(text)
        history = history[-10:]  # keep last 10
        _conversations[chat_id] = history

        # Run agent
        raw_response = run_agent(text, history, log_entries, log_url)

        # Format as a clean JSON reply
        reply = _format_reply(raw_response, log_url)

    except Exception as exc:
        log.exception("[%s] Agent error", run_id)
        log_entries.append({"event": "fatal_error", "error": str(exc), "ts": time.time()})
        reply = json.dumps({"answer": None, "log_url": log_url})

    # Store log
    _logs[run_id] = log_entries

    # Trim old logs (keep last 200 runs)
    if len(_logs) > 200:
        oldest_keys = sorted(_logs.keys())[:50]
        for k in oldest_keys:
            _logs.pop(k, None)

    # Send reply
    log.info("[%s] Replying: %s", run_id, reply[:200])
    _send_telegram(chat_id, reply)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    """Telegram webhook endpoint."""
    update = request.get_json(silent=True)
    if not update:
        return "bad request", 400

    update_id = update.get("update_id")
    message = update.get("message")
    if not message:
        return "ok"

    # Dedup
    if update_id in _seen_updates:
        return "ok"
    _seen_updates.add(update_id)
    # Trim dedup set
    if len(_seen_updates) > 5000:
        _seen_updates.clear()

    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip()
    if not text:
        return "ok"

    log.info("Received update=%s chat=%s text=%.80s…", update_id, chat_id, text)

    # Process in background so we return 200 immediately
    t = threading.Thread(
        target=_process_message,
        args=(chat_id, text, update_id),
        daemon=True,
    )
    t.start()

    return "ok"


@app.route("/logs/<run_id>")
def get_log(run_id):
    """Serve the JSONL run log — must be wget-able."""
    entries = _logs.get(run_id)
    if entries is None:
        return "not found", 404

    lines = [json.dumps(e, default=str) for e in entries]
    return Response(
        "\n".join(lines) + "\n",
        content_type="application/x-ndjson",
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.route("/health")
def health():
    return {"status": "ok", "runs_stored": len(_logs)}


@app.route("/set-webhook", methods=["POST"])
def set_webhook():
    """Convenience endpoint to register the Telegram webhook."""
    webhook_url = f"{BASE_URL}/webhook"
    r = httpx.post(
        f"{BOT_API}/setWebhook",
        json={"url": webhook_url, "allowed_updates": ["message"]},
        timeout=15,
    )
    log.info("setWebhook -> %s %s", r.status_code, r.text)
    return {"webhook_url": webhook_url, "telegram_response": r.json()}


@app.route("/")
def index():
    return {"message": "TDS Data Analyst Bot is running", "health": "/health"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    log.info("Starting bot on port %s, BASE_URL=%s", port, BASE_URL)

    # Auto-set webhook on startup if BASE_URL is configured
    if BASE_URL and TELEGRAM_BOT_TOKEN:
        webhook_url = f"{BASE_URL}/webhook"
        try:
            r = httpx.post(
                f"{BOT_API}/setWebhook",
                json={"url": webhook_url, "allowed_updates": ["message"]},
                timeout=15,
            )
            log.info("Auto-registered webhook: %s -> %s", webhook_url, r.json())
        except Exception as exc:
            log.warning("Could not auto-register webhook: %s", exc)

    app.run(host="0.0.0.0", port=port)
