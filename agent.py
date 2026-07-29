"""
LLM agent with tool-use loop via OpenRouter.
Receives a data-analysis question, iteratively fetches data and executes
Python code, then returns the answer as a JSON string.
"""

import os
import io
import json
import time
import traceback
import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenRouter client — FREE models only
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Free models on OpenRouter (tried in order). All support tool/function calling.
FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
    "inclusionai/ling-3.0-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]
MODEL = os.environ.get("MODEL", FREE_MODELS[0])

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a world-class data analysis agent. You receive data-analysis questions and must provide accurate answers.

TOOLS AVAILABLE:
1. **web_fetch** – Download content from any public URL (HTML pages, CSV, JSON, Excel).
2. **python_exec** – Execute Python code. Libraries available: pandas, numpy, json, math, statistics, re, csv, io, collections, datetime, BeautifulSoup, openpyxl. Assign your computed answer to the variable `RESULT`.

WORKFLOW:
1. Read the question. Identify what data is needed and what analysis to perform.
2. If the question embeds data inline, parse it directly in python_exec.
3. If URLs are mentioned, fetch them with web_fetch first.
4. If the question references a well-known dataset (MOSPI, etc.) without a URL, search/fetch data using web_fetch OR use python_exec / your knowledge to answer.
5. Use python_exec to run analysis code. You may call tools multiple times.
6. When you have the final answer, reply with ONLY a single JSON object.

CRITICAL FORMATTING RULES:
- Your final text reply MUST be a single valid JSON object and NOTHING ELSE.
- Do NOT include prose, explanation, markdown code fences, or citation markers (like 【...】).
- Match the exact JSON keys/structure requested by the question.
- For the `log_url` field, use exactly: __LOG_URL__
- Use proper casing for names (e.g. "Assam" or "Uttar Pradesh").
- Numeric answers: match the exact type asked for (int vs float).

EXAMPLE:
Question: Which state has highest maternal mortality rate based on MOSPI data? Reply with ONLY this JSON object and nothing else: {"answer": {"state": "<state name>"}, "log_url": "<url>"}
Your Response:
{"answer": {"state": "Uttar Pradesh"}, "log_url": "__LOG_URL__"}
"""

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Download content from a public URL. Returns extracted text for HTML, "
                "raw text for CSV/JSON, or CSV representation for Excel files. "
                "Max 80 000 chars returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute Python code for data analysis. "
                "Available: pandas (pd), numpy (np), json, math, statistics, re, "
                "csv, io, collections, datetime, BeautifulSoup, requests-style httpx. "
                "Assign your final answer to `RESULT`. Use print() for debug output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
MAX_FETCH = 80_000  # chars returned to the LLM


def _do_web_fetch(url: str) -> str:
    """Fetch a URL and return usable text."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 DataBot/1.0"
            )
        }
        with httpx.Client(follow_redirects=True, timeout=45, headers=headers) as h:
            resp = h.get(url)
            resp.raise_for_status()

        ct = resp.headers.get("content-type", "").lower()
        url_l = url.lower()

        # Excel ----------------------------------------------------------
        if "spreadsheet" in ct or url_l.endswith((".xlsx", ".xls")):
            import pandas as pd

            df = pd.read_excel(io.BytesIO(resp.content))
            csv_text = df.to_csv(index=False)
            return f"[Excel: {len(df)} rows × {len(df.columns)} cols]\n{csv_text}"[:MAX_FETCH]

        # HTML ------------------------------------------------------------
        if "text/html" in ct:
            from bs4 import BeautifulSoup
            import pandas as pd

            soup = BeautifulSoup(resp.text, "lxml")
            # try tables first
            try:
                tables = pd.read_html(io.StringIO(resp.text))
                if tables:
                    parts = [f"[HTML page with {len(tables)} table(s)]"]
                    for i, df in enumerate(tables[:10]):
                        parts.append(f"\n--- Table {i + 1} ({len(df)} rows) ---\n{df.to_csv(index=False)}")
                    return "\n".join(parts)[:MAX_FETCH]
            except Exception:
                pass
            return soup.get_text(separator="\n", strip=True)[:MAX_FETCH]

        # CSV / JSON / plain text -----------------------------------------
        return resp.text[:MAX_FETCH]

    except Exception as exc:
        return f"ERROR fetching {url}: {exc}"


class PythonExecutor:
    """Maintains a persistent namespace across calls within one agent run."""

    def __init__(self):
        self._ns: dict = {}
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        import pandas as pd
        import numpy as np
        import json as _json
        import math, statistics, re, csv, collections, datetime

        self._ns.update(
            {
                "__builtins__": __builtins__,
                "pd": pd,
                "np": np,
                "json": _json,
                "math": math,
                "statistics": statistics,
                "re": re,
                "csv": csv,
                "io": io,
                "collections": collections,
                "datetime": datetime,
                "Counter": collections.Counter,
                "defaultdict": collections.defaultdict,
            }
        )
        try:
            from bs4 import BeautifulSoup
            self._ns["BeautifulSoup"] = BeautifulSoup
        except ImportError:
            pass
        self._initialized = True

    def run(self, code: str) -> str:
        self._ensure_init()
        import sys

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            exec(code, self._ns)
            out = buf.getvalue()
            result = self._ns.get("RESULT")
            parts = []
            if out:
                parts.append(out.rstrip())
            if result is not None:
                parts.append(f"RESULT = {json.dumps(result, default=str)}")
            return "\n".join(parts) or "(no output)"
        except Exception:
            out = buf.getvalue()
            tb = traceback.format_exc()
            return f"{out}\nERROR:\n{tb}"
        finally:
            sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# LLM call with automatic fallback across free models
# ---------------------------------------------------------------------------

def _call_llm_with_fallback(messages, log_entries, log_url):
    """Try each free model in order until one succeeds."""
    models_to_try = [MODEL] + [m for m in FREE_MODELS if m != MODEL]

    for model in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=16384,
            )
            log_entries.append({"event": "llm_ok", "model": model, "ts": time.time()})
            return resp
        except Exception as exc:
            log_entries.append(
                {"event": "llm_error", "model": model, "error": str(exc), "ts": time.time()}
            )
            time.sleep(1)
            continue

    log_entries.append({"event": "all_models_failed", "ts": time.time()})
    return None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 20


def run_agent(
    question: str,
    conversation_history: list[str],
    log_entries: list[dict],
    log_url: str,
) -> str:
    """
    Run the tool-use agent loop.
    Returns the raw text the bot should send as its Telegram reply.
    """
    system = SYSTEM_PROMPT.replace("__LOG_URL__", log_url)

    # Build message list ---------------------------------------------------
    messages: list[dict] = [{"role": "system", "content": system}]

    # Prior turns (multi-turn context)
    for prior in conversation_history[:-1]:
        messages.append({"role": "user", "content": prior})
        messages.append({"role": "assistant", "content": "(noted)"})

    messages.append({"role": "user", "content": question})

    log_entries.append(
        {
            "event": "agent_start",
            "ts": time.time(),
            "question": question,
            "history_len": len(conversation_history),
            "model": MODEL,
        }
    )

    executor = PythonExecutor()

    for iteration in range(MAX_ITERATIONS):
        t0 = time.time()
        log_entries.append({"event": "llm_request", "iter": iteration, "ts": t0})

        resp = _call_llm_with_fallback(messages, log_entries, log_url)
        if resp is None:
            return json.dumps({"answer": None, "log_url": log_url})

        choice = resp.choices[0]
        msg = choice.message

        # ---- tool calls --------------------------------------------------
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}

                log_entries.append(
                    {"event": "tool_call", "tool": name, "args": args, "ts": time.time()}
                )

                if name == "web_fetch":
                    result_str = _do_web_fetch(args.get("url", ""))
                elif name == "python_exec":
                    result_str = executor.run(args.get("code", ""))
                else:
                    result_str = f"Unknown tool: {name}"

                # truncate for LLM context
                if len(result_str) > MAX_FETCH:
                    result_str = result_str[:MAX_FETCH] + "\n…(truncated)"

                log_entries.append(
                    {
                        "event": "tool_result",
                        "tool": name,
                        "chars": len(result_str),
                        "preview": result_str[:500],
                        "ts": time.time(),
                    }
                )

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                )
            continue  # next iteration

        # ---- final text response -----------------------------------------
        content = (msg.content or "").strip()
        log_entries.append({"event": "agent_done", "response": content, "ts": time.time()})
        return content

    # exhausted iterations
    log_entries.append({"event": "max_iterations", "ts": time.time()})
    return json.dumps({"answer": None, "log_url": log_url})
