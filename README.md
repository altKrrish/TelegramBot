# TDS Data Analyst Telegram Bot

An LLM-powered Telegram bot that answers data-analysis questions. It receives plain-text questions, uses an LLM agent (via OpenRouter) to fetch data, run Python analysis, and replies with a structured JSON answer.

## Architecture

```
Telegram → POST /webhook → Flask server
                              ↓
                     OpenRouter LLM (tool-use loop)
                     Tools: web_fetch, python_exec
                              ↓
                     JSON reply → Telegram
                     JSONL log  → GET /logs/<run_id>
```

## Setup

### 1. Create a Telegram Bot
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the bot token

### 2. Get an OpenRouter API Key
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up and create an API key

### 3. Environment Variables
```
TELEGRAM_BOT_TOKEN=<your bot token>
OPENROUTER_API_KEY=<your openrouter key>
BASE_URL=<your deployment URL, e.g. https://my-app.up.railway.app>
MODEL=google/gemini-2.5-flash   # optional, defaults to gemini-2.5-flash
```

### 4. Deploy on Railway
1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app), create a new project
3. Connect your GitHub repo
4. Add the environment variables above
5. Railway auto-detects the Dockerfile and deploys
6. Copy the public URL and set it as `BASE_URL`
7. The webhook registers automatically on startup

### 5. Local Development
```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=...
export OPENROUTER_API_KEY=...
export BASE_URL=https://<ngrok-url>
python main.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/webhook` | POST | Telegram webhook |
| `/logs/<run_id>` | GET | Public JSONL run log |
| `/health` | GET | Health check |
| `/set-webhook` | POST | Register webhook with Telegram |

## Response Format

The bot replies with exactly one JSON object:
```json
{"answer": <answer shaped as question asks>, "log_url": "https://host/logs/<run_id>"}
```
