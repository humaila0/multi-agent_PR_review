# 🤖 AI PR Review Agent

An automated Pull Request reviewer that uses LangGraph multi-agent system to catch bugs, security vulnerabilities, performance issues, and breaking changes — on every PR, automatically.

---

## 🚀 Features

- **Automatic reviews** on every Pull Request via GitHub Webhooks
- **Cross-file awareness** — detects breaking changes across related files
- **Multi-agent system** — specialist agents for Security, Performance, and Maintainability
- **Smart triage** — only runs relevant agents, saving 60-84% tokens
- **Token usage tracking** — logged per agent in every PR comment
- **Production ready** — handles large PRs, filters irrelevant files, truncates safely

---

## 🏗️ Architecture

```
Developer opens PR
        ↓
GitHub Webhook → FastAPI Backend
        ↓
Phase 1: Fetch PR Diff
        ↓
Phase 2: Repo Context Search (related files)
        ↓
Phase 3: LangGraph Multi-Agent Review
   ┌─────────────────────────────┐
   │  Triage Agent               │ ← detects issue type
   │    ↓                        │
   │  Security Agent    🔒       │ ← only if needed
   │    ↓                        │
   │  Performance Agent ⚡       │ ← only if needed
   │    ↓                        │
   │  Maintainability   🔧       │ ← only if needed
   │    ↓                        │
   │  Judge Agent       ⚖️       │ ← combines all reviews
   └─────────────────────────────┘
        ↓
AI Comment Posted on PR
```

---

## 📋 What It Detects

| Category | Examples |
|---|---|
| **Security** | SQL injection, hardcoded secrets, missing auth |
| **Performance** | N+1 queries, missing caching, blocking operations |
| **Maintainability** | Breaking changes, duplicate code, missing tests |
| **Cross-file** | Renamed variables breaking importers |

---

## 🛠️ Tech Stack

- **Backend** — FastAPI + Uvicorn
- **AI** — OpenAI GPT-4o-mini
- **Agent Framework** — LangGraph
- **GitHub** — PyGithub + Webhooks
- **Tunnel** — Ngrok (development)

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/pr-review-agent
cd pr-review-agent
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file:

```env
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_WEBHOOK_SECRET=your_secret_here
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

| Variable | How to get |
|---|---|
| `GITHUB_TOKEN` | GitHub → Settings → Developer Settings → Personal Access Tokens |
| `GITHUB_WEBHOOK_SECRET` | Any random string you choose |
| `OPENAI_API_KEY` | platform.openai.com → API Keys |

### 3. Run the Server

```bash
uvicorn main:app --reload
```

### 4. Expose with Ngrok

```bash
.\ngrok.exe http 8000
```

Copy the `https://xxx.ngrok-free.app` URL.

### 5. Add GitHub Webhook

Go to your repo → **Settings → Webhooks → Add webhook**

| Field | Value |
|---|---|
| Payload URL | `https://xxx.ngrok-free.app/webhooks/github` |
| Content type | `application/json` |
| Secret | Same as `GITHUB_WEBHOOK_SECRET` |
| Events | Pull requests only |

---

## 📁 Project Structure

```
pr-review-agent/
├── main.py          # FastAPI webhook server
├── agent.py         # Core agent logic (all 3 phases)
├── .env             # Secret keys (never commit!)
├── requirements.txt
└── ngrok.exe
```

---

## 🧪 Test Cases

| Test | Change | Expected |
|---|---|---|
| TC1 — Config Break | `API_TIMEOUT` → `REQUEST_TIMEOUT` | ❌ Needs Changes |
| TC2 — Hardcoded Secret | `SECRET_KEY = "abc123"` | ❌ Needs Changes |
| TC3 — SQL Injection | String concat in query | ❌ Needs Changes |
| TC4 — N+1 Query | DB call inside loop | ❌ Needs Changes |
| TC5 — Duplicate Code | Same function twice | ⚠️ Minor Issues |
| TC6 — Safe PR | README update | ✅ Looks Good |

---

## 📊 Token Efficiency

| Setup | Tokens per PR | Cost |
|---|---|---|
| Phase 2 (single agent) | ~2000 | $$ |
| Phase 3 (all 4 agents) | ~4600 | $$$$ |
| Phase 3 (with triage) | ~500-750 | ¢ |

Optimizations implemented:
- `compress_diff()` — removes unchanged lines (~70% saving)
- Triage Agent — skips unnecessary specialist agents
- Context filtering — only relevant files sent to LLM
- Per-agent token limits
- Judge gets compact input only

---

## 💬 Sample PR Comment

```
🤖 AI Code Review

### Summary
- API_TIMEOUT renamed to REQUEST_TIMEOUT in config.py
- client.py still imports old name — will cause NameError

### Issues
- Severity: High
- File: client.py
- Problem: Outdated import
- Why it matters: ImportError at runtime
- Suggestion: Update to REQUEST_TIMEOUT

### Test Gaps
- No tests cover this config change

### Verdict
❌ Needs Changes

### Token Usage
- Total tokens: 750
```

---

## 🔮 Phases

- ✅ **Phase 1** — Single agent, PR diff → LLM → Review
- ✅ **Phase 2** — Context retrieval, Repo search → LLM Review
- ✅ **Phase 3** — Multi-agent system with LangGraph + Triage

---

## 📄 License

MIT
