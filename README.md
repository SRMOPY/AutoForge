# ⚒️ AutoForge

> Describe any software project. Three AI agents plan it, build it, and review it — autonomously.

```
You:  "a REST API for a blog with posts, comments, and user auth"
       ↓
🏗️  Architect  → architecture plan, file tree, security design
       ↓
💻  Coder      → writes every file to disk
       ↓
🔍  Reviewer   → audits for bugs & security issues, applies fixes
       ↓
📁  output/your-project/  ← ready to run
```

---

## Free Models Supported

| Provider | Model | Free? | Speed |
|---|---|---|---|
| **Gemini** (default) | gemini-2.0-flash | ✅ Free tier | Fast |
| **Groq** | llama-3.3-70b | ✅ Free tier | Very fast |
| **OpenAI** | gpt-4o-mini | ❌ Paid | Fast |

No GPU. No heavy RAM. Runs on any machine — agents run in the cloud.

---

## Requirements

- Python 3.10+
- Free API key from [Google AI Studio](https://aistudio.google.com/apikey) or [Groq](https://console.groq.com)

---

## Setup

```bash
git clone https://github.com/yourusername/autoforge
cd autoforge

python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

pip install -e .

cp .env.example .env
# Open .env and paste your API key
```

---

## Usage

**Interactive:**
```bash
python main.py
```

**One-liner:**
```bash
python main.py --project "a CLI tool that renames files in bulk"
```

**Switch model provider:**
```bash
python main.py --project "a Discord bot" --provider groq
```

**Resume an interrupted run:**
```bash
python main.py --resume 1718293847_12345
```
AutoForge prints your Run ID at the start of every run. If it crashes or gets rate-limited, resume right where it left off.

---

## Output

Each run creates a timestamped folder:
```
output/
  20240615_143022_a_cli_tool_that/
    src/
    README.md
    .env.example
    .gitignore
    REVIEW_REPORT.md    ← bugs found, fixes applied, quality score
```

---

## Examples

```bash
python main.py --project "a REST API for a blog with posts, comments, and auth"
python main.py --project "a CLI tool that watches a folder and auto-renames files"
python main.py --project "a web scraper that saves results to CSV and SQLite"
python main.py --project "a Discord bot that tracks daily standup responses"
python main.py --project "a FastAPI backend for a personal finance tracker"
```

---

## Extending AutoForge

Want more agents? Open `src/crew.py` and follow the pattern — add an `Agent`, a `Task`, include it in the `Crew`. Ideas:

- `test_writer` — generates pytest/jest tests for the output
- `devops_engineer` — writes Dockerfile + GitHub Actions deploy workflow
- `security_auditor` — dedicated OWASP Top 10 scan

---

## Development

```bash
pip install -e ".[dev]"
ruff check .        # lint
pytest              # run tests (no API key needed)
```

CI runs on every push via GitHub Actions.

---

## Environment Variables

| Variable | Required for | Where to get |
|---|---|---|
| `GEMINI_API_KEY` | Gemini (default) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | Groq | [console.groq.com](https://console.groq.com) |
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) |
| `MODEL_PROVIDER` | All | Set in `.env`: `gemini`, `groq`, or `openai` |

---

## License

MIT
