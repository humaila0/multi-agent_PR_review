import os
import re
import time
import logging
from typing import TypedDict, List, Dict, Optional, Set

from github import Github
from github.GithubException import GithubException
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is missing in .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env")

github_client = Github(GITHUB_TOKEN)

# -----------------------------
# LLM / LangGraph
# -----------------------------
MODEL_NAME = "gpt-4"

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# -----------------------------
# Config
# -----------------------------
MAX_TOTAL_DIFF_CHARS = 18000
MAX_PATCH_CHARS_PER_FILE = 5000
MAX_FILES_TO_REVIEW = 10

MAX_CONTEXT_FILES = 6
MAX_CONTEXT_CHARS_PER_FILE = 1500
MAX_TOTAL_CONTEXT_CHARS = 3500

BOT_COMMENT_HEADER = "## 🤖 AI Code Review"

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp4", ".mp3", ".wav", ".avi",
    ".lock", ".min.js", ".min.css",
    ".map", ".exe", ".dll", ".so", ".bin",
    ".pyc", ".pyo", ".pyd",
}
SKIP_FILE_NAMES = {
    "readme.md",
    "readme.rst",
    "changelog.md",
    "contributing.md",
    "license",
    "license.md",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "composer.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    ".gitignore",
    ".env",
    ".env.local",
    ".env.example",
    ".env.production",
    ".env.staging",
}
SKIP_DIRS = {
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".git",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    "migrations",
    ".idea",
    ".vscode",
}

COMMON_CONTEXT_WORDS = {
    "src", "app", "main", "core", "test", "tests", "utils",
    "common", "index", "lib", "config", "models", "views"
}

# -----------------------------
# Helpers
# -----------------------------
def should_skip_file(filename: str) -> bool:
    lower_name = filename.lower()
    parts = lower_name.split("/")

    for part in parts[:-1]:
        if part in SKIP_DIRS:
            return True

    if parts[-1] in SKIP_FILE_NAMES:
        return True

    for ext in SKIP_EXTENSIONS:
        if lower_name.endswith(ext):
            return True

    if "dist/" in lower_name or "build/" in lower_name or "node_modules/" in lower_name:
        return True

    return False


def clean_patch(patch: str, max_chars: int = MAX_PATCH_CHARS_PER_FILE) -> str:
    patch = patch.strip()
    if len(patch) <= max_chars:
        return patch
    return patch[:max_chars] + "\n... [PATCH TRUNCATED]"


def format_files_for_prompt(files_data: List[Dict]) -> str:
    parts = []

    for item in files_data:
        parts.append(
            f"File: {item['filename']}\n"
            f"Status: {item['status']}\n"
            f"Additions: {item['additions']} | Deletions: {item['deletions']}\n"
            f"Patch:\n{item['patch']}\n"
            f"{'-' * 80}"
        )

    text = "\n\n".join(parts)

    if len(text) > MAX_TOTAL_DIFF_CHARS:
        text = text[:MAX_TOTAL_DIFF_CHARS] + "\n\n... [TOTAL DIFF TRUNCATED]"

    return text


def extract_keywords_from_filename(filename: str) -> Set[str]:
    parts = re.split(r"[\/._\-]", filename)
    keywords = set()

    for part in parts:
        part = part.lower().strip()
        if len(part) > 3 and part not in COMMON_CONTEXT_WORDS:
            keywords.add(part)

    return keywords




def extract_changed_symbols_from_patch(patch_text: str) -> Set[str]:
    """
    Extract likely changed symbols from +/- diff lines.
    Focus on constants, identifiers, imports, and function names.
    """
    symbols = set()

    for line in (patch_text or "").splitlines():
        if not line.startswith("+") and not line.startswith("-"):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue

        # UPPER_CASE constants like API_TIMEOUT
        symbols.update(re.findall(r"\b[A-Z_]{3,}\b", line))

        # imported names / identifiers
        symbols.update(
            word for word in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", line)
            if word not in {
                "from", "import", "return", "def", "class", "for", "while", "if",
                "else", "elif", "try", "except", "with", "True", "False", "None"
            }
        )

    return symbols


def is_relevant_file(changed_file: str, candidate_file: str, candidate_content: str, changed_symbols: Set[str]) -> bool:
    """
    Include candidate files only if they are actually relevant to the changed file.

    Rules:
    - Matching test files are relevant
    - Shared filename keywords are relevant
    - Files containing changed symbols are relevant
    - Files importing / using changed module name are relevant
    """
    changed_keywords = extract_keywords_from_filename(changed_file)
    candidate_keywords = extract_keywords_from_filename(candidate_file)

    changed_base = changed_file.split("/")[-1].rsplit(".", 1)[0].lower()
    candidate_lower = candidate_file.lower()
    content_lower = (candidate_content or "").lower()

    # Matching tests remain relevant
    if f"test_{changed_base}" in candidate_lower or f"{changed_base}_test" in candidate_lower:
        return True

    # Shared filename keywords
    if changed_keywords & candidate_keywords:
        return True

    # Candidate imports/uses changed module name
    module_patterns = [
        f"import {changed_base}",
        f"from {changed_base} import",
        changed_base,
    ]
    if any(pattern in content_lower for pattern in module_patterns):
        return True

    # Candidate uses changed symbols like API_TIMEOUT / REQUEST_TIMEOUT
    for symbol in changed_symbols:
        if symbol and symbol in candidate_content:
            return True

    return False

# -----------------------------
# Phase 1 - PR Diff
# -----------------------------
def get_pr_data(repo_name: str, pr_number: int) -> Dict:
    repo = github_client.get_repo(repo_name)
    pull_request = repo.get_pull(pr_number)
    files = pull_request.get_files()

    selected_files = []
    reviewed_count = 0

    for file in files:
        if reviewed_count >= MAX_FILES_TO_REVIEW:
            logger.info("Max reviewable files reached, skipping remaining files")
            break

        if should_skip_file(file.filename):
            logger.info("Skipping file: %s", file.filename)
            continue

        if not file.patch:
            logger.info("Skipping file with no patch: %s", file.filename)
            continue

        selected_files.append({
            "filename": file.filename,
            "status": file.status,
            "additions": file.additions,
            "deletions": file.deletions,
            "patch": clean_patch(file.patch)
        })
        reviewed_count += 1

    return {
        "repo": repo,
        "pull_request": pull_request,
        "title": pull_request.title or f"PR #{pr_number}",
        "files_data": selected_files
    }


# -----------------------------
# Phase 2 - Repo Context
# -----------------------------
def get_same_directory_candidates(repo, changed_file: str) -> List[str]:
    directory = changed_file.rsplit("/", 1)[0] if "/" in changed_file else ""
    candidates = []

    try:
        items = repo.get_contents(directory) if directory else repo.get_contents("")
        for item in items:
            if item.type == "file" and item.path != changed_file and not should_skip_file(item.path):
                candidates.append(item.path)
    except GithubException:
        pass

    return candidates


def search_code_paths(repo, search_terms: List[str], changed_file: str, max_per_term: int = 5) -> List[str]:
    candidates = []
    seen = set()

    for term in search_terms:
        try:
            query = f"{term} repo:{repo.full_name}"
            logger.info("GitHub code search query: %s", query)

            code_results = github_client.search_code(query)
            count = 0

            for item in code_results:
                path = item.path

                if path == changed_file:
                    continue
                if should_skip_file(path):
                    continue
                if path in seen:
                    continue

                candidates.append(path)
                seen.add(path)
                count += 1

                if count >= max_per_term:
                    break

            time.sleep(1)

        except Exception as e:
            logger.warning("Code search failed for term '%s': %s", term, str(e))
            continue

    return candidates


def get_similar_filename_candidates(repo, changed_file: str) -> List[str]:
    filename_only = changed_file.split("/")[-1]
    basename = filename_only.rsplit(".", 1)[0].lower()
    keywords = extract_keywords_from_filename(changed_file)

    search_terms = [basename] + list(keywords)[:2]
    return search_code_paths(repo, search_terms, changed_file, max_per_term=5)


def get_test_file_candidates(repo, changed_file: str) -> List[str]:
    filename_only = changed_file.split("/")[-1]
    basename = filename_only.rsplit(".", 1)[0].lower()

    possible_tests = [
        f"test_{basename}",
        f"{basename}_test",
        basename
    ]

    results = search_code_paths(repo, possible_tests, changed_file, max_per_term=4)

    filtered = []
    for path in results:
        lower_path = path.lower()
        if "test" in lower_path or "tests/" in lower_path:
            filtered.append(path)

    return filtered


def fetch_context_file_content(repo, filepath: str) -> Optional[str]:
    try:
        file_obj = repo.get_contents(filepath)

        if isinstance(file_obj, list):
            return None

        content = file_obj.decoded_content.decode("utf-8", errors="ignore").strip()

        if not content:
            return None

        return content[:MAX_CONTEXT_CHARS_PER_FILE]

    except Exception as e:
        logger.warning("Failed to fetch context file %s: %s", filepath, str(e))
        return None


def get_repo_context(repo, files_data: List[Dict]) -> str:
    context_blocks = []
    added_paths = set()
    total_context_chars = 0

    for file_data in files_data:
        changed_file = file_data["filename"]
        changed_symbols = extract_changed_symbols_from_patch(file_data.get("patch", ""))

        same_dir = get_same_directory_candidates(repo, changed_file)
        similar = get_similar_filename_candidates(repo, changed_file)
        tests = get_test_file_candidates(repo, changed_file)

        ranked_candidates = same_dir + similar + tests

        for candidate in ranked_candidates:
            if candidate == changed_file:
                continue
            if candidate in added_paths:
                continue

            content = fetch_context_file_content(repo, candidate)
            if not content:
                continue

            if not is_relevant_file(changed_file, candidate, content, changed_symbols):
                logger.info("Skipping symbol-unrelated file: %s", candidate)
                continue

            block = f"--- Related File: {candidate} ---\n{content}\n"

            if total_context_chars + len(block) > MAX_TOTAL_CONTEXT_CHARS:
                remaining = MAX_TOTAL_CONTEXT_CHARS - total_context_chars
                if remaining > 200:
                    context_blocks.append(block[:remaining] + "\n... [CONTEXT TRUNCATED]")
                return "\n\n".join(context_blocks)

            context_blocks.append(block)
            added_paths.add(candidate)
            total_context_chars += len(block)

            if len(context_blocks) >= MAX_CONTEXT_FILES:
                return "\n\n".join(context_blocks)

    return "\n\n".join(context_blocks)


# -----------------------------
# Phase 3 - Multi-Agent Review
# -----------------------------
def compress_diff(diff_text: str) -> str:
    lines = diff_text.split('\n')
    compressed = []

    for line in lines:
        if line.startswith('File:') or \
           line.startswith('@@') or \
           line.startswith('---') or \
           line.startswith('+++') or \
           line.startswith('diff ') or \
           line.startswith('Status:') or \
           line.startswith('Additions:'):
            compressed.append(line)
        elif line.startswith('+') or line.startswith('-'):
            compressed.append(line)
        elif line.strip() and (set(line.strip()) == {'-'} or set(line.strip()) == {'='}):
            compressed.append(line)

    result = '\n'.join(compressed)

    logger.info(
        "Diff compressed: %d chars -> %d chars (%.0f%% reduction)",
        len(diff_text),
        len(result),
        (1 - len(result) / max(len(diff_text), 1)) * 100
    )

    return result


class ReviewState(TypedDict):
    repo_name: str
    pr_number: int
    pr_title: str
    diff_text: str
    compressed_diff: str
    repo_context: str
    security_review: str
    performance_review: str
    maintainability_review: str
    final_review: str


def security_agent(state: ReviewState) -> ReviewState:
    logger.info("Phase 3 | Security Agent running...")

    prompt = f"""Security review only. Be concise.

Find ONLY these issues:
- Hardcoded secrets, passwords, API keys
- SQL/command injection risks
- Missing input validation
- Auth/authorization flaws
- Sensitive data exposure
- Insecure dependencies

Diff:
{state["compressed_diff"]}

Format each issue as:
[HIGH/MED/LOW] File: problem — fix

If none: write "No security issues found."
Max 5 issues."""

    response = llm.invoke(prompt)
    state["security_review"] = response.content.strip()

    logger.info("Phase 3 | Security Agent done | output: %d chars", len(state["security_review"]))
    return state


def performance_agent(state: ReviewState) -> ReviewState:
    logger.info("Phase 3 | Performance Agent running...")

    prompt = f"""Performance review only. Be concise.

Find ONLY these issues:
- Unnecessary loops or nested loops
- N+1 database query patterns
- Missing caching opportunities
- Blocking/synchronous operations
- Memory leaks or large allocations
- Slow algorithms (O(n²) etc.)

Diff:
{state["compressed_diff"]}

Format each issue as:
[HIGH/MED/LOW] File: problem — fix

If none: write "No performance issues found."
Max 5 issues."""

    response = llm.invoke(prompt)
    state["performance_review"] = response.content.strip()

    logger.info("Phase 3 | Performance Agent done | output: %d chars", len(state["performance_review"]))
    return state


def maintainability_agent(state: ReviewState) -> ReviewState:
    logger.info("Phase 3 | Maintainability Agent running...")

    context_section = ""
    if state["repo_context"]:
        context_section = f"""
Related files for context:
{state["repo_context"]}

IMPORTANT: Check if changed function/variable/config names
still match what related files expect.
"""

    prompt = f"""Maintainability review only. Be concise.

Find ONLY these issues:
- Breaking changes (renamed vars, functions, configs)
- Callers/importers using old names (check related files!)
- Missing or outdated tests
- Code duplication
- Missing error handling
- Overly complex functions

STRICT RULES:
- ONLY report issues based on the provided diff and context
- DO NOT assume removal or renaming unless clearly visible in the diff
- If a function still exists, DO NOT report it as removed
- DO NOT guess changes in other files unless explicitly shown or referenced
- If you are not sure an issue is real, do NOT report it
- Only report missing tests if the changed behavior clearly requires new or updated tests

Diff:
{state["compressed_diff"]}
{context_section}

Format each issue as:
[HIGH/MED/LOW] File: problem — fix

If none: write "No maintainability issues found."
Max 5 issues."""

    response = llm.invoke(prompt)
    state["maintainability_review"] = response.content.strip()

    logger.info("Phase 3 | Maintainability Agent done | output: %d chars", len(state["maintainability_review"]))
    return state


def judge_agent(state: ReviewState) -> ReviewState:
    logger.info("Phase 3 | Judge Agent running...")

    prompt = f"""You are a senior engineering lead.
Combine these 3 specialist reviews into ONE final PR review.
CRITICAL RULE:
- Remove any issue that is not directly supported by the diff or context
- Do not include speculative or assumed issues

PR: {state["pr_title"]} (#{state["pr_number"]})
Repo: {state["repo_name"]}

Security Review:
{state["security_review"]}

Performance Review:
{state["performance_review"]}

Maintainability Review:
{state["maintainability_review"]}

Instructions:
- Remove duplicates
- Sort by severity (High first)
- Be specific and actionable
- Mention exact file names

Return in this EXACT markdown format:

### Summary
- 2 to 4 concise bullets covering the key changes and risks

### Issues
For each issue:
- Severity: High / Medium / Low
- File: <filename>
- Problem: <short title>
- Why it matters: <1-2 lines>
- Suggestion: <1-2 lines>

If no issues: write "- No major issues found."

### Test Gaps
- Brief mention of missing or weak tests

### Verdict
Choose exactly one:
- ✅ Looks Good
- ⚠️ Minor Issues
- ❌ Needs Changes"""

    response = llm.invoke(prompt)
    state["final_review"] = response.content.strip()

    logger.info("Phase 3 | Judge Agent done | output: %d chars", len(state["final_review"]))
    return state


def build_review_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("security", security_agent)
    graph.add_node("performance", performance_agent)
    graph.add_node("maintainability", maintainability_agent)
    graph.add_node("judge", judge_agent)

    graph.set_entry_point("security")
    graph.add_edge("security", "performance")
    graph.add_edge("performance", "maintainability")
    graph.add_edge("maintainability", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


review_graph = build_review_graph()


def run_multi_agent_review(
    repo_name: str,
    pr_number: int,
    pr_title: str,
    diff_text: str,
    repo_context: str = ""
) -> str:
    logger.info("Phase 3 | Starting multi-agent review...")

    compressed = compress_diff(diff_text)
    logger.info("Phase 3 | Compressed diff ready (%d chars)", len(compressed))

    initial_state: ReviewState = {
        "repo_name": repo_name,
        "pr_number": pr_number,
        "pr_title": pr_title,
        "diff_text": diff_text,
        "compressed_diff": compressed,
        "repo_context": repo_context,
        "security_review": "",
        "performance_review": "",
        "maintainability_review": "",
        "final_review": "",
    }

    result = review_graph.invoke(initial_state)

    logger.info("Phase 3 | Multi-agent review complete!")
    return result["final_review"]


def review_with_openai(
    repo_name: str,
    pr_number: int,
    pr_title: str,
    files_data: List[Dict],
    repo_context: str = ""
) -> str:
    diff_text = format_files_for_prompt(files_data)

    logger.info("Phase 3 | Files selected for review: %s", [f["filename"] for f in files_data])
    logger.info("Phase 3 | Repo context length: %d chars", len(repo_context))
    logger.info("Phase 3 | Diff length: %d chars", len(diff_text))

    final_review = run_multi_agent_review(
        repo_name=repo_name,
        pr_number=pr_number,
        pr_title=pr_title,
        diff_text=diff_text,
        repo_context=repo_context
    )

    return final_review


# -----------------------------
# Comment Handling
# -----------------------------
def build_comment_body(review_text: str, reviewed_files: List[Dict]) -> str:
    file_list = "\n".join([f"- `{item['filename']}`" for item in reviewed_files])

    return f"""{BOT_COMMENT_HEADER}

{review_text}

---

### Reviewed Files
{file_list if file_list else "- No reviewable files found"}

_Auto-generated by AI PR Review Agent (Phase 3 — Multi-Agent)_
"""


def no_files_comment_body() -> str:
    return f"""{BOT_COMMENT_HEADER}

### Summary
- No reviewable code diff was found for this PR.

### Issues
- No major issues found in the visible diff.

### Test Gaps
- Not enough code changes to assess test coverage.

### Verdict
✅ Looks Good

---

_Auto-generated by AI PR Review Agent_
"""


def find_existing_bot_comment(pull_request) -> Optional[object]:
    try:
        comments = pull_request.get_issue_comments()
        for comment in comments:
            if comment.body and comment.body.startswith(BOT_COMMENT_HEADER):
                return comment
    except GithubException as e:
        logger.warning("Failed to fetch existing comments: %s", str(e))

    return None


def upsert_pr_comment(pull_request, comment_body: str) -> None:
    existing_comment = find_existing_bot_comment(pull_request)

    if existing_comment:
        logger.info("Updating existing bot comment")
        existing_comment.edit(comment_body)
    else:
        logger.info("Creating new bot comment")
        pull_request.create_issue_comment(comment_body)


# -----------------------------
# Main Entry Point
# -----------------------------
def review_pr(repo_name: str, pr_number: int) -> None:
    logger.info("Starting PR review | repo=%s | pr=%s", repo_name, pr_number)

    try:
        pr_data = get_pr_data(repo_name, pr_number)
        repo = pr_data["repo"]
        pull_request = pr_data["pull_request"]
        pr_title = pr_data["title"]
        files_data = pr_data["files_data"]

        if not files_data:
            logger.info("No reviewable files found | repo=%s | pr=%s", repo_name, pr_number)
            upsert_pr_comment(pull_request, no_files_comment_body())
            return

        logger.info("Phase 2 | Fetching repo context...")
        repo_context = get_repo_context(repo, files_data)

        if repo_context:
            logger.info("Phase 3 | Context found, starting multi-agent review...")
        else:
            logger.info("Phase 3 | No context found, starting multi-agent review (diff only)...")

        review_text = review_with_openai(
            repo_name=repo_name,
            pr_number=pr_number,
            pr_title=pr_title,
            files_data=files_data,
            repo_context=repo_context
        )

        comment_body = build_comment_body(review_text, files_data)
        upsert_pr_comment(pull_request, comment_body)

        logger.info("PR review finished successfully | repo=%s | pr=%s", repo_name, pr_number)

    except GithubException as e:
        logger.exception("GitHub API error | repo=%s | pr=%s | error=%s", repo_name, pr_number, str(e))
        raise
    except Exception as e:
        logger.exception("Unexpected error | repo=%s | pr=%s | error=%s", repo_name, pr_number, str(e))
        raise


if __name__ == "__main__":
    # Replace with your actual repo and PR number when testing
    # Example:
    # review_pr("your-org/your-repo", 123)
    print("Single-file PR review agent loaded. Call review_pr('owner/repo', pr_number)")
