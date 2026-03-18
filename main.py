#!/usr/bin/env python3
"""
AutoForge — Multi-Agent Coding Crew
-------------------------------------
Describe any software project. Three AI agents plan it,
build it, and review it — autonomously.

Usage:
  python main.py
  python main.py --project "a REST API for a todo list with auth"
  python main.py --project "a Discord bot" --provider groq
  python main.py --resume <run_id>
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.checkpoint import (
    delete_checkpoint,
    find_resumable,
    load_checkpoint,
    new_checkpoint,
)
from src.crew import run_crew
from src.validate import ValidationError, validate_project_description, validate_provider

load_dotenv()


# -----------------------------------------------------------
# ENV VALIDATION
# -----------------------------------------------------------

def validate_env(provider: str):
    key_map = {
        "gemini": ("GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
        "groq": ("GROQ_API_KEY", "https://console.groq.com"),
        "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    }
    key_name, key_url = key_map[provider]
    if not os.getenv(key_name):
        print(f"\n❌  {key_name} is not set!")
        print(f"    Get a free key at: {key_url}")
        print(f"    Then add to .env:  {key_name}=your_key_here\n")
        sys.exit(1)


# -----------------------------------------------------------
# OUTPUT DIR
# -----------------------------------------------------------

def make_output_dir(project_description: str) -> str:
    slug = "_".join(project_description.split()[:5]).lower()
    slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"{timestamp}_{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


# -----------------------------------------------------------
# INPUT PROMPT
# -----------------------------------------------------------

def prompt_project_description() -> str:
    print("\n⚒️   AutoForge — Multi-Agent Coding Crew")
    print("=" * 50)
    print("  🏗️  Architect  → plans the structure & stack")
    print("  💻  Coder      → writes all files to disk")
    print("  🔍  Reviewer   → finds bugs & fixes them")
    print("=" * 50)
    print("\nDescribe the project you want to build.")
    print("(Press Enter twice when done)\n")

    lines = []
    while True:
        try:
            line = input("> " if not lines else "  ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()


# -----------------------------------------------------------
# RESUME FLOW
# -----------------------------------------------------------

def handle_resume(run_id: str | None, project_description: str | None) -> tuple:
    """
    Returns (checkpoint, project_description, output_dir) for a resumed run,
    or (None, None, None) if nothing to resume.
    """
    if run_id:
        cp = load_checkpoint(run_id)
        if not cp:
            print(f"❌  No checkpoint found for run ID: {run_id}")
            sys.exit(1)
        print(f"\n🔄  Resuming run {run_id}...")
        print(f"    Project: {cp.project_description[:80]}")
        print(f"    Completed stages: {', '.join(cp.completed_stages) or 'none'}")
        return cp, cp.project_description, cp.output_dir

    if project_description:
        cp = find_resumable(project_description)
        if cp:
            print(f"\n🔄  Found an incomplete run for this project (ID: {cp.run_id})")
            print(f"    Completed stages: {', '.join(cp.completed_stages) or 'none'}")
            answer = input("    Resume it? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                return cp, cp.project_description, cp.output_dir

    return None, None, None


# -----------------------------------------------------------
# SAVE REPORT
# -----------------------------------------------------------

def save_report(output_dir: str, result: str, project_description: str):
    report_path = Path(output_dir) / "REVIEW_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# AutoForge Review Report\n\n")
        f.write(f"**Project:** {project_description}\n\n")
        f.write("---\n\n")
        f.write(result)
    return str(report_path)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AutoForge — describe a project, get working code"
    )
    parser.add_argument("--project", type=str, help="Project description")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "groq", "openai"],
        default=None,
        help="AI model provider (overrides .env)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="RUN_ID",
        help="Resume a specific previous run by ID",
    )
    args = parser.parse_args()

    # Provider resolution: CLI flag > .env > default
    provider_raw = args.provider or os.getenv("MODEL_PROVIDER", "gemini")
    try:
        provider = validate_provider(provider_raw)
    except ValidationError as e:
        print(f"\n❌  {e}")
        sys.exit(1)

    validate_env(provider)

    # Get project description
    raw_description = args.project or (None if args.resume else prompt_project_description())

    # Validate description (skip if resuming by run_id only)
    project_description = None
    if raw_description:
        try:
            project_description = validate_project_description(raw_description)
        except ValidationError as e:
            print(f"\n❌  {e}\n")
            sys.exit(1)

    # Check for resume
    checkpoint, project_description, output_dir = handle_resume(
        args.resume, project_description
    )

    # Fresh run
    if not checkpoint:
        if not project_description:
            print("❌  No project description provided.")
            sys.exit(1)
        output_dir = make_output_dir(project_description)
        checkpoint = new_checkpoint(project_description, output_dir, provider)

    print(f"\n📁  Output directory: {output_dir}")
    print(f"🆔  Run ID: {checkpoint.run_id}  (use --resume {checkpoint.run_id} if interrupted)")
    print(f"🚀  Starting AutoForge for: {project_description[:80]}...")
    print("    Takes ~5–10 min. If interrupted, resume with the Run ID above.\n")

    try:
        result = run_crew(
            project_description=project_description,
            output_dir=output_dir,
            provider=provider,
            checkpoint=checkpoint,
        )
    except KeyboardInterrupt:
        print(f"\n\n⏸️   Interrupted. Resume later with:")
        print(f"    python main.py --resume {checkpoint.run_id}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Run failed: {e}")
        print(f"    Resume with: python main.py --resume {checkpoint.run_id}\n")
        sys.exit(1)

    report_path = save_report(output_dir, result, project_description)
    delete_checkpoint(checkpoint.run_id)

    print("\n" + "=" * 50)
    print("🎉  AutoForge complete!")
    print("=" * 50)
    print(f"📁  Project files : {output_dir}/")
    print(f"📋  Review report : {report_path}")
    print(f"\n    cd {output_dir} && cat README.md")


if __name__ == "__main__":
    main()
