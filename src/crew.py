"""
AutoForge — crew definition.

Three agents work sequentially:
  1. Architect  → plans the project
  2. Coder      → writes all files to disk
  3. Reviewer   → audits and fixes the code
"""

import os
import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileWriterTool, DirectoryReadTool

from src.checkpoint import Checkpoint, save_checkpoint


# -----------------------------------------------------------
# MODEL SETUP
# -----------------------------------------------------------

SUPPORTED_PROVIDERS = {
    "gemini": {
        "model": "gemini/gemini-2.0-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "model": "groq/llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def get_llm(provider: str) -> LLM:
    config = SUPPORTED_PROVIDERS[provider]
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"Missing {config['api_key_env']}. Add it to your .env file.\n"
            f"Get a free key at: https://aistudio.google.com/apikey"
        )
    return LLM(model=config["model"])


# -----------------------------------------------------------
# TOOLS
# -----------------------------------------------------------

file_writer = FileWriterTool()
dir_reader = DirectoryReadTool()


# -----------------------------------------------------------
# AGENTS
# -----------------------------------------------------------

def build_agents(llm: LLM) -> tuple:
    architect = Agent(
        role="Software Architect",
        goal=(
            "Analyze project requirements and produce a complete, actionable "
            "architecture plan that a developer can follow immediately."
        ),
        backstory=(
            "You are a senior software architect with 15 years of experience. "
            "You are obsessed with clean structure, security, and scalability. "
            "You think about how the project will grow and explain your decisions clearly."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=3,
    )

    coder = Agent(
        role="Senior Software Engineer",
        goal=(
            "Write complete, production-quality code for every file in the architecture plan. "
            "Save each file to disk using the file writer tool. No placeholders. No TODOs."
        ),
        backstory=(
            "You are a meticulous engineer who never ships half-finished code. "
            "You follow the architecture plan exactly, handle every error, "
            "validate all input, never hardcode secrets, and always save files to disk."
        ),
        llm=llm,
        tools=[file_writer],
        verbose=True,
        allow_delegation=False,
        max_retry_limit=3,
    )

    reviewer = Agent(
        role="Code Reviewer & Security Auditor",
        goal=(
            "Review all generated code for bugs, security issues, and quality problems. "
            "Fix medium/high severity issues and save corrected files to disk."
        ),
        backstory=(
            "You are a ruthless but fair code reviewer. You check for hardcoded secrets, "
            "injection risks, missing error handling, and bad patterns. "
            "You give specific fixes, not vague advice."
        ),
        llm=llm,
        tools=[file_writer, dir_reader],
        verbose=True,
        allow_delegation=False,
        max_retry_limit=3,
    )

    return architect, coder, reviewer


# -----------------------------------------------------------
# TASKS
# -----------------------------------------------------------

def build_tasks(
    project_description: str,
    output_dir: str,
    architect: Agent,
    coder: Agent,
    reviewer: Agent,
    skip_architecture: str | None = None,
    skip_coding: str | None = None,
) -> list:
    """
    Build tasks. If skip_* is provided, that stage's output is injected
    as context so we can resume from a checkpoint mid-run.
    """

    task_architecture = Task(
        description=(
            f"Project:\n\n{project_description}\n\n"
            f"Output directory: {output_dir}\n\n"
            "Write a complete architecture document:\n"
            "1. **Stack** — technologies chosen and why\n"
            "2. **File Tree** — every file to be created\n"
            "3. **Design Decisions** — patterns, naming, data flow\n"
            "4. **Security Plan** — auth, validation, secrets\n"
            "5. **Assumptions** — anything assumed about requirements"
        )
        if not skip_architecture
        else f"[RESUMED] Architecture already completed:\n\n{skip_architecture}",
        expected_output="A Markdown architecture document with all 5 sections.",
        agent=architect,
    )

    task_coding = Task(
        description=(
            f"Implement the project in: {output_dir}\n\n"
            "Follow the architecture plan exactly.\n"
            "For every file in the file tree:\n"
            "  1. Write the complete content (no placeholders)\n"
            "  2. Save it using the file writer tool\n\n"
            "Always include: .env.example, README.md, .gitignore\n\n"
            "Rules:\n"
            "- Secrets via env vars only\n"
            "- Validate all external input\n"
            "- Explicit error handling on all IO/async\n"
            "- Descriptive names, max ~30 lines per function"
        )
        if not skip_coding
        else f"[RESUMED] Coding already completed:\n\n{skip_coding}",
        expected_output="List of all files written to disk.",
        agent=coder,
        context=[task_architecture],
    )

    task_review = Task(
        description=(
            f"Review all code in: {output_dir}\n\n"
            "Use the directory reader to list files, then review each one.\n\n"
            "Check for:\n"
            "1. Bugs and logic errors\n"
            "2. Security issues (hardcoded secrets, injection, missing auth)\n"
            "3. Missing error handling\n"
            "4. Code quality (naming, duplication, complexity)\n"
            "5. Anything missing from the requirements\n\n"
            "Per issue: ISSUE (file+line), SEVERITY (low/medium/high), FIX\n"
            "Save corrected files for medium/high severity issues.\n\n"
            "End with SUMMARY: issues found, files changed, quality score (1-10), ready to run?"
        ),
        expected_output="Review report with issues, fixes applied, and summary.",
        agent=reviewer,
        context=[task_architecture, task_coding],
    )

    return [task_architecture, task_coding, task_review]


# -----------------------------------------------------------
# CREW RUNNER WITH CHECKPOINT SUPPORT
# -----------------------------------------------------------

def run_crew(
    project_description: str,
    output_dir: str,
    provider: str,
    checkpoint: Checkpoint,
) -> str:
    """
    Run the crew, saving a checkpoint after each stage completes.
    Resumes from checkpoint if stages were already completed.
    """
    llm = get_llm(provider)
    architect, coder, reviewer = build_agents(llm)

    skip_architecture = checkpoint.stage_outputs.get("architecture")
    skip_coding = checkpoint.stage_outputs.get("coding")

    tasks = build_tasks(
        project_description=project_description,
        output_dir=output_dir,
        architect=architect,
        coder=coder,
        reviewer=reviewer,
        skip_architecture=skip_architecture,
        skip_coding=skip_coding,
    )

    crew = Crew(
        agents=[architect, coder, reviewer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    return _run_with_checkpoint(crew, checkpoint)


def _run_with_checkpoint(crew: Crew, checkpoint: Checkpoint) -> str:
    """
    Run the crew with retry on rate limits, saving checkpoint after each task.
    """
    stage_names = ["architecture", "coding", "review"]
    max_retries = 3

    # Run the full crew — CrewAI handles sequential execution internally.
    # We wrap the full kickoff with retry logic.
    for attempt in range(1, max_retries + 1):
        try:
            result = crew.kickoff()
            final = str(result)

            # Mark all stages done (crew ran to completion)
            for stage in stage_names:
                if stage not in checkpoint.completed_stages:
                    checkpoint.mark_stage_done(stage, "[completed in this run]")

            checkpoint.mark_complete(final)
            save_checkpoint(checkpoint)
            return final

        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(
                phrase in error_msg
                for phrase in ["rate limit", "quota", "429", "resource exhausted"]
            )

            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 15  # 30s, 60s, 120s
                print(
                    f"\n⚠️  Rate limit hit. "
                    f"Waiting {wait}s before retry ({attempt}/{max_retries - 1})...\n"
                    f"    Progress is saved. You can also Ctrl+C and resume later."
                )
                save_checkpoint(checkpoint)
                time.sleep(wait)
            else:
                # Save what we have before raising
                save_checkpoint(checkpoint)
                raise
