"""
Checkpoint system for AutoForge.

Saves progress after each agent completes so a crashed or rate-limited
run can resume from where it left off instead of starting over.

Checkpoint file: checkpoints/<run_id>.json
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

CHECKPOINT_DIR = Path("checkpoints")
STAGES = ["architecture", "coding", "review"]


@dataclass
class Checkpoint:
    run_id: str
    project_description: str
    output_dir: str
    provider: str
    created_at: float
    completed_stages: list = field(default_factory=list)
    stage_outputs: dict = field(default_factory=dict)
    final_result: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.final_result is not None

    @property
    def next_stage(self) -> Optional[str]:
        for stage in STAGES:
            if stage not in self.completed_stages:
                return stage
        return None

    def mark_stage_done(self, stage: str, output: str):
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.stage_outputs[stage] = output

    def mark_complete(self, final_result: str):
        self.final_result = final_result


def _checkpoint_path(run_id: str) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    return CHECKPOINT_DIR / f"{run_id}.json"


def save_checkpoint(checkpoint: Checkpoint):
    """Persist checkpoint to disk."""
    path = _checkpoint_path(checkpoint.run_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(checkpoint), f, indent=2)


def load_checkpoint(run_id: str) -> Optional[Checkpoint]:
    """Load a checkpoint by run ID. Returns None if not found."""
    path = _checkpoint_path(run_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return Checkpoint(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def find_resumable(project_description: str) -> Optional[Checkpoint]:
    """
    Look for an incomplete checkpoint for this exact project description.
    Returns the most recent one if found.
    """
    if not CHECKPOINT_DIR.exists():
        return None

    candidates = []
    for path in CHECKPOINT_DIR.glob("*.json"):
        cp = load_checkpoint(path.stem)
        if (
            cp
            and not cp.is_complete
            and cp.project_description == project_description
        ):
            candidates.append(cp)

    if not candidates:
        return None

    # Return the most recent incomplete run
    return max(candidates, key=lambda c: c.created_at)


def new_checkpoint(
    project_description: str,
    output_dir: str,
    provider: str,
) -> Checkpoint:
    """Create and save a fresh checkpoint."""
    run_id = f"{int(time.time())}_{os.getpid()}"
    cp = Checkpoint(
        run_id=run_id,
        project_description=project_description,
        output_dir=output_dir,
        provider=provider,
        created_at=time.time(),
    )
    save_checkpoint(cp)
    return cp


def delete_checkpoint(run_id: str):
    """Remove a completed checkpoint to keep the folder clean."""
    path = _checkpoint_path(run_id)
    if path.exists():
        path.unlink()
