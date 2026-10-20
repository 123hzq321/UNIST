"""Resolve the experimental data directory without relying on a local drive."""
import os
from pathlib import Path


def joint_root():
    return Path(os.environ.get("UNIST_JOINT_ROOT", "joint_training_data")).expanduser().resolve()
