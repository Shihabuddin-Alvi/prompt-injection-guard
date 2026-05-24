"""Loader functions for each public prompt injection dataset source."""

import os
from datasets import load_dataset, DatasetDict


def _token() -> str:
    """Read HF token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set in environment")
    return token


def load_deepset() -> DatasetDict:
    """Load deepset/prompt-injections from HF Hub."""
    return load_dataset("deepset/prompt-injections", token=_token())


def load_jasperai() -> DatasetDict:
    """Load JasperLS/prompt-injections from HF Hub."""
    return load_dataset("JasperLS/prompt-injections", token=_token())


def load_wildguard() -> DatasetDict:
    """Load WildGuardMix from HF Hub (prompt injection subset)."""
    return load_dataset("allenai/wildguardmix", "wildguardtest", token=_token())


def load_lakera() -> DatasetDict:
    """Load Lakera gandalf dataset from HF Hub."""
    return load_dataset("Lakera/gandalf_ignore_instructions", token=_token())


def load_safeguard() -> DatasetDict:
    """Load xTRam1/safe-guard-prompt-injection from HF Hub."""
    return load_dataset("xTRam1/safe-guard-prompt-injection", token=_token())
