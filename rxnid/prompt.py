"""Prompt templates for IdtVP inference."""

from __future__ import annotations

import json
from typing import Iterable


USER_PROMPT = "<image>\nReturn only the JSON list."


def build_system_prompt(idts: Iterable[str] | None = None) -> str:
    """Build the IdtVP system prompt with an optional identifier vocabulary."""
    idt_list = [str(item) for item in (idts or []) if str(item).strip()]
    idt_json = json.dumps(idt_list, ensure_ascii=False, separators=(",", ":"))
    return (
        "You parse chemical reaction diagrams. Return only a JSON list.\n"
        "Available IDTs (use these exact strings for structures):\n"
        f"{idt_json}\n"
        "Rules:\n"
        "- Classify into 'reactants', 'conditions', 'products'.\n"
        "- Use {\"idt\":\"<id>\"} for known structures from the IDT list; "
        "otherwise use {\"text\":\"<content>\"}.\n"
        "- Pay attention to the arrow direction in the diagram when determining roles.\n"
        "Example:\n"
        "```json\n"
        "[\n"
        "  {\"reactants\": [{\"idt\": \"E-3\"}, {\"text\": \"H2O\"}],\n"
        "   \"conditions\": [{\"text\": \"heat\"}, {\"text\": \"80%\"}],\n"
        "   \"products\": [{\"idt\": \"5\"}]}\n"
        "]\n"
        "```\n"
    )

