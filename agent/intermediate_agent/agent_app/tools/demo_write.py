"""Opt-in local write tool used to demonstrate per-call approval."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from ..ingestion import DEFAULT_DATA


DEMO_NOTES_FILE = DEFAULT_DATA / "approved_demo_notes.txt"


class DemoNoteInput(BaseModel):
    note: str = Field(min_length=1, max_length=500, description="Short note to append")


@tool(args_schema=DemoNoteInput)
def save_demo_note(note: str) -> str:
    """Append a short note to the local demo notes file."""
    clean = " ".join(note.split())
    if not clean:
        raise ValueError("note must contain text")
    DEMO_NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DEMO_NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write(clean + "\n")
    return f"Saved demo note to {DEMO_NOTES_FILE}."


save_demo_note.metadata = {"data_access": "write"}
