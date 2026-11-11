"""Data structures for persistent conversation memory.

The data structures and local store also preserve LangChain messages so an
agent can resume with native tool calls and tool results intact.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4
import warnings

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
    messages_from_dict, messages_to_dict,
)

from .ingestion import DEFAULT_DATA


MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class MessageRecord:
    """One conversation message in a JSON-compatible form."""

    role: str
    content: str
    message_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=_utc_now)
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.role not in MESSAGE_ROLES:
            raise ValueError(f"Unsupported message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("Message content must be a string")
        if not self.message_id.strip():
            raise ValueError("message_id must not be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._schema_version,
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MessageRecord":
        if not isinstance(value, dict):
            raise TypeError("A message must be an object")
        version = value.get("schema_version", 1)
        if version != cls._schema_version:
            raise ValueError(f"Unsupported message schema version: {version}")
        required = ("role", "content", "message_id", "timestamp")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"Message is missing fields: {', '.join(missing)}")
        return cls(
            role=value["role"],
            content=value["content"],
            message_id=value["message_id"],
            timestamp=value["timestamp"],
            name=value.get("name"),
            tool_call_id=value.get("tool_call_id"),
            metadata=dict(value.get("metadata") or {}),
        )

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> "MessageRecord":
        """Keep the full framework message alongside readable fields."""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            raise TypeError(f"Unsupported LangChain message: {type(message).__name__}")
        return cls(
            role=role,
            content=message.text,
            name=message.name,
            tool_call_id=message.tool_call_id if isinstance(message, ToolMessage) else None,
            metadata={"langchain_message": messages_to_dict([message])[0]},
        )

    def to_langchain(self) -> BaseMessage:
        """Restore a native message, including assistant tool-call arguments."""
        serialized = self.metadata.get("langchain_message")
        if serialized is not None:
            message = messages_from_dict([serialized])[0]
            if not isinstance(message, _ROLE_TO_MESSAGE[self.role]):
                raise ValueError("Saved LangChain message role does not match the record")
            return message
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("A saved tool message needs a tool_call_id")
            return ToolMessage(content=self.content, tool_call_id=self.tool_call_id, name=self.name)
        return _ROLE_TO_MESSAGE[self.role](content=self.content, name=self.name)


@dataclass
class SessionState:
    """Conversation state identified by a caller-provided session ID."""

    session_id: str
    messages: list[MessageRecord] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    _schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(self.messages, list) or not all(isinstance(item, MessageRecord) for item in self.messages):
            raise TypeError("messages must be a list of MessageRecord objects")

    def add(self, message: MessageRecord) -> MessageRecord:
        if not isinstance(message, MessageRecord):
            raise TypeError("message must be a MessageRecord")
        self.messages.append(message)
        self.updated_at = _utc_now()
        return message

    def langchain_messages(self) -> list[BaseMessage]:
        """Restore the ordered history for a framework agent invocation."""
        return [message.to_langchain() for message in self.messages]

    def replace_messages(self, messages: list[MessageRecord]) -> None:
        """Record a completed agent turn as one coherent session snapshot."""
        if not isinstance(messages, list) or not all(isinstance(item, MessageRecord) for item in messages):
            raise TypeError("messages must be a list of MessageRecord objects")
        self.messages = messages
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._schema_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionState":
        if not isinstance(value, dict):
            raise TypeError("A session must be an object")
        version = value.get("schema_version", 1)
        if version != cls._schema_version:
            raise ValueError(f"Unsupported session schema version: {version}")
        if "session_id" not in value:
            raise ValueError("Session is missing session_id")
        messages = [MessageRecord.from_dict(item) for item in value.get("messages", [])]
        return cls(
            session_id=value["session_id"],
            messages=messages,
            created_at=value.get("created_at", _utc_now()),
            updated_at=value.get("updated_at", _utc_now()),
        )


class SessionStore:
    """Save one JSON file per session under the local data directory.

    Session IDs are hashed for filenames, so an ID cannot become a path. Writes
    replace a complete file atomically; callers should serialize concurrent
    writes to the same session.
    """

    def __init__(self, directory: Path | str = DEFAULT_DATA / "sessions", *,
                 max_turns: int = 20, max_age_days: int | None = 30) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if max_age_days is not None and (
            isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or max_age_days < 1
        ):
            raise ValueError("max_age_days must be a positive integer or None")
        self.directory = Path(directory).resolve()
        self.max_turns = max_turns
        self.max_age_days = max_age_days

    def _path(self, session_id: str) -> Path:
        SessionState(session_id)  # Reuse the session ID validation.
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def load(self, session_id: str) -> SessionState:
        """Return saved history, archiving damaged or expired files first."""
        path = self._path(session_id)
        if not path.exists():
            return SessionState(session_id)
        try:
            session = SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if session.session_id != session_id:
                raise ValueError("Session ID does not match its filename")
            session.langchain_messages()  # Detect damaged framework message data.
            if not isinstance(session.updated_at, str):
                raise ValueError("Session updated_at must be a timestamp string")
            updated = datetime.fromisoformat(session.updated_at.replace("Z", "+00:00"))
            if updated.tzinfo is None:
                raise ValueError("Session updated_at needs a timezone")
        except (ValueError, TypeError, KeyError) as error:
            self._archive(path, "corrupt", f"Session {session_id!r} could not be read: {error}")
            return SessionState(session_id)
        if self.max_age_days is not None and datetime.now(timezone.utc) - updated > timedelta(days=self.max_age_days):
            self._archive(path, "expired", f"Session {session_id!r} expired")
            return SessionState(session_id)
        return session

    @staticmethod
    def _archive(path: Path, reason: str, message: str) -> None:
        archived = path.with_name(f"{path.stem}.{reason}.{uuid4().hex}.json")
        path.replace(archived)
        warnings.warn(f"{message}; previous file retained at {archived}", RuntimeWarning, stacklevel=2)

    def _limited_messages(self, messages: list[MessageRecord]) -> list[MessageRecord]:
        starts = [index for index, message in enumerate(messages) if message.role == "user"]
        if len(starts) <= self.max_turns:
            return messages
        # Cut only at a user turn boundary. Tool requests and results stay together.
        return messages[starts[-self.max_turns]:]

    def save(self, session: SessionState) -> Path:
        """Write a full session snapshot and return its path."""
        if not isinstance(session, SessionState):
            raise TypeError("session must be a SessionState")
        path = self._path(session.session_id)
        session.messages = self._limited_messages(session.messages)
        payload = json.dumps(session.to_dict(), indent=2, ensure_ascii=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f"{path.stem}.{uuid4().hex}.tmp")
        try:
            pending.write_text(payload, encoding="utf-8")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return path

    def add_message(self, session_id: str, role: str, content: str) -> MessageRecord:
        """Append and persist one user or assistant message."""
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages are supported by add_message")
        message = MessageRecord(role=role, content=content)
        session = self.load(session_id)
        session.add(message)
        self.save(session)
        return message
