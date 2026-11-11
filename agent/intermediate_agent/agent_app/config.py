"""Central settings. Later lessons can add retrieval and session settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "qwen3:1.7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 180.0
    temperature: float = 0.2
    max_output_tokens: int = 512
    keep_alive: str = "30m"
