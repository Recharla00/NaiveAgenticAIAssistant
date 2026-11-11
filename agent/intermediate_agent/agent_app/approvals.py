"""Require a fresh human decision before a tool can change data."""

from dataclasses import dataclass
import json
import sys
from threading import Lock
from typing import Callable

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from .flow_logging import logger


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict
    call_id: str


ApprovalHandler = Callable[[ApprovalRequest], bool]


def interactive_approval(request: ApprovalRequest) -> bool:
    """Show the exact proposed call; only an explicit yes approves it."""
    print("\nApproval required before changing data:", file=sys.stderr)
    print(f"Tool: {request.tool_name}", file=sys.stderr)
    print(f"Arguments: {json.dumps(request.arguments, ensure_ascii=False, indent=2, default=str)}",
          file=sys.stderr)
    try:
        answer = input("Approve this tool call? [y/N]: ")
    except EOFError:
        return False
    return answer.strip().casefold() in {"y", "yes"}


def require_data_change_approval(handler: ApprovalHandler | None):
    """Unknown tools need approval; only marked read-only tools bypass it."""
    prompt_lock = Lock()

    @wrap_tool_call(name="RequireDataChangeApproval")
    def require_approval(request, execute):
        tool = request.tool
        if tool is None or (tool.metadata or {}).get("data_access") == "read_only":
            return execute(request)
        call = ApprovalRequest(tool_name=request.tool_call["name"],
                               arguments=dict(request.tool_call["args"]),
                               call_id=request.tool_call["id"])
        logger.info("Approval required: tool=%s call_id=%s", call.tool_name, call.call_id)
        approved = False
        if handler is not None:
            try:
                with prompt_lock:
                    approved = handler(call) is True
            except Exception as error:
                logger.warning("Approval check failed: tool=%s error=%s", call.tool_name, error)
        if not approved:
            logger.info("Approval denied: tool=%s call_id=%s", call.tool_name, call.call_id)
            return ToolMessage(content=f"Approval was not granted for {call.tool_name}. No data was changed.",
                               tool_call_id=call.call_id, name=call.tool_name,
                               status="error", artifact={"approval_denied": True})
        logger.info("Approval granted: tool=%s call_id=%s", call.tool_name, call.call_id)
        return execute(request)

    return require_approval
