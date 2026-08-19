"""Backend-neutral execution policy and approval authorization hooks.

The core library remains permissive by default for backwards compatibility.
Embedders that require external enforcement can set ``require_controls=True``
on a backend client; missing hooks then deny at the execution boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from .state import JsonObject


class ExecutionControlError(RuntimeError):
    """Raised when an injected execution control denies or malfunctions."""


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    backend: str
    operation: str
    requested_policy: JsonObject
    action_digest: str
    thread_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalAuthorizationRequest:
    backend: str
    request_id: str
    action_type: str
    action_digest: str
    thread_id: str | None = None
    turn_id: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    allowed: bool
    reason: str | None = None


class ExecutionPolicyGuard(Protocol):
    async def validate(self, request: ExecutionRequest) -> AuthorizationDecision: ...


class ApprovalAuthorizer(Protocol):
    async def authorize(self, request: ApprovalAuthorizationRequest) -> AuthorizationDecision: ...


class PermissiveExecutionPolicyGuard:
    async def validate(self, request: ExecutionRequest) -> AuthorizationDecision:
        del request
        return AuthorizationDecision(allowed=True)


class PermissiveApprovalAuthorizer:
    async def authorize(self, request: ApprovalAuthorizationRequest) -> AuthorizationDecision:
        del request
        return AuthorizationDecision(allowed=True)


class UnavailableExecutionPolicyGuard:
    async def validate(self, request: ExecutionRequest) -> AuthorizationDecision:
        del request
        return AuthorizationDecision(allowed=False, reason="Execution policy guard is unavailable.")


class UnavailableApprovalAuthorizer:
    async def authorize(self, request: ApprovalAuthorizationRequest) -> AuthorizationDecision:
        del request
        return AuthorizationDecision(allowed=False, reason="Approval authorizer is unavailable.")


def configure_execution_controls(
    target: Any,
    *,
    execution_policy_guard: ExecutionPolicyGuard | None,
    approval_authorizer: ApprovalAuthorizer | None,
    require_controls: bool,
) -> None:
    if require_controls:
        target._execution_policy_guard = execution_policy_guard or UnavailableExecutionPolicyGuard()
        target._approval_authorizer = approval_authorizer or UnavailableApprovalAuthorizer()
    else:
        target._execution_policy_guard = execution_policy_guard or PermissiveExecutionPolicyGuard()
        target._approval_authorizer = approval_authorizer or PermissiveApprovalAuthorizer()


async def validate_execution(
    target: Any,
    *,
    operation: str,
    action: JsonObject,
    requested_policy: JsonObject,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> ExecutionRequest:
    request = ExecutionRequest(
        backend=str(target.backend),
        operation=operation,
        requested_policy=dict(requested_policy),
        action_digest=canonical_action_digest(action),
        thread_id=thread_id,
        turn_id=turn_id,
    )
    try:
        decision = await target._execution_policy_guard.validate(request)
    except Exception as exc:
        raise ExecutionControlError(f"Execution policy validation failed for {operation}.") from exc
    _require_allowed(decision, control="Execution policy", operation=operation)
    return request


async def authorize_approval(
    authorizer: ApprovalAuthorizer,
    *,
    backend: str,
    request_id: str | int,
    action_type: str,
    action: JsonObject,
    thread_id: str | None,
    turn_id: str | None,
    tool_call_id: str | None,
) -> ApprovalAuthorizationRequest:
    request = ApprovalAuthorizationRequest(
        backend=backend,
        request_id=str(request_id),
        action_type=action_type,
        action_digest=canonical_action_digest(action),
        thread_id=thread_id,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
    )
    try:
        decision = await authorizer.authorize(request)
    except Exception as exc:
        raise ExecutionControlError(f"Approval authorization failed for {action_type}.") from exc
    _require_allowed(decision, control="Approval", operation=action_type)
    return request


def canonical_action_digest(action: JsonObject) -> str:
    """Hash the complete in-memory action without accepting caller digests."""
    try:
        payload = json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionControlError("Action cannot be represented as canonical JSON.") from exc
    return hashlib.sha256(payload).hexdigest()


def approval_action(method: str, action_type: str, action_input: JsonObject) -> JsonObject:
    """Build the stable, unredacted object covered by an approval digest."""
    return {"method": method, "tool": action_type, "input": action_input}


def _require_allowed(decision: object, *, control: str, operation: str) -> None:
    if not isinstance(decision, AuthorizationDecision):
        raise ExecutionControlError(f"{control} returned an invalid decision for {operation}.")
    if not decision.allowed:
        suffix = f" {decision.reason}" if decision.reason else ""
        raise ExecutionControlError(f"{control} denied {operation}.{suffix}")
