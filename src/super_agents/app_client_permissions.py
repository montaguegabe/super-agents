from __future__ import annotations

from .app_models import PendingServerRequest, PermissionRequestCallback
from .app_permissions import (
    clear_shared_permission_request,
    is_permission_request,
    normalize_permission_response,
)
from .backend_config import normalize_backend
from .state import JsonObject


class PermissionClientMixin:
    def register_permission_callback(
        self, callback: PermissionRequestCallback | None
    ) -> PermissionRequestCallback | None:
        """Register a Python callback for app-server approval requests.

        The callback receives a PendingServerRequest. Return a JSON object such as
        {"decision": "accept"} to answer the request, or return None to leave it
        pending for the existing MCP codex_answer_request flow.
        """
        self._permission_callback = callback
        return callback

    def clear_permission_callback(self) -> None:
        self._permission_callback = None

    def pending_permission_requests(self) -> list[PendingServerRequest]:
        return [request for request in self._pending_server_requests.values() if is_permission_request(request.method)]

    def permission_overrides(self, input_data: JsonObject) -> JsonObject:
        result: JsonObject = {}
        approval_policy = input_data.get("approvalPolicy")
        if isinstance(approval_policy, str) and approval_policy:
            result["approvalPolicy"] = approval_policy
        sandbox_policy = input_data.get("sandboxPolicy") or input_data.get("sandbox")
        sandbox_type = input_data.get("sandboxType")
        if not sandbox_policy and sandbox_type == "dangerFullAccess":
            sandbox_policy = {"type": "dangerFullAccess"}
        elif sandbox_policy == "danger-full-access":
            sandbox_policy = {"type": "dangerFullAccess"}
        elif sandbox_policy == "workspace-write":
            sandbox_policy = {"type": "workspaceWrite"}
        elif sandbox_policy == "read-only":
            sandbox_policy = {"type": "readOnly"}
        if isinstance(sandbox_policy, str) and sandbox_policy:
            result["sandboxPolicy"] = sandbox_policy
        elif isinstance(sandbox_policy, dict) and sandbox_policy:
            result["sandboxPolicy"] = sandbox_policy
        return result

    async def answer_request(
        self,
        request_id: str | int,
        result: JsonObject,
        *,
        backend: str | None = None,
    ) -> JsonObject:
        selected_backend = normalize_backend(backend) if backend else self.backend
        if selected_backend != self.backend:
            raise ValueError(f"This client handles backend {self.backend}, not {selected_backend}.")
        await self.ensure_connected()
        if request_id not in self._pending_server_requests:
            raise ValueError(f"No pending app-server request found for id {request_id}.")
        request = self._pending_server_requests[request_id]
        await self.send(
            {
                "id": request_id,
                "result": normalize_permission_response(request, result),
            }
        )
        request = self._pending_server_requests.pop(request_id)
        self.remove_pending_request_from_turn(request_id)
        clear_shared_permission_request(request_id, self.approval_requests_file)
        return {"answered": True, "request": request.to_json()}
