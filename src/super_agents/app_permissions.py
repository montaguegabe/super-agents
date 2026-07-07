"""Shared approval-queue helpers, provided by the open-approvals package.

This module is the compatibility surface for the pre-open-approvals names;
new code should import from ``open-approvals`` directly. Store-path
resolution (including the legacy ``SUPER_AGENTS_APPROVAL_REQUESTS_FILE``
environment variable and the legacy ``~/.super-agents`` store location)
lives in ``open-approvals.store``.
"""

from __future__ import annotations

from open_approvals import DEFAULT_REQUESTS_FILE as DEFAULT_APPROVAL_REQUESTS_FILE
from open_approvals import clear_request as clear_shared_permission_request
from open_approvals import is_approval_request as is_permission_request
from open_approvals import pending_requests as shared_permission_requests
from open_approvals import pop_decision as pop_shared_permission_decision
from open_approvals import read_store as read_permission_store
from open_approvals import record_request as record_shared_permission_request
from open_approvals import write_decision as write_shared_permission_decision
from open_approvals import write_store as write_permission_store

__all__ = [
    "DEFAULT_APPROVAL_REQUESTS_FILE",
    "clear_shared_permission_request",
    "is_permission_request",
    "pop_shared_permission_decision",
    "read_permission_store",
    "record_shared_permission_request",
    "shared_permission_requests",
    "write_permission_store",
    "write_shared_permission_decision",
]
