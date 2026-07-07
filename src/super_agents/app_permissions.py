"""Shared approval-queue helpers, provided by the openapprovals package.

This module is the compatibility surface for the pre-openapprovals names;
new code should import from ``openapprovals`` directly. Store-path
resolution (including the legacy ``SUPER_AGENTS_APPROVAL_REQUESTS_FILE``
environment variable and the legacy ``~/.super-agents`` store location)
lives in ``openapprovals.store``.
"""

from __future__ import annotations

from openapprovals import DEFAULT_REQUESTS_FILE as DEFAULT_APPROVAL_REQUESTS_FILE
from openapprovals import clear_request as clear_shared_permission_request
from openapprovals import is_approval_request as is_permission_request
from openapprovals import pending_requests as shared_permission_requests
from openapprovals import pop_decision as pop_shared_permission_decision
from openapprovals import read_store as read_permission_store
from openapprovals import record_request as record_shared_permission_request
from openapprovals import write_decision as write_shared_permission_decision
from openapprovals import write_store as write_permission_store

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
