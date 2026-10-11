"""Query ownership at Claude result receipt, before asynchronous persistence."""

from dataclasses import dataclass


@dataclass(eq=False)
class PendingQuery:
    # Only queries present when a result arrived can have coalesced into it.
    covered: bool = False


class PendingResults:
    def __init__(self) -> None:
        self._queries: dict[str, list[PendingQuery]] = {}

    def __bool__(self) -> bool:
        return bool(self._queries)

    def count(self, session_id: str) -> int:
        return len(self._queries.get(session_id, ()))

    def requires_result(self, session_id: str) -> bool:
        return any(not query.covered for query in self._queries.get(session_id, ()))

    def register(self, session_id: str) -> PendingQuery:
        query = PendingQuery()
        self._queries.setdefault(session_id, []).append(query)
        return query

    def receive(self, session_id: str) -> bool:
        """Atomically classify and debit a result; return whether it was solicited."""
        queries = self._queries.get(session_id)
        if not queries:
            return False
        for query in queries:
            query.covered = True
        self.discard(session_id, queries[0])
        return True

    def discard(self, session_id: str, query: PendingQuery) -> None:
        # A failed send can finish after its ticket was consumed. Never debit
        # a different query that arrived while the send was suspended.
        queries = self._queries.get(session_id)
        if queries and query in queries:
            queries.remove(query)
            if not queries:
                self.clear(session_id)

    def clear(self, session_id: str) -> None:
        self._queries.pop(session_id, None)
