# Super Agents Agent Guide

## Product Boundary

Super Agents is a standalone MIT MCP server and Python library for controlling
Codex app-server threads, turns, callbacks, progress checks, steering, and
related generic agent-session coordination. It is used by Openbase Coder, but
it must remain installable and useful outside Openbase Coder as its own MCP
product.

Keep Openbase Coder product-domain features out of this repo. Do not add tools
or state models for Openbase teams, team activity feeds, reports, Cloud account
state, billing, onboarding, app navigation, or other Openbase-specific UX
flows directly here. Implement those in Openbase Coder repos (`cli`, `skills`,
`console`, mobile/desktop apps) or in a narrow adapter layer that calls generic
Super Agents primitives.

Openbase-aware conveniences may exist only when they preserve the standalone
contract: they must be optional, documented as integration helpers, and must
not make the core MCP tool surface require Openbase Coder.

## Development

- Prefer generic names and behavior in MCP tools, types, state files, and
  public APIs.
- Treat Openbase Coder as one client/integration of Super Agents, not as the
  product boundary of this repo.
- Keep README and tool descriptions clear for non-Openbase MCP users.
