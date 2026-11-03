---
description: "Use when changing the optional chat feature, including its API, command parsing, persistence, operations, or artifacts."
name: "Chat Feature Guidelines"
applyTo: "modules/chat/**/*.py, wiki/Chat.md"
---
# Chat Feature Guidelines

1. Preserve the boundaries between `modules/chat/api.py` request/response contracts, `modules/chat/commands.py` parsing and validation, `modules/chat/db.py` persistence, and `modules/chat/artifacts.py` artifact handling. Reuse these modules rather than duplicating their responsibilities.
2. Implement every chat command through the FastAPI routes in `modules/chat/api.py` and the service/parser code they call. Keep `modules/chat` independent of Gradio: do not add Gradio imports, callbacks, components, or command implementations. UI clients should call the API.
3. Treat Pydantic request/response models and the command allowlists/aliases as API contracts. Coordinate payload or command changes with the chat UI and keep validation at the API boundary.
4. Keep command handling deterministic and constrained to declared operations and endpoints. Validate arguments using the existing request models; do not turn free-form chat content into executable code or unrestricted API calls.
5. Keep SQLite access centralized in `modules/chat/db.py`. Reuse its connection and transaction helpers, use parameterized SQL for values, and keep configured database paths rather than hardcoding locations.
6. Make persisted schema changes versioned and migratable without losing existing data. Preserve transaction and foreign-key behavior, constraints, indexes, and schema-version bookkeeping.
7. Track long-running work through the operation lifecycle and existing generation/control APIs. Preserve cancellation and status behavior rather than running generation work outside the established control path.
8. Keep artifact paths and display names behind the artifact helpers, and preserve the existing handling for missing or deleted source files.
9. Document every new or changed chat command supported by the module in `wiki/Chat.md`, including its syntax, aliases, arguments, and behavior. Reflect other user-visible chat feature changes there as appropriate; avoid unrelated rewrites or cleanup.
