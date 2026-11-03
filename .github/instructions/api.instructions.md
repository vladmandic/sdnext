---
description: "Use when adding or changing FastAPI routes, request/response contracts, route registration, authentication, or API handlers."
name: "API Development Guidelines"
applyTo: "modules/api/**/*.py"
---
# API Development Guidelines

1. Treat `modules/api/api.py` as the route-registration hub. Register routes through the existing `Api.add_api_route(...)` or feature `register_api(...)` patterns rather than creating a parallel router or server.
2. Keep each route's method, path, handler signature, request source (path/query/body), and response model aligned. Define or reuse Pydantic models for stable request and response contracts instead of relying on untyped dictionaries where a schema is appropriate.
3. Preserve existing authentication, subpath, status-code, tag, and OpenAPI behavior. `Api.add_api_route(...)` applies shared authentication and subpath handling; make exceptions explicit and consistent with nearby routes.
4. For generation or processing work, reuse the existing API classes, queue lock, and shared helpers. Do not invoke expensive processing directly from a route in a way that bypasses established serialization or request handling.
5. Validate external input at the route boundary, including uploaded files and parsed JSON. Return intentional FastAPI `HTTPException` status codes for client errors, and avoid exposing internal tracebacks or filesystem details in responses.
6. Treat existing paths and payloads as client-facing contracts. Coordinate breaking changes with callers and update route registrations and models together.
7. For endpoint audits, use the `check-api` skill to verify registration, handler parameters, request/response signatures, authentication, and OpenAPI exposure.
