---
name: check-api
description: "Audit SD.Next API route definitions and verify endpoint parameters plus request/response signatures against declared FastAPI contracts."
argument-hint: "Optionally focus on specific route prefixes or endpoint groups"
---

# Check API Endpoints And Signatures

Read modules/api/api.py, enumerate all registered endpoints, and validate that each endpoint has coherent request parameters and response signatures.

## When To Use

- The user asks to audit API correctness
- A change touched API routes, endpoint handlers, or API models
- OpenAPI docs look wrong or clients report schema mismatches
- You need a pre-PR API contract sanity pass

## Primary File

- `modules/api/api.py`

This file is the route registration hub and must be treated as the source of truth for direct `add_api_route(...)` registrations.

## Secondary Files To Inspect

- `modules/api/models.py`
- `modules/api/endpoints.py`
- `modules/api/generate.py`
- `modules/api/process.py`
- `modules/api/control.py`
- Any module loaded via `register_api(...)` from `modules/api/*` and feature modules (caption, lora, gallery, civitai, rembg, etc.)

## Audit Goals

For every endpoint, verify:

1. Route method and path are valid and unique after subpath handling.
2. Handler call signature is compatible with the route declaration.
3. Declared `response_model` is coherent with returned payload shape.
4. Request body or query params implied by handler type hints are consistent with expected client usage.
5. Authentication behavior is intentional (`auth=True` default in `add_api_route`).
6. OpenAPI schema exposure is correct (including trailing-slash duplicate suppression).

## Procedure

### 1. Enumerate Routes In modules/api/api.py

- Collect every `self.add_api_route(...)` call.
- Capture: path, methods, handler symbol, response_model, tags, auth flag.
- Note route groups: server, generation, processing, scripts, enumerators, functional.

### 2. Resolve Handler Definitions

For each handler symbol:

- Locate the callable definition.
- Read its function signature and type hints.
- Identify required positional args, optional args, and body model annotations.

Flag issues such as:

- Required parameters that cannot be supplied by FastAPI.
- Mismatch between route method and handler intent (for example body expected on GET).
- Ambiguous or missing type hints for public API handlers.

### 3. Validate Request Signatures

- Confirm request model classes exist and are importable.
- Confirm endpoint signature reflects expected request source (path/query/body).
- Check optional vs required semantics for compatibility-sensitive endpoints.

### 4. Validate Response Signatures

- Compare declared `response_model` to handler return shape.
- Ensure list/dict wrappers match actual payload structure.
- Flag obvious drift where `response_model` implies fields never returned.

### 5. Include register_api(...) Modules

`Api.register()` delegates additional endpoints through module-level `register_api(...)` calls.

- Inspect each delegated module for routes.
- Apply the same request/response signature checks.
- Include these findings in the final report, not only direct routes in api.py.

### 6. Verify Runtime Schema (Optional But Preferred)

If feasible in the current environment:

- Build app and inspect `app.routes` metadata.
- Generate OpenAPI schema and spot-check key endpoints.
- Confirm trailing-slash duplicate suppression behavior remains correct.

If runtime schema checks are not feasible, explicitly state that and rely on static validation.

## Reporting Format

Report findings ordered by severity:

1. Breaking contract mismatches
2. Likely runtime errors
3. Schema quality or consistency issues
4. Minor style/typing improvements

For each finding include:

- Route path and method
- File and function
- Why it is a mismatch
- Minimal fix recommendation

If no issues are found, state that explicitly and mention residual risk (for example runtime-only behavior not executed).

## Output Expectations

When this skill is used, return:

- Total endpoints checked
- Direct routes vs delegated `register_api(...)` routes checked
- Findings list with severity and locations
- Clear pass/fail summary for request and response signature consistency
