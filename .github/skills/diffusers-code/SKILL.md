---
name: diffusers-code
description: "Create or edit code that is compliant with Hugging Face diffusers conventions, including models, pipelines, schedulers, tests, docs, and PR preparation targeting diffusers."
argument-hint: "Describe the target feature/bug, affected diffusers components, reference implementation links, and whether to prepare a PR-ready change"
---

# Diffusers Code Implementation And PR Skill

Use this skill to implement, edit, review, and prepare pull-request-ready changes for the diffusers library with high compliance to diffusers conventions.

## When To Use

- Adding or editing diffusers models, pipelines, schedulers, or loaders
- Fixing bugs in inference code paths in diffusers-compatible style
- Refactoring existing diffusers code while preserving behavior
- Adding tests and docs for diffusers changes
- Preparing a PR that targets the diffusers repository

## Primary Objectives

1. Keep behavior explicit, minimal, and inference-focused.
2. Match existing diffusers architecture and code patterns.
3. Preserve numerical behavior unless a behavior change is explicitly required.
4. Produce change sets that are clean, reviewable, and PR-ready.

## Hard Rules

- Keep logic simple and readable in the main forward or call path.
- Avoid defensive, speculative, or fallback code paths unless required by existing diffusers APIs.
- Do not silently guess intent. For unsupported inputs, raise concise errors.
- Do not introduce new mandatory dependencies without maintainer agreement.
- If optional dependencies are needed, guard imports and provide proper dummy paths.
- Keep implementation torch.compile-friendly: avoid graph-break patterns in core model paths.
- Prefer native PyTorch tensor ops over external reshape helpers.

## Code Structure Rules

### Models

- Use ModelMixin patterns and register constructor args with register_to_config.
- Keep layer invocation visible in forward, avoid hiding key module calls in extra helpers.
- Avoid hardcoded dtypes in forward paths; infer from tensors or module dtype.
- Follow existing model family patterns in src/diffusers/models/transformers.

### Attention

- Keep Attention class and processor together in the model file when following the standard diffusers pattern.
- Processor should perform the compute path and use dispatch_attention_fn pattern where applicable.
- Ensure processor registration and available processor declarations are complete.

### Pipelines

- Inherit from DiffusionPipeline.
- Decorate inference __call__ with @torch.no_grad().
- Support generator for reproducibility when workflow requires it.
- Support output_type="latent" where latent output skip is expected.
- Use self.progress_bar(timesteps) in denoising loops.
- Do not build variant behavior by subclassing an unrelated existing pipeline class.

### Schedulers

- Use SchedulerMixin and ConfigMixin.
- Keep scheduler config semantics consistent with existing scheduler implementations.

## Import And Registration Rules

- Register new classes in relevant __init__.py lazy import structures.
- Ensure import structure entries are complete for all newly exposed objects.
- Validate that public imports from diffusers work after edits.

## Copied Code Rules

- Respect # Copied from linkage.
- Do not manually diverge copied blocks unless intentionally breaking linkage.
- Run make fix-copies after changes that touch copied sources or copied blocks.

## Change Workflow

1. Gather context
- Confirm target files, model family, and expected behavior.
- Obtain reference implementation and runnable inference flow when porting.

2. Plan minimal scope
- Separate structural adaptation from algorithmic changes.
- Keep one coherent workflow per change set.

3. Implement
- Edit only required files.
- Preserve naming, config shape, and API contracts unless change requires otherwise.

4. Validate
- Run focused tests first, then broader checks as needed.
- Confirm imports and serialization/deserialization behavior.

5. Polish
- Run make style.
- Run make fix-copies.
- Re-run impacted tests.

## Testing Expectations

Include tests for the exact behavior being changed:

- Model tests for shape, dtype/device behavior, serialization, and config parity
- Pipeline tests for deterministic generation paths, outputs, and parameter handling
- Scheduler tests when scheduler logic or config behavior changes
- Regression tests for any bug fix

When parity with a reference implementation is required:

- Add component-level parity checks
- Add end-to-end parity checks
- Use explicit tolerances and deterministic seeds

## PR Preparation For Diffusers

When asked to prepare a PR targeting diffusers, produce:

1. Scope statement
- One-paragraph summary: problem, solution, and non-goals.

2. Change map
- File-by-file list describing what changed and why.

3. Validation evidence
- Commands run, tests passed, and any skipped tests with reasons.

4. Compatibility notes
- Backward compatibility, serialization impact, and optional dependency impact.

5. Reviewer guidance
- Key files to review first, known tradeoffs, and follow-up items.

### PR Quality Checklist

- [ ] Minimal focused diff
- [ ] No unrelated refactors mixed with behavior changes
- [ ] New/updated tests for changed behavior
- [ ] Docs updated when public APIs or user-facing behavior changed
- [ ] make style completed
- [ ] make fix-copies completed
- [ ] Relevant test suites pass
- [ ] Commit messages are clear and scoped

## Common Failure Modes To Prevent

- Missing lazy import registration causes runtime ImportError
- New config params not registered, causing from_pretrained mismatch
- Pipeline __call__ missing @torch.no_grad(), causing memory growth
- Hardcoded dtype assumptions break mixed precision usage
- Hidden behavior changes introduced during structural refactor
- Unnecessary dependency additions for simple tensor reshaping

## Output Contract For This Skill

When using this skill, provide:

- Implementation summary
- Exact files changed
- Validation summary with command outcomes
- Residual risks or deferred follow-ups
- PR-ready summary text when requested
