# Auxiliary LLM backends

`ModelRef(provider, model_id)` identifies an auxiliary model. Provider identity is
independent of the model's name; only `resolve_model()` translates the existing
legacy catalog names at application boundaries. It must not be used to guess the
provider of an externally discovered model.

`Backend.generate()` accepts a `Request` and returns a `Response`. The initial
adapters preserve existing Transformers and Gemini/Grok behavior, including native
generation options. SDK imports are lazy. Task-specific instructions, formatting,
and detection results remain in their existing consumers; image-pipeline text
encoders, OpenCLIP, and standalone tagger entry points are not replaced.

Local inference is bound to the existing consumer instance through
`InProcessBackend`; the interface does not introduce another model cache or loader.
Model catalogs and UI/API selection remain unchanged in this refactor. Discovery
and external inference are a separate change.

## Work scopes and future lifecycle control

Use `with backend.session(model_ref) as session:` to group related requests.
`session.generate(request)` uses the same backend and model identity. Story mode
shares one session across captioning, planning, and slide enhancement, closing it
before image generation, including on failure. A session is not chat history.

Sessions currently have **no residency side effects**: they neither load nor unload
models and do not guarantee exclusive memory. A future coordinator can wrap the
scope and use an optional resource-controller interface, without adding meaningless
load/unload methods to cloud providers. Exclusive-memory support will also require
earlier processing-order changes; session boundaries alone cannot provide it.

`Request.thinking=None` leaves support/control unspecified. Future discovery should
preserve unknown capability states rather than infer reasoning from model names or
tool support. `Response.reasoning` can carry separately returned reasoning even
when request-side thinking controls are unknown. Each backend is responsible for
its parameter translation; remote `supported_parameters` metadata must not be
treated as an exhaustive allowlist without a server guarantee.

Run offline regression tests with the project venv activated:

```sh
python test/test-llm-backends.py
```
