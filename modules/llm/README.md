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
`OpenAIBackend` adds external discovery and inference. Configure its API base URL
under Settings > Model Options > External LLM. The shared catalog invalidates its
cache when the endpoint or credentials change; imports and cache reads do not
perform discovery. Discovery is explicit on refresh, in API model listings, or
in the optional pre-UI callback. Inference also discovers on first use if needed.

Remote selections use `external/` followed by a URL-encoded model ID. The backend
always receives `ModelRef('openai', original_model_id)`; remote names containing
Qwen, Gemma, or cloud-provider names cannot select an in-process handler.
UI labels show the configured endpoint's hostname instead, for example
`llm.example.com / model-id`, with a non-default port when present. The spaced
separator keeps labels distinct from local/cloud repository IDs. This is only a
display alias: API selection IDs, saved UI defaults, and caption preferences keep
their existing `external/` identifiers. No endpoint alias configuration is needed.
Recent server modality metadata is required, with no legacy fallback or probing.
The interface does not interpret `supported_parameters` as an exhaustive list.

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

## External server setup

1. Open **Settings > Model Options > External LLM**.
2. Set **External LLM API URL** to the API base URL, including `/v1`, for example
   `http://llm.example.com/v1`. Set the optional API key, socket connection/read
   timeout, and default output-token budget (8,192 by default).
3. Save settings and refresh each model selector, or restart SD.Next.
4. Select a model under the endpoint's hostname prefix. Models are discovered
   remotely, not configured individually within SD.Next. API requests continue to
   use the `external/` selection IDs returned by discovery.

The server's `/v1/models` response must include
`architecture.input_modalities` and `architecture.output_modalities`, as exposed
by recent llama-server/llama-swap versions. No legacy fallback or capability
guessing is performed. The configured serving context length is used when
advertised; it is not assumed to be the model's theoretical maximum.

Text-generation models appear in prompt/video enhancement and story mode.
Captioning/analysis lists only models also advertising image input. Text-only
stories work when no image captioning or image reference is requested. API
discovery is available through GET `/sdapi/v1/prompt-enhance` and
GET `/sdapi/v1/vqa/models`, including remote friendly names and original IDs.
Failed external discovery is logged without hiding the existing local/cloud
model lists or retaining stale external choices.

Remote generation supports token limits, temperature, top-p, seed, and
llama-server's top-k/repeat-penalty parameters. Local beam search, minimum-token
constraints, Python logits processing, and local custom arguments are rejected
remotely. Prefill is sent as a final assistant message. The disabled thinking
control does not force reasoning off on the server; allow enough tokens for
models whose defaults enable reasoning. Separately returned reasoning can still
be displayed with **Keep Thinking Trace**.

All external requests share the **External LLM default output tokens** setting
(**8,192** by default), including
reasoning and the final answer. Enhancement, captioning, analysis, batch jobs,
and every story stage use this policy. Story planning keeps a larger automatic
estimate when needed for many slides. There is no new story-specific control.
Explicit API/Python `max_tokens` values remain hard limits, even when smaller.
An omitted caption API limit does not inherit the local caption model's setting.
The existing UI token controls adopt the external default when selecting an
external model, unless the user has entered a limit or restored one from saved UI
defaults. Saved limits remain explicit even when equal to an old provider default.
External caption UI limits do not overwrite the saved local-model preference.
Local/cloud defaults are unchanged. A larger limit is not a guarantee that an
unbounded reasoning model will finish; context limits and server defaults still
apply, and this policy does not retry or turn thinking off.

The default can be set between 16 and 32,768 tokens. After saving a changed
default, reselect the model to update existing UI token controls; explicit UI
limits are not replaced. Requests without an explicit limit read the current
setting when generated.

The request timeout applies to blocking socket connection/read operations, not
the total request duration. Responses are non-streaming, so waiting for response
headers can include model loading and generation. Progress inside the remote
model does not itself reset the socket timeout.

Local load/unload controls do not manage remote models. The **OpenAI interface**
checkbox serves an in-process SD.Next model and is not needed for external access.
Image-generation text encoders and built-in prompt understanding are unaffected.

There is no image/LLM memory coordination yet: both may remain loaded. A listed
remote model may also fail to load due to missing files or insufficient memory.

Run offline regression tests with the project venv activated:

```sh
python test/test-llm-backends.py
python test/test-llm-openai.py
```
