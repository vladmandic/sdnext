"""Small, dependency-free contract shared by auxiliary LLM consumers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelRef:
    """A model ID belongs to a provider, not to a globally unique name."""

    provider: str
    model_id: str


@dataclass
class Request:
    prompt: str
    system: str | None = None
    image: Any = None
    prefill: str = ''
    thinking: bool | None = None
    # Existing adapters retain their native generation options in this refactor.
    # A backend must translate/validate its own options, not another backend's.
    parameters: dict = field(default_factory=dict)


@dataclass
class Response:
    text: str
    reasoning: str = ''
    usage: dict = field(default_factory=dict)
    # Local decoders may need the unstripped text to extract reasoning tags.
    raw_text: str = ''
    metrics: dict = field(default_factory=dict)


class Backend(ABC):
    """Inference contract. Resource ownership is deliberately not implied."""

    @abstractmethod
    def generate(self, model: ModelRef, request: Request) -> Response:
        raise NotImplementedError

    @contextmanager
    def session(self, model: ModelRef):
        """Group related calls; a future coordinator can wrap this boundary.

        This is a work scope, not a chat history or an exclusive-memory promise.
        In particular, exiting it does not unload anything in the current backends.
        """
        yield Session(self, model)


@dataclass(frozen=True)
class Session:
    backend: Backend
    model: ModelRef

    def generate(self, request: Request) -> Response:
        return self.backend.generate(self.model, request)
