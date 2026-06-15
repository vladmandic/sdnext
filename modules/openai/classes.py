from dataclasses import dataclass
from typing import Any, List, Optional
from transformers import StoppingCriteria


@dataclass
class Model:
    name: Optional[str] = None
    cls: str = "UnknownModelClass"
    tokenizer: str = "UnknownTokenizerClass"
    processor: Optional[str] = None
    type: Optional[str] = None


@dataclass
class Config:
    max_context_tokens: int = 4096
    max_new_tokens: int = 512
    stream: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    use_cache: bool = True


@dataclass
class Attention:
    implementation: str = "eager"
    flash: bool = False
    arch: str = ""


@dataclass
class Req:
    id: int
    client: Optional[str] = None
    url: Optional[str] = None
    config: dict[str, Any] = None


class CustomTokenStopCriteria(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_ids


class Stats:
    id: int = 0
    messages: int = 0
    prompt: int = 0
    ttft: float = 0.0
    latency: float = 0.0
    tps: float = 0.0
    tools: int = 0
    tokens: dict[str, int] = {}
    chunks: dict[str, int] = {}
    streaming: bool = None
    thinking: bool = None

    def __str__(self):
        return f"Res(id={self.id}, messages={self.messages}, prompt={self.prompt}, ttft={self.ttft:.3f}, latency={self.latency:.3f}, tps={self.tps:.3f}, tools={self.tools}, tokens={self.tokens}, chunks={self.chunks}, streaming={self.streaming}, thinking={self.thinking})"
