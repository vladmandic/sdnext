import time
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .serve import OpenAIServer
from modules import logger


logger.setup_logging(debug=True)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


logger.log.info("OpenAI: load model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True
)

logger.log.info("OpenAI: create server...")
server = OpenAIServer(
    model=model,
    tokenizer=tokenizer,
    host="127.0.0.1",
    port=8000
)

logger.log.info("OpenAI: start server...")
server.start()
while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        break
