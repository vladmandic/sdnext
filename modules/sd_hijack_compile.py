import time
import logging
from modules.timer import dynamo


fn = None
ts = None


class CompilationLogInterceptor(logging.Handler):
    def emit(self, record):
        try:
            global fn, ts # pylint: disable=global-statement
            if 'torchdynamo start tracing' in record.msg:
                fn = record.msg.split('torchdynamo start tracing')[-1].strip() # extract first string after 'torchdynamo start tracing' and start timer
                fn = fn.split(' ')[0] # extract first word after 'torchdynamo start tracing'
                ts = time.time()
            if 'run_gc_after_compile' in record.msg:
                if fn is not None:
                    dynamo.ts(fn, ts) # log the time taken for compilation
                    fn = None
        except Exception:
            pass


def install():
    dynamo_logger = logging.getLogger("torch._dynamo")
    dynamo_logger.setLevel(logging.INFO)
    if not any(isinstance(h, CompilationLogInterceptor) for h in dynamo_logger.handlers):
        dynamo_interceptor = CompilationLogInterceptor()
        dynamo_logger.addHandler(dynamo_interceptor)
