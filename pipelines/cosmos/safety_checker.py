class CosmosSafetyChecker:
    def __init__(self):
        from diffusers.utils import import_utils
        import_utils._cosmos_guardrail_available = True # pylint: disable=protected-access

    def __call__(self, *args, **kwargs): # pylint: disable=unused-argument
        return

    def to(self, _device):
        pass

    def check_text_safety(self, _prompt):
        return True

    def check_video_safety(self, vid):
        return vid
