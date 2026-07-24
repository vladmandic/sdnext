import diffusers


class MageFlowPipeline(diffusers.DiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
