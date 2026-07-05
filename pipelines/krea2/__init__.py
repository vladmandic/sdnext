from pipelines.krea2.transformer_krea2 import Krea2Transformer2DModel
from pipelines.native_transformer import TransformerSpec


# Checkpoint keys are bare (`first.`, `blocks.N.`, `txtfusion.`, ...) and the transformer's
# module tree mirrors them exactly, so no state-dict conversion is needed. The model has no
# rope/pos buffers, so the default acceptable-missing set is sufficient.
#
# `last.up`/`last.down` are a dormant zero-init residual branch (base ships last.up.weight
# all-zeros). Finetunes predating it omit both keys; zero-fill so the branch stays a no-op.
KREA2_SPEC = TransformerSpec(
    cls=Krea2Transformer2DModel,
    converter=None,
    zero_init_missing=("last.down.", "last.up."),
)
