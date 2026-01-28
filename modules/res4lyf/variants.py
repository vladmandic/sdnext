from .abnorsett_scheduler import ABNorsettScheduler
from .common_sigma_scheduler import CommonSigmaScheduler
from .etdrk_scheduler import ETDRKScheduler
from .lawson_scheduler import LawsonScheduler
from .pec_scheduler import PECScheduler
from .res_multistep_scheduler import RESMultistepScheduler
from .res_multistep_sde_scheduler import RESMultistepSDEScheduler
from .res_singlestep_scheduler import RESSinglestepScheduler
from .res_singlestep_sde_scheduler import RESSinglestepSDEScheduler
from .res_unified_scheduler import RESUnifiedScheduler
from .riemannian_flow_scheduler import RiemannianFlowScheduler

# RES Unified Variants

"""
    Supports RES 2M, 3M, 2S, 3S, 5S, 6S
    Supports DEIS 1S, 2M, 3M
"""

class RESU2MScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "res_2m"
        super().__init__(**kwargs)


class RESU3MScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "res_3m"
        super().__init__(**kwargs)


class RESU2SScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "res_2s"
        super().__init__(**kwargs)


class RESU3SScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "res_3s"
        super().__init__(**kwargs)


class RESU5SScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "res_5s"
        super().__init__(**kwargs)


class RESU6SScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "res_6s"
        super().__init__(**kwargs)


class DEISU1SScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "deis_1s"
        super().__init__(**kwargs)


class DEISU2MScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "deis_2m"
        super().__init__(**kwargs)


class DEISU3MScheduler(RESUnifiedScheduler):
    def __init__(self, **kwargs):
        kwargs["rk_type"] = "deis_3m"
        super().__init__(**kwargs)


# RES Multistep Variants
class RES2MScheduler(RESMultistepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_2m"
        super().__init__(**kwargs)


class RES3MScheduler(RESMultistepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_3m"
        super().__init__(**kwargs)


class DEIS2MScheduler(RESMultistepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "deis_2m"
        super().__init__(**kwargs)


class DEIS3MScheduler(RESMultistepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "deis_3m"
        super().__init__(**kwargs)


# RES Multistep SDE Variants
class RES2MSDEScheduler(RESMultistepSDEScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_2m"
        super().__init__(**kwargs)


class RES3MSDEScheduler(RESMultistepSDEScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_3m"
        super().__init__(**kwargs)


# RES Singlestep (Multistage) Variants
class RES2SScheduler(RESSinglestepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_2s"
        super().__init__(**kwargs)


class RES3SScheduler(RESSinglestepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_3s"
        super().__init__(**kwargs)


class RES5SScheduler(RESSinglestepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_5s"
        super().__init__(**kwargs)


class RES6SScheduler(RESSinglestepScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_6s"
        super().__init__(**kwargs)


# RES Singlestep SDE Variants
class RES2SSDEScheduler(RESSinglestepSDEScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_2s"
        super().__init__(**kwargs)


class RES3SSDEScheduler(RESSinglestepSDEScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_3s"
        super().__init__(**kwargs)


class RES5SSDEScheduler(RESSinglestepSDEScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_5s"
        super().__init__(**kwargs)


class RES6SSDEScheduler(RESSinglestepSDEScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "res_6s"
        super().__init__(**kwargs)


# ETDRK Variants
class ETDRK2Scheduler(ETDRKScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "etdrk2_2s"
        super().__init__(**kwargs)


class ETDRK3AScheduler(ETDRKScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "etdrk3_a_3s"
        super().__init__(**kwargs)


class ETDRK3BScheduler(ETDRKScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "etdrk3_b_3s"
        super().__init__(**kwargs)


class ETDRK4Scheduler(ETDRKScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "etdrk4_4s"
        super().__init__(**kwargs)


class ETDRK4AltScheduler(ETDRKScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "etdrk4_4s_alt"
        super().__init__(**kwargs)


# Lawson Variants
class Lawson2AScheduler(LawsonScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "lawson2a_2s"
        super().__init__(**kwargs)


class Lawson2BScheduler(LawsonScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "lawson2b_2s"
        super().__init__(**kwargs)


class Lawson4Scheduler(LawsonScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "lawson4_4s"
        super().__init__(**kwargs)


# ABNorsett Variants
class ABNorsett2MScheduler(ABNorsettScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "abnorsett_2m"
        super().__init__(**kwargs)


class ABNorsett3MScheduler(ABNorsettScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "abnorsett_3m"
        super().__init__(**kwargs)


class ABNorsett4MScheduler(ABNorsettScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "abnorsett_4m"
        super().__init__(**kwargs)


# PEC Variants
class PEC2H2SScheduler(PECScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "pec423_2h2s"
        super().__init__(**kwargs)


class PEC2H3SScheduler(PECScheduler):
    def __init__(self, **kwargs):
        kwargs["variant"] = "pec433_2h3s"
        super().__init__(**kwargs)


# Riemannian Flow Variants
class EuclideanFlowScheduler(RiemannianFlowScheduler):
    def __init__(self, **kwargs):
        kwargs["metric_type"] = "RiemannianFlowScheduler"
        super().__init__(**kwargs)


class HyperbolicFlowScheduler(RiemannianFlowScheduler):
    def __init__(self, **kwargs):
        kwargs["metric_type"] = "hyperbolic"
        super().__init__(**kwargs)


class SphericalFlowScheduler(RiemannianFlowScheduler):
    def __init__(self, **kwargs):
        kwargs["metric_type"] = "spherical"
        super().__init__(**kwargs)


class LorentzianFlowScheduler(RiemannianFlowScheduler):
    def __init__(self, **kwargs):
        kwargs["metric_type"] = "lorentzian"
        super().__init__(**kwargs)


# Common Sigma Variants
class SigmoidSigmaScheduler(CommonSigmaScheduler):
    def __init__(self, **kwargs):
        kwargs["profile"] = "sigmoid"
        super().__init__(**kwargs)


class SineSigmaScheduler(CommonSigmaScheduler):
    def __init__(self, **kwargs):
        kwargs["profile"] = "sine"
        super().__init__(**kwargs)


class EasingSigmaScheduler(CommonSigmaScheduler):
    def __init__(self, **kwargs):
        kwargs["profile"] = "easing"
        super().__init__(**kwargs)


class ArcsineSigmaScheduler(CommonSigmaScheduler):
    def __init__(self, **kwargs):
        kwargs["profile"] = "arcsine"
        super().__init__(**kwargs)


class SmoothstepSigmaScheduler(CommonSigmaScheduler):
    def __init__(self, **kwargs):
        kwargs["profile"] = "smoothstep"
        super().__init__(**kwargs)
