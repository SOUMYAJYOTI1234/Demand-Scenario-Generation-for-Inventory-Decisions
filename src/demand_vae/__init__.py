"""demand_vae: conditional demand scenario generation for inventory decisions.

Pipeline: data (M5, weekly FOODS) -> conditional sampler (classical baselines
or CVAE) -> SAA newsvendor decision -> two-level evaluation (distributional
metrics + realized decision cost). See docs/project-design-document.md.
"""

from demand_vae.config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
