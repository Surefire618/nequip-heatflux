from .hardy_heatflux import HardyCalculator
from .stress_calculator import StressCalculator
from .unfolded_heatflux import UnfoldedHeatFluxCalculator, FoldedHeatFluxCalculator
from .load_model import load_nequip_model

def calculator(**kwargs):
    return load_nequip_model(**kwargs)
