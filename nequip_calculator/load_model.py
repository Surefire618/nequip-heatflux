from typing import Optional, Tuple, List
from pathlib import Path

import torch
import ase.data

from nequip.data.transforms import TypeMapper
from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from nequip.train import Trainer
from nequip.scripts.train import default_config, check_code_version
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY, TYPE_NAMES_KEY

def _load_deployed_or_traindir(
    path: Path, device, freeze: bool = True
) -> Tuple[torch.nn.Module, bool, float, List[str]]:
    loaded_deployed_model: bool = False
    model_r_max = None
    type_names = None
    try:
        model, metadata = load_deployed_model(
            path,
            device=device,
            set_global_options=True,  # don't warn that setting
            freeze=freeze,
        )
        # the global settings for a deployed model are set by
        # set_global_options in the call to load_deployed_model
        # above
        model_r_max = float(metadata[R_MAX_KEY])
        type_names = metadata[TYPE_NAMES_KEY].split(" ")
        loaded_deployed_model = True
        model_config = metadata
    except ValueError:  # its not a deployed model
        loaded_deployed_model = False
    # we don't do this in the `except:` block to avoid "during handing of this exception another exception"
    # chains if there is an issue loading the training session model. This makes the error messages more
    # comprehensible:
    if not loaded_deployed_model:
        # Use the model config, regardless of dataset config
        global_config = path.parent / "config.yaml"
        global_config = Config.from_file(str(global_config), defaults=default_config)
        _set_global_options(global_config)
        check_code_version(global_config)
        del global_config

        # load a training session model
        model, model_config = Trainer.load_model_from_training_session(
            traindir=path.parent, model_name=path.name
        )
        model = model.to(device)
        model_r_max = model_config["r_max"]
        type_names = model_config["type_names"]
    model.eval()
    return model, loaded_deployed_model, model_r_max, type_names, model_config


def load_nequip_model(
    model_file, if_config=False, device="cuda", calculator_name="nequip"
):
    model_file = Path(model_file).resolve()
    if if_config:
        config = Config.from_file(str(model_file))
        model = model_from_config(config, initialize=True)
        chemical_symbol_to_type = config.get("chemical_symbol_to_type")
        model_r_max = config.get("r_max")

    else:
        model, loaded_deployed_model, model_r_max, type_names, config = (
            _load_deployed_or_traindir(model_file, device=device)
        )
        print(f"    loaded{' deployed' if loaded_deployed_model else ''} model")

        # build typemapper
        species_to_type_name = {s: s for s in ase.data.chemical_symbols}
        type_name_to_index = {n: i for i, n in enumerate(type_names)}
        chemical_symbol_to_type = {
            sym: type_name_to_index[species_to_type_name[sym]]
            for sym in ase.data.chemical_symbols
            if sym in type_name_to_index
        }
        if len(chemical_symbol_to_type) != len(type_names):
            raise ValueError(
                "The default mapping of chemical symbols as type names didn't make sense; please provide an explicit mapping in `species_to_type_name`"
            )
    transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

    if calculator_name == "nequip":
        from nequip.ase import NequIPCalculator

        return NequIPCalculator(
            model=model,
            r_max=model_r_max,
            device=device,
            transform=transform,
        )

    elif calculator_name == "hardy":
        from nequip_calculator import HardyCalculator

        return HardyCalculator(
            model=model,
            r_max=model_r_max,
            device=device,
            transform=transform,
            # barycenter=True,
        )

    elif calculator_name == "virial":
        from nequip_calculator import StressCalculator

        return StressCalculator(
            model=model,
            r_max=model_r_max,
            device=device,
            transform=transform,
        )

    elif calculator_name == "unfolded":
        from nequip_calculator import UnfoldedHeatFluxCalculator

        return UnfoldedHeatFluxCalculator(
            model=model,
            r_max=model_r_max,
            n_interactions=3.,
            device=device,
            transform=transform,
        )

    elif calculator_name == "folded":
        from nequip_calculator import FoldedHeatFluxCalculator

        return FoldedHeatFluxCalculator(
            model=model,
            r_max=model_r_max,
            device=device,
            transform=transform,
        )
