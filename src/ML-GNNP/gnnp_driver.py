"""
Copyright (c) 2025, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

from ase import Atoms
from ase.calculators.mixing import SumCalculator

import os
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    try:
        import tensorflow as tf
        CUDA_AVAILABLE = tf.config.list_physical_devices("GPU")
    except ImportError as e:
        raise ImportError("Neither `torch` nor `tensorflow` is available. Please install one of them.") from e

_USING_TORCH_DFTD3 = True


def gnnp_initialize(
    gnnp_type: str,
    model_name: str = None,
    as_path: bool = False,
    dftd3: bool = False,
    gpu: bool = True,
) -> tuple[float, int]:
    """
    Initialize GNNP.

    Args:
        gnnp_type (str): type of GNNP. -> {chgnet|fairchem|grace|matgl|mace|mace-off|mattersim|orb}
        model_name (str): name of the model for GNNP.
        as_path (bool): if true, model_name is path of the model file. this is only for chgnet, mace, and fairchem.
        dftd3 (bool): to add correction of DFT-D3.
        gpu (bool): using GPU, if possible.

    Returns:
        cutoff (float): cutoff radius.
        with_stress (int): to calculate stress or not.
    """
    global myCalculator
    global gnnpCalculator
    global dftd3Calculator
    global myAtoms

    if gnnp_type is None:
        raise ValueError("gnnp_type is not defined.")

    gpu = gpu and CUDA_AVAILABLE
    device = "cuda" if gpu else "cpu"

    myAtoms = None
    myCalculator = None
    dftd3Calculator = None

    gnnp_type = gnnp_type.lower()

    if gnnp_type == "chgnet":
        from chgnet.model import CHGNet, CHGNetCalculator

        myPotential = (
            CHGNet.from_file(model_name) if as_path else
            CHGNet.load(use_device=device, model_name=model_name) if model_name else
            CHGNet.load(use_device=device)
        )

        myCalculator = CHGNetCalculator(
                model=myPotential,
                use_device=device
        )

        ratom = float(myPotential.graph_converter.atom_graph_cutoff)
        rbond = float(myPotential.graph_converter.bond_graph_cutoff)
        cutoff = max(ratom, rbond)

    elif gnnp_type == "fairchem":
        from fairchem.core import OCPCalculator

        if as_path:
            myCalculator = OCPCalculator(
                    checkpoint_path=model_name,
                    cpu=not gpu
            )

        else:
            myCalculator = OCPCalculator(
                    model_name=model_name,
                    local_cache=os.path.normpath(os.path.join(os.path.expanduser("~"), ".fairchem")),
                    cpu=not gpu
            )

        cutoff = myCalculator.config["model"].get("max_radius", 8.0)

    elif gnnp_type == "grace":
        from tensorpotential.calculator.foundation_models import grace_fm

        myCalculator = grace_fm(
            model=model_name or "GRACE-2L-OAM",
        )

        cutoff = myCalculator.cutoff

    elif gnnp_type == "hienet":
        from hienet.hienet_calculator import HIENetCalculator

        myCalculator = HIENetCalculator(
                model=model_name,
                device=device
        )

        cutoff = myCalculator.model.cutoff

    elif gnnp_type == "matgl":
        import matgl
        from matgl.ext.ase import PESCalculator

        torch.set_default_device(device)

        myPotential = (
                matgl.load_model(model_name) if model_name else
                matgl.load_model("M3GNet-MP-2021.2.8-PES")
        )

        myCalculator = PESCalculator(
                potential=myPotential
        )

        cutoff = myPotential.model.cutoff

    elif gnnp_type == "mace":
        from mace.calculators import mace_mp

        myCalculator = mace_mp(
                model=model_name,
                device=device,
                dispersion=dftd3,
                damping="zero",
        )

        if dftd3:
            dftd3 = False

        if isinstance(myCalculator, SumCalculator):
            cutoff = myCalculator.mixer.calcs[0].r_max
        else:
            cutoff = myCalculator.r_max

    elif gnnp_type == "mace-off":
        from mace.calculators import mace_off

        myCalculator = mace_off(
                model=model_name,
                device=device
        )

        cutoff = myCalculator.r_max

    elif gnnp_type == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        myCalculator = MatterSimCalculator(
                load_path=model_name,
                device=device
        )

        cutoff = myCalculator.potential.model.model_args.get("cutoff", 5.0)

    elif gnnp_type == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_func = pretrained.ORB_PRETRAINED_MODELS.get(model_name, pretrained.orb_v3_conservative_20_omat)

        if model_name and "d3" in model_name and dftd3:
            dftd3 = False

        orbff = model_func(device=device)
        myCalculator = ORBCalculator(orbff, device=device)

        cutoff = float(orbff.model.system_config.radius)

    else:
        raise ValueError("gnnp_type is incorrect: " + gnnp_type)

    if "stress" in myCalculator.implemented_properties:
        with_stress = 1
    else:
        with_stress = 0

    gnnpCalculator = myCalculator

    if dftd3:
        if _USING_TORCH_DFTD3:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

            dftd3Calculator = TorchDFTD3Calculator(
                    xc="pbe",
                    damping="zero",
                    abc=False
            )

        else:
            from dftd3.ase import DFTD3

            dftd3Calculator = DFTD3(
                    method="PBE",
                    damping="d3zero",
                    s9=0.0
            )

        myCalculator = SumCalculator([gnnpCalculator, dftd3Calculator])

    return cutoff, with_stress


def gnnp_get_energy_forces_stress(
        cell,
        atomic_numbers,
        positions,
        with_stress: bool = True
) -> tuple:
    """
    Predict total energy, atomic forces and stress w/ pre-trained GNNP.
    Args:
        cell: lattice vectors in angstroms.
        atomic_numbers: atomic numbers for all atoms.
        positions: xyz coordinates for all atoms in angstroms.
        with_stress: to return stress, if True.
    Returns:
        energy: total energy.
        forcces: atomic forces.
        stress: stress tensor (Voigt order).
    """
    global myAtoms
    global myCalculator
    global gnnpCalculator
    global dftd3Calculator

    if myAtoms is not None and len(myAtoms.numbers) != len(atomic_numbers):
        myAtoms = None

    if myAtoms is None:
        myAtoms = Atoms(
                numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=[True, True, True]
        )

        myAtoms.calc = myCalculator

    else:
        myAtoms.set_cell(cell)
        myAtoms.set_atomic_numbers(atomic_numbers)
        myAtoms.set_positions(positions)

    # Predicting energy, forces and stress
    energy = myAtoms.get_potential_energy()
    if not isinstance(energy, float):
        energy = energy.item()

    forces = myAtoms.get_forces().tolist()

    if not with_stress:
        return energy, forces

    if dftd3Calculator is None:
        stress = myAtoms.get_stress().tolist()
    else:
        # to avoid the bug of SumCalculator
        myAtoms.calc = gnnpCalculator
        stress1 = myAtoms.get_stress()

        myAtoms.calc = dftd3Calculator
        stress2 = myAtoms.get_stress()

        stress = stress1 + stress2
        stress = stress.tolist()

        myAtoms.calc = myCalculator

    return energy, forces, stress
