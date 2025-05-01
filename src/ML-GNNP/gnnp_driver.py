"""
Copyright (c) 2025, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

from ase import Atoms
from ase.calculators.mixing import SumCalculator

import os
import torch

_USING_TORCH_DFTD3 = True


def gnnp_initialize(
    gnnp_type: str,
    model_name: str = None,
    as_path: bool = False,
    dftd3: bool = False,
    gpu: bool = True,
):
    """
    Initialize GNNP.

    Args:
        gnnp_type (str): type of GNNP. -> {matgl|chgnet|mace|mace-off|orb|mattersim|fairchem}
        model_name (str): name of model for GNNP.
        as_path (bool): if true, model_name is path of model file. this is only for chgnet/orb/fairchem.
        dftd3 (bool): to add correction of DFT-D3.
        gpu (bool): using GPU, if possible.

    Returns:
        cutoff (float): cutoff radius.
        with_stress (int): to calculate stress, or not.
    """
    global myCalculator
    global gnnpCalculator
    global dftd3Calculator
    global myAtoms

    if gnnp_type is None:
        raise ValueError("gnnp_type is not defined.")

    gpu = gpu and torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"

    myAtoms = None
    myCalculator = None
    dftd3Calculator = None
    cutoff = -1.0

    gnnp_type = gnnp_type.lower()

    if gnnp_type == "matgl":
        import matgl
        from matgl.ext.ase import PESCalculator

        torch.set_default_device(device)

        if model_name is not None:
            myPotential = matgl.load_model(model_name)
        else:
            myPotential = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        myCalculator = PESCalculator(
                potential=myPotential,
                compute_stress=True,
                stress_unit="eV/A3"
        )

        cutoff = myPotential.model.cutoff

    elif gnnp_type == "chgnet":
        from chgnet.model import CHGNet, CHGNetCalculator

        if model_name is None:
            myPotential = CHGNet.load(use_device=device)
        elif not as_path:
            myPotential = CHGNet.load(use_device=device, model_name=model_name)
        else:
            myPotential = CHGNet.from_file(model_name)

        myCalculator = CHGNetCalculator(
                model=myPotential,
                use_device=device
        )

        ratom = float(myPotential.graph_converter.atom_graph_cutoff)
        rbond = float(myPotential.graph_converter.bond_graph_cutoff)
        cutoff = max(ratom, rbond)

    elif gnnp_type == "mace":
        from mace.calculators import mace_mp

        if model_name is None:
            model = None

        elif model_name.startswith("mace-osaka24"):
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.normpath(os.path.join(base_path, "mace-osaka24"))
            model_path = os.path.normpath(os.path.join(model_dir, model_name))

            if not model_path.endswith(".model"):
                model_path += ".model"

            model = model_path

        else:
            model = model_name

        myCalculator = mace_mp(
                model=model,
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

    elif gnnp_type == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        if as_path:
            orbff = pretrained.orb_v2(
                    weights_path=model_name,
                    device=device
            )

        else:
            if model_name is not None and model_name in pretrained.ORB_PRETRAINED_MODELS:
                model_func = pretrained.ORB_PRETRAINED_MODELS[model_name]
            else:
                model_func = pretrained.orb_v2

            if model_name is not None and "d3" in model_name:
                if dftd3:
                    dftd3 = False

            orbff = model_func(device=device)

        myCalculator = ORBCalculator(orbff, device=device)

        cutoff = float(orbff.model.gnn_stacks[0]._r_max)

    elif gnnp_type == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        myCalculator = MatterSimCalculator(
                load_path=model_name,
                device=device
        )

        cutoff = myCalculator.potential.model.model_args.get("cutoff", 5.0)

    elif gnnp_type == "fairchem":
        from fairchem.core import OCPCalculator

        if as_path:
            myCalculator = OCPCalculator(
                    checkpoint_path=model_name,
                    cpu=not gpu
            )

        else:
            OMAT_CHECKPTS = {
                "EquiformerV2-31M-OMat": "eqV2_31M_omat.pt",
                "EquiformerV2-86M-OMat": "eqV2_86M_omat.pt",
                "EquiformerV2-153M-OMat": "eqV2_153M_omat.pt",
                "EquiformerV2-31M-MP": "eqV2_31M_mp.pt",
                "EquiformerV2-31M-DeNS-MP": "eqV2_dens_31M_mp.pt",
                "EquiformerV2-86M-DeNS-MP": "eqV2_dens_86M_mp.pt",
                "EquiformerV2-153M-DeNS-MP": "eqV2_dens_153M_mp.pt",
                "EquiformerV2-31M-OMat-Alex-MP": "eqV2_31M_omat_mp_salex.pt",
                "EquiformerV2-86M-OMat-Alex-MP": "eqV2_86M_omat_mp_salex.pt",
                "EquiformerV2-153M-OMat-Alex-MP": "eqV2_153M_omat_mp_salex.pt",
            }

            if model_name is not None:
                checkpt_name = OMAT_CHECKPTS.get(model_name)
            else:
                checkpt_name = OMAT_CHECKPTS.get("EquiformerV2-31M-OMat")

            if checkpt_name is not None:
                base_path = os.path.dirname(os.path.abspath(__file__))
                checkpt_dir = os.path.normpath(os.path.join(base_path, "fairchem-omat24"))
                model_path = os.path.normpath(os.path.join(checkpt_dir, checkpt_name))

                myCalculator = OCPCalculator(
                        checkpoint_path=model_path,
                        cpu=not gpu
                )

            else:
                base_path = os.path.expanduser("~")
                checkpt_dir = os.path.normpath(os.path.join(base_path, ".fairchem"))

                myCalculator = OCPCalculator(
                        model_name=model_name,
                        local_cache=checkpt_dir,
                        cpu=not gpu
                )

        cutoff = myCalculator.config["model"].get("max_radius", 8.0)

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
):
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

