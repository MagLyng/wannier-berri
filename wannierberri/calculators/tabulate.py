import numpy as np
from .calculator import Calculator
from ..formula import covariant as frml
from ..result import KBandResult, TABresult
from ..factors import factor_morb_evA2_to_muB


# The base classes for Tabulating
# particular calculators are below


class Tabulator(Calculator):

    def __init__(self, Formula, ibands=None, kwargs_formula=None, constant_factor=1, **kwargs):
        self.Formula = Formula
        self.ibands = np.array(ibands) if (ibands is not None) else None
        self.kwargs_formula = kwargs_formula if kwargs_formula is not None else {}
        self.constant_factor = constant_factor
        super().__init__(**kwargs)

    def __call__(self, data_K):
        formula = self.Formula(data_K, **self.kwargs_formula)
        nk = data_K.nk
        NB = data_K.num_wann
        ibands = self.ibands
        if ibands is None:
            ibands = np.arange(NB)
        band_groups = data_K.get_bands_in_range_groups(
            -np.inf, np.inf, degen_thresh=self.degen_thresh, degen_Kramers=self.degen_Kramers, sea=False)
        # bands_groups  is a digtionary (ib1,ib2):E
        # now select only the needed groups
        band_groups = [
            [n for n in groups.keys() if np.any((ibands >= n[0]) * (ibands < n[1]))] for groups in band_groups
        ]  # select only the needed groups
        group = [[] for _ in range(nk)]
        for ik in range(nk):
            for ib in ibands:
                for n in band_groups[ik]:
                    if n[1] > ib >= n[0]:
                        group[ik].append(n)
                        break

        rslt = np.zeros((nk, len(ibands)) + (3,) * formula.ndim)
        for ik in range(nk):
            values = {}
            for n in band_groups[ik]:
                inn = np.arange(n[0], n[1])
                out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
                values[n] = formula.trace(ik, inn, out) / (n[1] - n[0])
            for ib, b in enumerate(ibands):
                rslt[ik, ib] = values[group[ik][ib]]
        rslt *= self.constant_factor
        return KBandResult(rslt, transformTR=formula.transformTR, transformInv=formula.transformInv)


class TabulatorAll(Calculator):
    """
    TabulatorAll - a pack of all k-resolved calculators (Tabulators)
    """

    def __init__(self, tabulators, ibands=None, mode="grid", save_mode="bin", print_comment=False):
        """ tabulators - dict 'key':tabulator
        one of them should be "Energy" """
        self.tabulators = tabulators
        mode = mode.lower()
        assert mode in ("grid", "path")
        self.mode = mode

        self.save_mode = save_mode
        if "Energy" not in self.tabulators.keys():
            self.tabulators["Energy"] = Energy()
        if ibands is not None:
            ibands = np.array(ibands)
        for k, v in self.tabulators.items():
            if hasattr(v, 'ibands'):
                if v.ibands is not None:
                    try:
                        assert len(v.ibands) == len(ibands)
                        assert np.all(v.ibands == ibands)
                    except AssertionError:
                        raise ValueError(
                            f"tabulator {k} has ibands={v.ibands} not equal to ibands={ibands} required in TabulatorAll")
                else:
                    v.ibands = ibands

        self.comment = (self.__doc__ + "\n Includes the following tabulators : \n" + "-" * 50 + "\n" + "\n".join(
            f""" "{key}" : {val} : {val.comment}\n""" for key, val in self.tabulators.items()) +
            "\n" + "-" * 50 + "\n")
        self._set_comment(print_comment)

    def __call__(self, data_K):
        return TABresult(
            kpoints=data_K.kpoints_all.copy(),
            mode=self.mode,
            recip_lattice=data_K.system.recip_lattice,
            save_mode=self.save_mode,
            results={key: val(data_K)
                     for key, val in self.tabulators.items()})

    @property
    def allow_path(self):
        return self.mode == "path"

    @property
    def allow_grid(self):
        return self.mode == "grid"


###############################################
###############################################
###############################################
###############################################
####                                     ######
####        Implemented calculators      ######
####                                     ######
###############################################
###############################################
###############################################
###############################################


class Energy(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Eavln, **kwargs)


class Velocity(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Velocity, **kwargs)


class InvMass(Tabulator):
    r""" second derivative of energy :math:`\partial_a\partial_b E` in units of eV*angstrom^2"""

    def __init__(self, **kwargs):
        super().__init__(frml.InvMass, **kwargs)


class Der3E(Tabulator):
    r""" third derivative of energy :math:`\partial_a\partial_b\partial_c E` in units of eV*angstrom^3"""

    def __init__(self, **kwargs):
        super().__init__(frml.Der3E, **kwargs)


class BerryCurvature(Tabulator):
    r""" Berry curvature :math:`\Omega_a` in units of angstrom^2"""

    def __init__(self, **kwargs):
        super().__init__(frml.Omega, **kwargs)


class DerBerryCurvature(Tabulator):
    r"Derivative of Berry curvature :math:`X_{ab}\partial_b\Omega_a` in units of angstrom^3"

    def __init__(self, **kwargs):
        super().__init__(frml.DerOmega, **kwargs)


class Der2BerryCurvature(Tabulator):
    r"Second Derivative of Berry curvature :math:`X_{ab}\partial_bc\Omega_a` in units of angstrom^4"

    def __init__(self, **kwargs):
        super().__init__(frml.Der2Omega, **kwargs)


class Spin(Tabulator):
    r""" Spin expectation :math:` \langle u | \mathbf{\sigma} | u \rangle`"""

    def __init__(self, **kwargs):
        super().__init__(frml.Spin, **kwargs)


class DerSpin(Tabulator):
    r"Derivative of Spin :math:`\partial_a \langle u | \mathbf{\sigma} | u \rangle` in units of angstrom"

    def __init__(self, **kwargs):
        super().__init__(frml.DerSpin, **kwargs)


class Der2Spin(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Der2Spin, **kwargs)


class OrbitalMoment(Tabulator):
    r"Orbital moment :math:`m_{orb}` in units of Bohr magneton"

    def __init__(self, **kwargs):
        super().__init__(frml.morb, constant_factor=factor_morb_evA2_to_muB, **kwargs)


class DerOrbitalMoment(Tabulator):
    r"Derivative of orbital moment :math:`\partial_a m_{orb}` in units of Bohr magneton*angstrom"

    def __init__(self, **kwargs):
        super().__init__(frml.Dermorb, constant_factor=factor_morb_evA2_to_muB, **kwargs)


class DerOrbitalMoment_test(Tabulator):
    r""" Derivative of orbital moment :math:`\partial_a m_{orb}` in units of Bohr magneton*angstrom"""

    def __init__(self, **kwargs):
        super().__init__(frml.DerMorb_test, constant_factor=factor_morb_evA2_to_muB, **kwargs)


class Der2OrbitalMoment(Tabulator):

    r""" Second derivative of orbital moment :math:`\partial_a\partial_b m_{orb}` in units of Bohr magneton*angstrom^2"""

    def __init__(self, **kwargs):
        super().__init__(frml.Der2morb, constant_factor=factor_morb_evA2_to_muB, **kwargs)


class SpinBerry(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.SpinOmega, **kwargs)

from ..symmetry.point_symmetry import transform_ident, transform_trans, transform_odd, transform_odd_trans_021
from ..formula import Formula
import itertools

# Band resolved version of above (directly inputting band1, band2)
class Formula_OptCond_Band_Resolved(Formula):
    
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)

        A = data_K.get_A_H(external_terms=self.external_terms)
        self.AA = 1j * A[:, :, :, :, None] * A.swapaxes(1, 2)[:, :, :, None, :] # (4, 4, 3, 3) shape
        self.ndim = 2
        self.transformTR = transform_trans
        self.transformInv = transform_ident

    # .sum(axis=0) sums over the first axis
    # AA[ik, inn] flattens first index, but not the second, since inn,out are arrays of indices (see __call__ method for Tabulator)
    def trace(self, ik, inn, out):
        return self.AA[ik, inn].sum(axis=0)[out].sum(axis=0)

class OptCond_Band_Resolved(Tabulator):
    r"""Band resolved Optical conductivity tabulator. 
    Gives (1.j) times absolute square of Berry connection, outputting five indices: (nk, band1, band2, dir1, dir2)
    Usually band1, band2 are traced over, since most tabulated quantities are band diagonal, 
    which is the motivation for creating a custom call function to hack our way out of this.
    """

    def __init__(self, **kwargs):
        super().__init__(Formula_OptCond_Band_Resolved, **kwargs)
    # Custom __call__ routine for getting all bands
    def __call__(self, data_K):
            formula = self.Formula(data_K, **self.kwargs_formula)
            nk = data_K.nk
            NB = data_K.num_wann
            ibands = self.ibands
            if ibands is None:
                ibands = np.arange(NB)
            # band_groups should be all combinations of bands (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), ...
            band_groups = [[] for _ in range(nk)] # Making empty array
            for ik in range(nk):
                band_groups[ik] = list(itertools.product(ibands, repeat=2)) # Filling each entry with tuples over bands
            # group is just band_groups, keeping for consistency with general __call__ method
            group = band_groups
            rslt = np.zeros((nk, len(ibands), len(ibands)) + (3,) * formula.ndim)
            for ik in range(nk):
                values = {}
                for n in band_groups[ik]:
                    inn = [n[0]]
                    out = [n[1]]
                    values[n] = formula.trace(ik, inn, out) 
                # We need to enumerate over the product in order to get correct index for group
                for i, (ib1, ib2) in enumerate(itertools.product(ibands, repeat=2)):
                    rslt[ik, ib1, ib2] = values[group[ik][i]]
            rslt *= self.constant_factor
            return KBandResult(rslt, transformTR=formula.transformTR, transformInv=formula.transformInv)