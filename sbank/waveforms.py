# Copyright (C) 2011  Nickolas Fotopoulos
# Copyright (C) 2012-2017 Stephen Privitera
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import (division, print_function)

import logging
import sys
from math import isnan
import numpy as np
from numpy import float32, int16
import lal
import lalsimulation as lalsim
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI
from lal import CreateREAL8Vector, CreateCOMPLEX8FrequencySeries
from ligo.lw.lsctables import SnglInspiralTable as llwsit

from .overlap import SBankComputeMatchSkyLoc, SBankComputeMatch
from .overlap import SBankComputeFiveCompMatch, SBankComputeFiveCompFactorMatch
from .psds import get_neighborhood_PSD, get_ASD
from .tau0tau3 import m1m2_to_tau0tau3

np.seterr(all="ignore")
_sit_cols = llwsit.validcolumns


class SnglInspiralTable(llwsit):
    def __init__(self, *args, **kwargs):
        llwsit.__init__(self, *args, **kwargs)
        for entry in _sit_cols.keys():
            if not(hasattr(self, entry)):
                if _sit_cols[entry] in ['real_4', 'real_8']:
                    setattr(self, entry, 0.)
                elif _sit_cols[entry] == 'int_4s':
                    setattr(self, entry, 0)
                elif _sit_cols[entry] == 'lstring':
                    setattr(self, entry, '')
                elif _sit_cols[entry] == 'ilwd:char':
                    setattr(self, entry, '')
            else:
                print("Column %s not recognized" % entry, file=sys.stderr)
                raise ValueError


def compute_mchirp(m1, m2):
    return (m1 * m1 * m1 * m2 * m2 * m2 / (m1 + m2))**0.2


def ceil_pow_2(number):
    return int(2**(np.ceil(np.log2(number))))


#
# Represent template waveforms
#
def compute_fplus_fcross(theta, phi, psi):
    # compute antenna factors Fplus and Fcross
    fp = 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.cos(2 * psi)
    fp -= np.cos(theta) * np.sin(2 * phi) * np.sin(2 * psi)
    fc = 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.sin(2 * psi)
    fc += np.cos(theta) * np.sin(2 * phi) * np.cos(2 * psi)
    return fp, fc

def project_hplus_hcross(hplus, hcross, theta, phi, psi):
    # compute antenna factors Fplus and Fcross
    fp, fc = compute_fplus_fcross(theta, phi, psi)

    # form strain signal in detector
    hplus.data.data = fp * hplus.data.data + fc * hcross.data.data

    return hplus


def compute_sigmasq(htilde, deltaF):
    """
    Find norm of whitened h(f) array.
    """
    # vdot is dot with complex conjugation
    return float(np.vdot(htilde, htilde).real * 4 * deltaF)


def compute_correlation(htilde1, htilde2, deltaF):
    """
    Find the real component of correlation between htilde1 and htilde2.
    """
    # vdot is dot with complex conjugation
    return float(np.vdot(htilde1, htilde2).real * 4 * deltaF)

def compute_complex_correlation(htilde1, htilde2, deltaF):
    """
    Find the complex correlation between htilde1 and htilde2.
    """
    # vdot is dot with complex conjugation
    return np.vdot(htilde1, htilde2) * 4 * deltaF

def FrequencySeries_to_COMPLEX8FrequencySeries(fs):
    """
    Create a new COMPLEX8FrequencySeries and copy the results of the input
    FrequencySeries to it.
    """
    new = CreateCOMPLEX8FrequencySeries(fs.name, fs.epoch, fs.f0, fs.deltaF,
                                        fs.sampleUnits, fs.data.length)
    new.data.data[:] = fs.data.data[:]  # numpy automagic conversion
    return new


def compute_beta(tmplt):
    """ Calculate beta (thetaJL) using code from
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_8c_source.html#l06105
    """
    m1 = tmplt.m1
    m2 = tmplt.m2
    s1x = tmplt.spin1x
    s1y = tmplt.spin1y
    s1z = tmplt.spin1z
    s2x = tmplt.spin2x
    s2y = tmplt.spin2y
    s2z = tmplt.spin2z
    flow = tmplt.flow

    eta = m1 * m2 / (m1 + m2) / (m1 + m2);
    v0 = ((m1 + m2) * MTSUN_SI * PI * flow) ** (1. / 3.)

    lmag = (m1 + m2) * (m1 + m2) * eta / v0
    lmag *= (1.0 + v0 * v0 * (1.5 + eta / 6.))
    
    s1x = m1 * m1 * s1x
    s1y = m1 * m1 * s1y
    s1z = m1 * m1 * s1z
    s2x = m2 * m2 * s2x
    s2y = m2 * m2 * s2y
    s2z = m2 * m2 * s2z
    jx = s1x + s2x
    jy = s1y + s2y
    jz = lmag + s1z + s2z
    
    jnorm = (jx * jx + jy * jy + jz * jz) ** (1. / 2.)
    jhatz = jz / jnorm

    return np.arccos(jhatz)


def create_moments(df, flow, len_PSD):
    # moments go from flow to fmax unlike the PSD, which starts at 0
    n = len_PSD - int(flow / df)
    momI_0 = lal.CreateREAL8Vector(n)
    momI_2 = lal.CreateREAL8Vector(n)
    momI_3 = lal.CreateREAL8Vector(n)
    momI_4 = lal.CreateREAL8Vector(n)
    momI_5 = lal.CreateREAL8Vector(n)
    momI_6 = lal.CreateREAL8Vector(n)
    momI_7 = lal.CreateREAL8Vector(n)
    momI_8 = lal.CreateREAL8Vector(n)
    momI_9 = lal.CreateREAL8Vector(n)
    momI_10 = lal.CreateREAL8Vector(n)
    momI_11 = lal.CreateREAL8Vector(n)
    momI_12 = lal.CreateREAL8Vector(n)
    momI_13 = lal.CreateREAL8Vector(n)
    momI_14 = lal.CreateREAL8Vector(n)
    momI_15 = lal.CreateREAL8Vector(n)
    momI_16 = lal.CreateREAL8Vector(n)
    momJ_5 = lal.CreateREAL8Vector(n)
    momJ_6 = lal.CreateREAL8Vector(n)
    momJ_7 = lal.CreateREAL8Vector(n)
    momJ_8 = lal.CreateREAL8Vector(n)
    momJ_9 = lal.CreateREAL8Vector(n)
    momJ_10 = lal.CreateREAL8Vector(n)
    momJ_11 = lal.CreateREAL8Vector(n)
    momJ_12 = lal.CreateREAL8Vector(n)
    momJ_13 = lal.CreateREAL8Vector(n)
    momJ_14 = lal.CreateREAL8Vector(n)
    momK_10 = lal.CreateREAL8Vector(n)
    momK_11 = lal.CreateREAL8Vector(n)
    momK_12 = lal.CreateREAL8Vector(n)
    return (momI_0, momI_2, momI_3, momI_4, momI_5, momI_6, momI_7, momI_8,
            momI_9, momI_10, momI_11, momI_12, momI_13, momI_14, momI_15,
            momI_16, momJ_5, momJ_6, momJ_7, momJ_8, momJ_9, momJ_10, momJ_11,
            momJ_12, momJ_13, momJ_14, momK_10, momK_11, momK_12)


PI_p5 = PI**5


def compute_chirptimes(mchirp, eta, chi, flow):
    theta0 = (125. / 2. / (16. * PI * flow * mchirp * MTSUN_SI)**5)**(1./3)
    theta3 = (16. * PI_p5 / 25. * theta0 * theta0 / (eta * eta * eta))**0.2
    theta3s = 113. / (48 * PI) * chi * theta3
    return theta0, theta3, theta3s


def compute_tau0(mc, flow):
    return 5. * mc * MTSUN_SI / (256 * (PI * flow * mc * MTSUN_SI)**(8./3))


def compute_tau0_40(mc):
    return compute_tau0(mc, 40.)


class AlignedSpinTemplate(object):
    """
    Base class that handles whitening, normalization, and
    zero-padding/unpadding. Derived classes should:

    * Add __slots__ for whatever new data they need to store.
    * Add param_names and param_formats tuples to the class describing
      the parameters that uniquely specify a template.
    * Store template information during __init__.
    * Call Template.__init__(self) during __init__.
    * Provide a _compute_waveform method that takes df and f_final to generate
      the frequency-domain waveform as a COMPLEX16FrequencySeries.
    * Provide a classmethod from_sngl, which creates an instance based on
      a sngl_inspiral object.
    * Provide a classmethod to_sngl, which creates a sngl_inspiral object
      from an instance of the subclass.
    * Provide a classmethod from_sim, which creates an instance based on
      a sim_inspiral object.
    """
    approximant = None
    __slots__ = ("m1", "m2", "spin1z", "spin2z", "chieff", "bank", "tau0",
                 "_dur", "_mchirp", "_wf", "_metric", "sigmasq",
                 "is_seed_point", "_f_final", "_fhigh_max")
    param_names = ("m1", "m2", "spin1z", "spin2z")
    param_formats = ("%.2f", "%.2f", "%.2f", "%.2f")
    hdf_dtype = [('mass1', float32), ('mass2', float32),
                 ('spin1z', float32), ('spin2z', float32),
                 ('template_duration', float32), ('f_lower', float32),
                 ('approximant', 'S32')]

    def __init__(self, m1, m2, spin1z, spin2z, bank, flow=None, duration=None):

        self.m1 = float(m1)
        self.m2 = float(m2)
        self.spin1z = float(spin1z)
        self.spin2z = float(spin2z)
        self.chieff = lalsim.SimIMRPhenomBComputeChi(self.m1, self.m2,
                                                     self.spin1z, self.spin2z)
        self.bank = bank

        if flow is None:
            self.flow = bank.flow
            if bank.optimize_flow is not None:
                self.optimize_flow(bank.flow, bank.fhigh_max, bank.noise_model,
                                   sigma_frac=bank.optimize_flow)
        else:
            self.flow = flow

        self._wf = {}
        self._metric = None
        self.sigmasq = 0.
        self._mchirp = compute_mchirp(m1, m2)
        self.tau0_40 = compute_tau0_40(self._mchirp)
        self.tau0 = compute_tau0(self._mchirp, bank.flow)
        self._dur = duration
        self._f_final = None
        self._fhigh_max = bank.fhigh_max

    def optimize_flow(self, flow_min, fhigh_max, noise_model, df=0.1,
                      sigma_frac=0.99):
        """Set the template's flow as high as possible but still recovering
        at least the given fraction of the template's sigma when calculated
        from the minimum allowed flow. This avoids having unnecessarily long
        templates.
        """
        # compute the whitened waveform
        asd = get_ASD(df, flow_min, fhigh_max, noise_model)
        wf = self._compute_waveform(df, fhigh_max)
        if wf.data.length > len(asd):
            asd2 = np.ones(wf.data.length) * np.inf
            asd2[:len(asd)] = asd
        elif wf.data.length < len(asd):
            asd2 = asd[:wf.data.length]
        else:
            asd2 = asd
        wwf = wf.data.data / asd2
        # sum the power cumulatively from high to low frequencies
        integral = np.cumsum(np.flipud(wwf * wwf.conj()))
        ref_sigmasq = integral[-1]
        # find the frequency bin corresponding to the target loss
        i = np.searchsorted(integral, ref_sigmasq * sigma_frac ** 2)
        self.flow = (len(integral) - i) * df

    @property
    def params(self):
        return tuple(getattr(self, k) for k in self.param_names)

    @property
    def f_final(self):

        if self._f_final is None:
            f_final = self._get_f_final()
            if self._fhigh_max:
                f_final = min(f_final, self._fhigh_max)
            self._f_final = ceil_pow_2(f_final)

        return self._f_final

    @property
    def dur(self):

        if self._dur is None:
            self._dur = self._get_dur()
        return self._dur

    @dur.setter
    def dur(self, new_val):
        self._dur = new_val

    def _get_isco_f_final(self):
        return 6**-1.5 / (PI * (self.m1 + self.m2) * MTSUN_SI)  # ISCO

    def _get_chirp_dur(self):
        return lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(
            self.flow,
            self.m1 * MSUN_SI,
            self.m2 * MSUN_SI,
            self.chieff,
            7
        )

    def _get_imr_dur(self):
        """
        Following Jolien's suggestion for the duration of an IMR
        waveform, we compute the chirp time, scale by 10% and pad by
        one second.
        """
        # FIXME: Is a minimal time of 1.0s too long?
        return 1.1 * self._get_chirp_dur() + 1.0

    def _get_f_final(self):
        err_msg = "Template classes must provide a function _get_f_final "
        err_msg += "to compute the appropriate final frequency."
        raise NotImplementedError(err_msg)

    def _get_dur(self):
        err_msg = "Template classes must provide a function _get_dur "
        err_msg += "to compute template neighborhoods."
        raise NotImplementedError(err_msg)

    @classmethod
    def from_sim(cls, sim, bank):
        return cls(sim.mass1, sim.mass2, sim.spin1z, sim.spin2z, bank)

    @classmethod
    def from_sngl(cls, sngl, bank):
        return cls(sngl.mass1, sngl.mass2, sngl.spin1z, sngl.spin2z, bank)

    @classmethod
    def from_dict(cls, params, idx, bank):
        flow = float(params['f_lower'][idx])
        if not flow > 0:
            flow = None
        duration = float(params['template_duration'][idx])
        if not duration > 0:
            duration = None
        return cls(float(params['mass1'][idx]), float(params['mass2'][idx]),
                   float(params['spin1z'][idx]), float(params['spin2z'][idx]),
                   bank, flow=flow, duration=duration)

    def to_sngl(self):
        # All numerical values are initiated as 0 and all strings as ''
        row = SnglInspiralTable()

        row.mass1 = self.m1
        row.mass2 = self.m2
        row.mtotal = self.m1 + self.m2
        row.mchirp = self._mchirp
        row.eta = row.mass1 * row.mass2 / (row.mtotal * row.mtotal)
        row.tau0, row.tau3 = m1m2_to_tau0tau3(self.m1, self.m2, self.flow)
        row.template_duration = self.dur
        row.spin1z = self.spin1z
        row.spin2z = self.spin2z
        row.sigmasq = self.sigmasq
        if self.bank.flow_column:
            setattr(row, self.bank.flow_column, self.flow)

        return row

    def to_storage_arr(self):
        """Dump the template params to a numpy array."""
        new_tmplt = np.zeros(1, dtype=self.hdf_dtype)
        new_tmplt['mass1'] = self.m1
        new_tmplt['mass2'] = self.m2
        new_tmplt['spin1z'] = self.spin1z
        new_tmplt['spin2z'] = self.spin2z
        new_tmplt['template_duration'] = self.dur
        new_tmplt['f_lower'] = self.flow
        new_tmplt['approximant'] = self.approximant
        return new_tmplt

    def __repr__(self):
        return "(%s)" % ", ".join(self.param_formats) % self.params

    def finalize_as_template(self):
        """
        This template is being added to a bank. Compute metric coefficients
        or do whatever other expensive things you didn't want to do when
        it was just a proposal.
        """
        pass

    def _compute_waveform(self, df, f_final):

        phi0 = 0  # This is a reference phase, and not an intrinsic parameter
        approx = lalsim.GetApproximantFromString(self.approximant)

        if lalsim.SimInspiralImplementedFDApproximants(approx):
            hplus_fd, hcross_fd = lalsim.SimInspiralChooseFDWaveform(
                self.m1 * MSUN_SI, self.m2 * MSUN_SI,
                0., 0., self.spin1z, 0., 0., self.spin2z,
                1e6*PC_SI, 0., 0.,
                0., 0., 0.,
                df, self.flow, f_final, self.flow,
                None, approx)

        else:
            hplus_fd, hcross_fd = lalsim.SimInspiralFD(
                phi0, df, self.m1*MSUN_SI, self.m2*MSUN_SI, 0,
                0, self.spin1z, 0, 0, self.spin2z,
                1.e6*PC_SI, 0., 0.,
                0., 0., 0.,
                df, self.flow, f_final, 40.,
                None, approx)
        return hplus_fd

    def get_whitened_normalized(self, df, ASD=None, PSD=None):
        """
        Return a COMPLEX8FrequencySeries of the waveform, whitened by the
        given ASD and normalized. The waveform is not zero-padded to
        match the length of the ASD, so its normalization depends on
        its own length.
        """
        if df not in self._wf:
            wf = self._compute_waveform(df, self.f_final)
            if ASD is None:
                ASD = PSD**0.5
            if wf.data.length > len(ASD):
                ASD2 = np.ones(wf.data.length) * np.inf
                ASD2[:len(ASD)] = ASD
                ASD = ASD2
            arr_view = wf.data.data

            # whiten
            arr_view[:] /= ASD[:wf.data.length]
            arr_view[:int(self.flow / df)] = 0.
            arr_view[int(self.f_final/df):wf.data.length] = 0.

            # normalize
            self.sigmasq = compute_sigmasq(arr_view, df)
            arr_view[:] /= self.sigmasq**0.5

            # down-convert to single precision
            self._wf[df] = FrequencySeries_to_COMPLEX8FrequencySeries(wf)
        return self._wf[df]

    def metric_match(self, other, df, **kwargs):
        raise NotImplementedError

    def brute_match(self, other, df, workspace_cache, **kwargs):
        return SBankComputeMatch(self.get_whitened_normalized(df, **kwargs),
                                 other.get_whitened_normalized(df, **kwargs),
                                 workspace_cache[0])

    def clear(self):
        self._wf = {}


class IMRAlignedSpinTemplate(AlignedSpinTemplate):
    """
    A convenience class for IMR aligned spin templates. Specific
    implementations of IMR aligned spin templates should sub-class
    this class.
    """

    def _get_dur(self):
        return self._get_imr_dur()

    def _get_f_final(self):
        # assume IMR waveforms have their own physical termination
        return self._fhigh_max or 4096.


class InspiralAlignedSpinTemplate(AlignedSpinTemplate):
    """
    A convenience class for inspiral-only aligned spin
    templates. Specific implementations of inspiral-only aligned spin
    templates should sub-class this class.
    """
    __slots__ = ("chired")

    def __init__(self, m1, m2, spin1z, spin2z, bank, flow=None, duration=None):

        self.chired = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(
            m1,
            m2,
            spin1z,
            spin2z
        )
        AlignedSpinTemplate.__init__(self, m1, m2, spin1z, spin2z, bank,
                                     flow=flow, duration=duration)

    def _get_dur(self):
        return self._get_chirp_dur()

    def _get_f_final(self):
        return self._get_isco_f_final()


class IMRPhenomBTemplate(IMRAlignedSpinTemplate):
    approximant = "IMRPhenomB"
    param_names = ("m1", "m2", "chieff")
    param_formats = ("%.2f", "%.2f", "%+.2f")


class IMRPhenomCTemplate(IMRPhenomBTemplate):
    approximant = "IMRPhenomC"


class IMRPhenomDTemplate(IMRAlignedSpinTemplate):
    approximant = "IMRPhenomD"

    def _get_dur(self):
        dur = lalsim.SimIMRPhenomDChirpTime(self.m1 * MSUN_SI,
                                            self.m2 * MSUN_SI, self.spin1z,
                                            self.spin2z, self.flow)
        # add a 10% to be consistent with PyCBC's duration estimate,
        # may want to FIXME if that changes
        return dur * 1.1


class SEOBNRv2Template(IMRAlignedSpinTemplate):
    approximant = "SEOBNRv2"

    def _get_dur(self):
        seff = lalsim.SimIMRPhenomBComputeChi(self.m1, self.m2,
                                              self.spin1z, self.spin2z)
        dur = lalsim.SimIMRSEOBNRv2ChirpTimeSingleSpin(
            self.m1 * MSUN_SI, self.m2 * MSUN_SI, seff, self.flow)
        # add a 10% to be consistent with PyCBC's duration estimate,
        # may want to FIXME if that changes
        return dur * 1.1


class SEOBNRv2ROMDoubleSpinTemplate(SEOBNRv2Template):
    approximant = "SEOBNRv2_ROM_DoubleSpin"


class SEOBNRv2ROMDoubleSpinHITemplate(SEOBNRv2Template):
    approximant = "SEOBNRv2_ROM_DoubleSpin_HI"


class SEOBNRv4Template(IMRAlignedSpinTemplate):
    approximant = "SEOBNRv4"

    def _get_dur(self):
        dur = lalsim.SimIMRSEOBNRv4ROMTimeOfFrequency(
            self.flow, self.m1 * MSUN_SI, self.m2 * MSUN_SI,
            self.spin1z, self.spin2z)
        # Allow a 10% margin of error
        return dur * 1.1


class SEOBNRv4ROMTemplate(SEOBNRv4Template):
    approximant = "SEOBNRv4_ROM"


class EOBNRv2Template(SEOBNRv2Template):
    approximant = "EOBNRv2"
    param_names = ("m1", "m2")
    param_formats = ("%.2f", "%.2f")

    def __init__(self, m1, m2, bank, flow=None, duration=None):
        # Use everything from SEOBNRv2Template class except call
        # parent __init__ with spins set to zero
        SEOBNRv2Template.__init__(self, m1, m2, 0, 0, bank, flow=flow,
                                  duration=duration)


class TaylorF2RedSpinTemplate(InspiralAlignedSpinTemplate):
    approximant = "TaylorF2RedSpin"
    param_names = ("m1", "m2", "chired")
    param_formats = ("%.5f", "%.5f", "%+.4f")

    __slots__ = ("chired", "tau0", "_dur", "_mchirp", "_eta", "_theta0",
                 "_theta3", "_theta3s")

    def __init__(self, m1, m2, spin1z, spin2z, bank, flow=None, duration=None):

        warn_msg = ("DEPRECATION WARNING: It is not a good idea to use this "
                    "approximant. The TaylorF2 approximant is the best thing "
                    "to use if you are constructing a 'brute-force match' "
                    "bank. This allows the use of both spins correctly, and "
                    "the inclusion of all PN terms currently known. "
                    "If using the metric, the implementation in PyCBC, which "
                    "gets terms directly from lalsimulation, and so "
                    "automatically stays up to date, is more accurate.")
        logging.warn(warn_msg)
        AlignedSpinTemplate.__init__(self, m1, m2, spin1z, spin2z, bank,
                                     flow=flow, duration=duration)
        self.chired = lalsim.SimInspiralTaylorF2ReducedSpinComputeChi(
            m1,
            m2,
            spin1z,
            spin2z
        )
        self._eta = m1*m2/(m1+m2)**2
        self._theta0, self._theta3, self._theta3s = compute_chirptimes(
            self._mchirp,
            self._eta,
            self.chired,
            self.flow
        )

    def finalize_as_template(self):
        if not self.bank.use_metric:
            return

        df, PSD = get_neighborhood_PSD([self], self.flow,
                                       self.bank.noise_model)

        curr_len = (len(PSD) - self.flow // df)
        length_check = curr_len > self.bank._moments[df][0].length
        if df not in self.bank._moments or length_check:
            real8vector_psd = CreateREAL8Vector(len(PSD))
            real8vector_psd.data[:] = PSD
            self.bank._moments[df] = create_moments(df, self.flow, len(PSD))
            lalsim.SimInspiralTaylorF2RedSpinComputeNoiseMoments(
                *(self.bank._moments[df] + (real8vector_psd, self.flow, df))
            )

        self._metric = lalsim.SimInspiralTaylorF2RedSpinMetricChirpTimes(
            self._theta0,
            self._theta3,
            self._theta3s,
            self.flow,
            df,
            *self.bank._moments[df]
        )
        if isnan(self._metric[0]):
            raise ValueError("g00 is nan")

    def metric_match(self, other, df, **kwargs):
        g00, g01, g02, g11, g12, g22 = self._metric
        dx0 = other._theta0 - self._theta0
        dx1 = other._theta3 - self._theta3
        dx2 = other._theta3s - self._theta3s
        match = 1 - (g00 * dx0 * dx0 + 2 * g01 * dx0 * dx1
                     + 2 * g02 * dx0 * dx2 + g11 * dx1 * dx1
                     + 2 * g12 * dx1 * dx2 + g22 * dx2 * dx2)

        return match


class TaylorF2Template(InspiralAlignedSpinTemplate):
    approx_name = "TaylorF2"

    def _compute_waveform(self, df, f_final):
        phi0 = 0  # This is a reference phase, and not an intrinsic parameter
        LALpars = lal.CreateDict()
        approx = lalsim.GetApproximantFromString(self.approx_name)
        hplus_fd, hcross_fd = lalsim.SimInspiralChooseFDWaveform(
            self.m1*MSUN_SI, self.m2*MSUN_SI,
            0., 0., self.spin1z,
            0., 0., self.spin2z,
            1.e6*PC_SI, 0., phi0,
            0., 0., 0.,
            df, self.flow, f_final, self.flow,
            LALpars, approx)

        # Must set values greater than _get_f_final to 0
        act_f_max = self._get_f_final()
        f_max_idx = int(act_f_max / df + 0.999)
        hplus_fd.data.data[f_max_idx:] = 0

        return hplus_fd


class PrecessingSpinTemplate(AlignedSpinTemplate):
    """
    A generic class for precessing templates. These models require the
    full fifteen-dimensional parameter space to specify the observed
    signal in the detector.
    """
    param_names = ("m1", "m2", "spin1x", "spin1y", "spin1z", "spin2x",
                   "spin2y", "spin2z", "theta", "phi", "iota", "psi")
    param_formats = ("%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f",
                     "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f")
    __slots__ = param_names + ("bank", "chieff", "chipre", "tau0", "_dur",
                               "_mchirp", "_wf_hp", "_wf_hc", "_hpsigmasq",
                               "_hcsigmasq", "_hphccorr")
    hdf_dtype = AlignedSpinTemplate.hdf_dtype + \
        [('spin1x', float32), ('spin1y', float32), ('spin2x', float32),
         ('spin2y', float32), ('latitude', float32), ('longitude', float32),
         ('polarization', float32), ('inclination', float32),
         ('orbital_phase', float32)]

    def __init__(self, m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                 theta, phi, iota, psi, orb_phase, bank, flow=None,
                 duration=None):

        AlignedSpinTemplate.__init__(self, m1, m2, spin1z, spin2z, bank,
                                     flow=flow, duration=duration)
        self.spin1x = float(spin1x)
        self.spin1y = float(spin1y)
        self.spin2x = float(spin2x)
        self.spin2y = float(spin2y)

        self.theta = float(theta)
        self.phi = float(phi)
        self.iota = float(iota)
        self.psi = float(psi)
        self.orb_phase = float(orb_phase)

        outs = lalsim.SimIMRPhenomPCalculateModelParametersFromSourceFrame(
            self.m1,
            self.m2,
            self.flow,
            self.orb_phase,
            self.iota,
            self.spin1x,
            self.spin1y,
            self.spin1z,
            self.spin2x,
            self.spin2y,
            self.spin2z,
            lalsim.IMRPhenomPv2_V
        )
        self.chieff = outs[0]
        self.chipre = outs[1]

        self._wf = {}
        self._metric = None
        self.sigmasq = 0.
        self._wf_hp = {}
        self._wf_hc = {}
        self._hpsigmasq = {}
        self._hcsigmasq = {}
        self._hphccorr = {}

    @classmethod
    def from_sim(cls, sim, bank):
        # theta = polar angle wrt overhead
        #       = pi/2 - latitude (which is 0 on the horizon)
        return cls(sim.mass1, sim.mass2, sim.spin1x, sim.spin1y, sim.spin1z,
                   sim.spin2x, sim.spin2y, sim.spin2z, np.pi/2 - sim.latitude,
                   sim.longitude, sim.inclination, sim.polarization,
                   sim.coa_phase, bank)

    def _compute_waveform_comps(self, df, f_final):
        approx = lalsim.GetApproximantFromString(self.approximant)
        if lalsim.SimInspiralImplementedFDApproximants(approx):
            hplus_fd, hcross_fd = lalsim.SimInspiralChooseFDWaveform(
                self.m1*MSUN_SI, self.m2*MSUN_SI,
                self.spin1x, self.spin1y, self.spin1z,
                self.spin2x, self.spin2y, self.spin2z,
                1.e6*PC_SI, self.iota, self.orb_phase,
                0., 0., 0.,
                df, self.flow, f_final, self.flow,
                None, approx)
        else:
            hplus_fd, hcross_fd = lalsim.SimInspiralFD(
                self.m1*MSUN_SI, self.m2*MSUN_SI,
                self.spin1x, self.spin1y, self.spin1z,
                self.spin2x, self.spin2y, self.spin2z,
                1.e6*PC_SI, self.iota, self.orb_phase,
                0., 0., 0.,
                df, self.flow, f_final, self.flow,
                None, approx)

        return hplus_fd, hcross_fd

    def _compute_waveform(self, df, f_final):
        hplus_fd, hcross_fd = self._compute_waveform_comps(df, f_final)

        # project onto detector
        return project_hplus_hcross(hplus_fd, hcross_fd, self.theta, self.phi,
                                    self.psi)

    def get_whitened_normalized_comps(self, df, ASD=None, PSD=None):
        """
        Return a COMPLEX8FrequencySeries of h+ and hx, whitened by the
        given ASD and normalized. The waveform is not zero-padded to
        match the length of the ASD, so its normalization depends on
        its own length.
        """
        if df not in self._wf_hp:
            # Clear self._wf as it won't be needed any more if calling here
            self._wf = {}
            # Generate a new wf
            hp, hc = self._compute_waveform_comps(df, self.f_final)
            if ASD is None:
                ASD = PSD**0.5
            if hp.data.length > len(ASD):
                err_msg = "waveform has length greater than ASD; cannot whiten"
                raise ValueError(err_msg)
            arr_view_hp = hp.data.data
            arr_view_hc = hc.data.data

            # Whiten
            arr_view_hp[:] /= ASD[:hp.data.length]
            arr_view_hp[:int(self.flow / df)] = 0.
            arr_view_hp[int(self.f_final/df):hp.data.length] = 0.

            arr_view_hc[:] /= ASD[:hc.data.length]
            arr_view_hc[:int(self.flow / df)] = 0.
            arr_view_hc[int(self.f_final/df):hc.data.length] = 0.

            # Get normalization factors and normalize
            self._hpsigmasq[df] = compute_sigmasq(arr_view_hp, df)
            self._hcsigmasq[df] = compute_sigmasq(arr_view_hc, df)
            arr_view_hp[:] /= self._hpsigmasq[df]**0.5
            arr_view_hc[:] /= self._hcsigmasq[df]**0.5

            self._hphccorr[df] = compute_correlation(arr_view_hp, arr_view_hc,
                                                     df)

            self._wf_hp[df] = FrequencySeries_to_COMPLEX8FrequencySeries(hp)
            self._wf_hc[df] = FrequencySeries_to_COMPLEX8FrequencySeries(hc)

        return self._wf_hp[df], self._wf_hc[df], self._hphccorr[df]

    def brute_match(self, other, df, workspace_cache, **kwargs):

        # Template generates hp and hc
        hp, hc, hphccorr = self.get_whitened_normalized_comps(df, **kwargs)

        # Proposal generates h(t), sky loc is later discarded.
        try:
            proposal = other.get_whitened_normalized(df, **kwargs)
        except RuntimeError:
            # Waveform generation failed. Bad, but try to carry on
            return 1

        # maximize over sky position of template
        return SBankComputeMatchSkyLoc(hp, hc, hphccorr,
                                       proposal, workspace_cache[0],
                                       workspace_cache[1],
                                       phase_maximized=True)

    @classmethod
    def from_sngl(cls, sngl, bank):
        # FIXME: Using alpha columns to hold theta, phi, iota, psi
        return cls(sngl.mass1, sngl.mass2, sngl.spin1x, sngl.spin1y,
                   sngl.spin1z, sngl.spin2x, sngl.spin2y, sngl.spin2z,
                   sngl.alpha1, sngl.alpha2, sngl.alpha3, sngl.alpha4,
                   sngl.alpha5, bank)

    @classmethod
    def from_dict(cls, params, idx, bank):
        flow = float(params['f_lower'][idx])
        if not flow > 0:
            flow = None
        duration = float(params['template_duration'][idx])
        if not duration > 0:
            duration = None
        return cls(params['mass1'][idx], params['mass2'][idx],
                   params['spin1x'][idx], params['spin1y'][idx],
                   params['spin1z'][idx], params['spin2x'][idx],
                   params['spin2y'][idx], params['spin2z'][idx],
                   params['latitude'][idx], params['longitude'][idx],
                   params['polarization'][idx], params['inclination'][idx],
                   params['orbital_phase'][idx], bank,
                   flow=flow, duration=duration)

    def to_sngl(self):
        # All numerical values are initiated as 0 and all strings as ''
        row = SnglInspiralTable()
        row.mass1 = self.m1
        row.mass2 = self.m2
        row.mtotal = self.m1 + self.m2
        row.mchirp = self._mchirp
        row.eta = row.mass1 * row.mass2 / (row.mtotal * row.mtotal)
        row.tau0, row.tau3 = m1m2_to_tau0tau3(self.m1, self.m2, self.flow)
        row.template_duration = self.dur
        row.spin1x = self.spin1x
        row.spin1y = self.spin1y
        row.spin1z = self.spin1z
        row.spin2x = self.spin2x
        row.spin2y = self.spin2y
        row.spin2z = self.spin2z
        row.alpha1 = self.theta
        row.alpha2 = self.phi
        row.alpha3 = self.iota
        row.alpha4 = self.psi
        row.alpha5 = self.orb_phase
        row.sigmasq = self.sigmasq
        if self.bank.flow_column:
            setattr(row, self.bank.flow_column, self.flow)
        return row

    def to_storage_arr(self):
        """Dump the template params to a numpy array."""
        new_tmplt = super(PrecessingSpinTemplate, self).to_storage_arr()
        new_tmplt['spin1x'] = self.spin1x
        new_tmplt['spin1y'] = self.spin1y
        new_tmplt['spin2x'] = self.spin2x
        new_tmplt['spin2y'] = self.spin2y
        new_tmplt['latitude'] = self.theta
        new_tmplt['longitude'] = self.phi
        new_tmplt['polarization'] = self.psi
        new_tmplt['inclination'] = self.iota
        new_tmplt['orbital_phase'] = self.orb_phase
        return new_tmplt


class IMRPrecessingSpinTemplate(PrecessingSpinTemplate):
    """
    A convenience class for IMR precessing spin templates. Specific
    implementations of IMR precessing spin templates should sub-class
    this class.
    """

    def _get_dur(self):
        return self._get_imr_dur()

    def _get_f_final(self):
        # assume IMR waveforms have their own physical termination
        return self._fhigh_max or 4096.


class InspiralPrecessingSpinTemplate(PrecessingSpinTemplate):
    """
    A convenience class for inpsiral-only precessing spin templates. Specific
    implementations of inspiral-only precessing spin templates should sub-class
    this class.
    """

    def _get_dur(self):
        return self._get_chirp_dur()

    def _get_f_final(self):
        return self._get_isco_f_final()


class SpinTaylorF2Template(InspiralPrecessingSpinTemplate):
    approximant = "SpinTaylorF2"

    def __init__(self, m1, m2, spin1x, spin1y, spin1z,
                 theta, phi, iota, psi, orb_phase, bank, flow=None,
                 duration=None):
        super(SpinTaylorF2Template, self).__init__(m1, m2, spin1x, spin1y,
                                                   spin1z, 0, 0, 0, theta, phi,
                                                   iota, psi, orb_phase, bank,
                                                   flow=flow, duration=None)

    def _compute_waveform_comps(self, df, f_final):
        hplus_fd, hcross_fd = \
            super(SpinTaylorF2Template, self)._compute_waveform_comps(df,
                                                                      f_final)
        # Must set values greater than _get_f_final to 0
        act_f_max = self._get_f_final()
        f_max_idx = int(act_f_max / df + 0.999)
        hplus_fd.data.data[f_max_idx:] = 0
        hcross_fd.data.data[f_max_idx:] = 0

        return hplus_fd, hcross_fd

    @classmethod
    def from_sngl(cls, sngl, bank):
        # FIXME: Using alpha columns to hold theta, phi, iota, psi
        assert sngl.spin2x == sngl.spin2y == sngl.spin2z == 0
        return cls(sngl.mass1, sngl.mass2, sngl.spin1x, sngl.spin1y,
                   sngl.spin1z, sngl.alpha1, sngl.alpha2, sngl.alpha3,
                   sngl.alpha4, sngl.alpha5, bank)

    @classmethod
    def from_dict(cls, hdf_fp, idx, bank):
        raise NotImplementedError('Please write this!')

    def to_sngl(self):
        # All numerical values are initiated as 0 and all strings as ''
        row = SnglInspiralTable()
        row.mass1 = self.m1
        row.mass2 = self.m2
        row.mtotal = self.m1 + self.m2
        row.mchirp = self._mchirp
        row.eta = row.mass1 * row.mass2 / (row.mtotal * row.mtotal)
        row.tau0, row.tau3 = m1m2_to_tau0tau3(self.m1, self.m2, self.flow)
        row.template_duration = self.dur
        row.spin1x = self.spin1x
        row.spin1y = self.spin1y
        row.spin1z = self.spin1z
        row.spin2x = 0
        row.spin2y = 0
        row.spin2z = 0
        row.alpha1 = self.theta
        row.alpha2 = self.phi
        row.alpha3 = self.iota
        row.alpha4 = self.psi
        row.alpha5 = self.orb_phase
        row.sigmasq = self.sigmasq
        if self.bank.flow_column:
            setattr(row, self.bank.flow_column, self.flow)
        return row


class SpinTaylorT5FourierTemplate(InspiralPrecessingSpinTemplate):
    approximant = "SpinTaylorT5Fourier"


class SpinTaylorT4Template(InspiralPrecessingSpinTemplate):
    approximant = "SpinTaylorT4"


class SpinTaylorT5Template(InspiralPrecessingSpinTemplate):
    approximant = "SpinTaylorT5"


class SEOBNRv3Template(IMRPrecessingSpinTemplate):
    approximant = "SEOBNRv3"


class IMRPhenomPTemplate(IMRPrecessingSpinTemplate):
    approximant = "IMRPhenomP"


class IMRPhenomPv2Template(IMRPrecessingSpinTemplate):
    approximant = "IMRPhenomPv2"


class IMRPhenomPv3Template(IMRPrecessingSpinTemplate):
    approximant = "IMRPhenomPv3"

class IMRPhenomPv2THATemplate(IMRPrecessingSpinTemplate):
    """
    """
    param_names = ("m1", "m2", "spin1x", "spin1y", "spin1z", "spin2x",
                   "spin2y", "spin2z", "theta", "phi", "iota", "psi",
                   "orb_phase", "num_comps")
    param_formats = ("%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f",
                     "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f",
                     "%.2f", "%d")
    __slots__ = param_names + ("bank", "chieff", "chipre", "tau0", "_dur",
                               "_mchirp", "_wf_hp", "_wf_hc", "_hpsigmasq",
                               "_hcsigmasq", "_hphccorr")
    hdf_dtype = AlignedSpinTemplate.hdf_dtype + \
        [('spin1x', float32), ('spin1y', float32), ('spin2x', float32),
         ('spin2y', float32), ('latitude', float32), ('longitude', float32),
         ('polarization', float32), ('inclination', float32),
         ('orbital_phase', float32), ('num_comps', int16)]

    #param_names = ("m1", "m2", "spin1x", "spin1y", "spin1z", "spin2x",
    #               "spin2y", "spin2z", "thetaJN",
    #               "alpha0", "phi0", "theta", "phi", "psi")
    #param_formats = ("%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f",
    #                 "%.2f", "%.2f", "%.2f", "%.2f", "%.2f")
    #__slots__ = param_names + ("bank", "tau0", "_dur",
    #                           "_mchirp", "_wf_hp", "_wf_hc", "_hpsigmasq",
    #                           "_hcsigmasq", "_hphccorr")
    #hdf_dtype = [('m1', float32), ('m2', float32), ('chi1_l', float32),
    #             ('chi2_l', float32), ('chip', float32), ('thetaJN', float32),
    #             ('alpha0', float32), ('phi0', float32), ('theta', float32),
    #             ('phi', float32), ('psi', float32)]
    approximant = "IMRPhenomPv2"

    def __init__(self, m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                 theta, phi, iota, psi, orb_phase, bank, flow=None,
                 duration=None, num_comps=5):


        AlignedSpinTemplate.__init__(self, m1, m2, spin1z, spin2z, bank,
                                     flow=flow, duration=duration)

        self.spin1x = float(spin1x)
        self.spin1y = float(spin1y)
        self.spin2x = float(spin2x)
        self.spin2y = float(spin2y)

        self.theta = float(theta)
        self.phi = float(phi)
        self.iota = float(iota)
        self.psi = float(psi)
        self.orb_phase = float(orb_phase)

        self.num_comps = int(num_comps)

        outs = lalsim.SimIMRPhenomPCalculateModelParametersFromSourceFrame(
            self.m1, 
            self.m2,
            self.flow,
            self.orb_phase,
            self.iota,
            self.spin1x, 
            self.spin1y,
            self.spin1z,
            self.spin2x,
            self.spin2y,
            self.spin2z,
            lalsim.IMRPhenomPv2_V
        )          
        chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz = outs

        self.chi1_l = float(chi1_l)
        self.chi2_l = float(chi2_l)
        self.chip = float(chip)
        self.thetaJN = float(thetaJN)
        self.alpha0 = float(alpha0)
        self.phi0 = float(phi_aligned)
        self.theta = float(theta)
        self.phi = float(phi)
        self.psi = float(psi)
        # This is a correction on psi, currently unused
        self.psi_corr = zeta_polariz

        self.beta = compute_beta(self)

        self._wf = {}
        self._metric = None
        self.sigmasq = 0.
        self._wf_h1 = {}
        self._wf_h2 = {}
        self._wf_h3 = {}
        self._wf_h4 = {}
        self._wf_h5 = {}
        self._h1sigmasq = {}
        self._h2sigmasq = {}
        self._h3sigmasq = {}
        self._h4sigmasq = {}
        self._h5sigmasq = {}

    def _compute_waveform_five_comps(self, df, f_final):
        def gen_phenom_p_comp(thetaJN, alpha0, phi0):
            return lalsim.SimIMRPhenomP(
                self.chi1_l,
                self.chi2_l,
                self.chip,
                thetaJN,
                self.m1*MSUN_SI,
                self.m2*MSUN_SI,
                1.e6*PC_SI,
                alpha0,
                phi0,
                df,
                self.flow,
                f_final,
                self.flow,
                lalsim.IMRPhenomPv2_V,
                lalsim.NoNRT_V,
                None
            )

        hgen1a, _  = gen_phenom_p_comp(0., 0., 0.)
        # hgen1b is negative w.r.t. 1908.05707
        _, hgen1b = gen_phenom_p_comp(0., 0., np.pi/4.)
        # These are both negative w.r.t 1908.05707
        _, hgen2a = gen_phenom_p_comp(np.pi/2., 0., np.pi/4.)
        _, hgen2b = gen_phenom_p_comp(np.pi/2., np.pi/2., 0.)
        hgen3a, _ = gen_phenom_p_comp(np.pi/2., 0., 0.)
        hgen3b, _ = gen_phenom_p_comp(np.pi/2., np.pi/2., 0.)

        # Edit these arrays in place to avoid defining new LAL arrays
        tmp = hgen1a.data.data[:] + hgen1b.data.data[:]
        hgen1b.data.data[:] = (hgen1a.data.data[:] - hgen1b.data.data[:])/2.
        hgen1a.data.data[:] = tmp / 2.
        h1 = hgen1a
        h5 = hgen1b
        
        tmp = hgen2a.data.data[:] + hgen2b.data.data[:]
        hgen2b.data.data[:] = 0.25 * (hgen2a.data.data[:] - hgen2b.data.data[:])
        hgen2a.data.data[:] = 0.25 * tmp
        h2 = hgen2a
        h4 = hgen2b
        hgen3a.data.data[:] = \
            1./6. * (hgen3a.data.data[:] + hgen3b.data.data[:])
        h3 = hgen3a
        
        hs = (h1, h2, h3, h4, h5)

        return hs

    def get_whitened_normalized_comps(self, df, ASD=None, PSD=None):
        """
        Return a COMPLEX8FrequencySeries of h+ and hx, whitened by the
        given ASD and normalized. The waveform is not zero-padded to
        match the length of the ASD, so its normalization depends on
        its own length.
        """
        if df not in self._wf_h1:
            # Clear self._wf as it won't be needed any more if calling here
            self._wf = {}
            # Generate a new wf
            h1, h2, h3, h4, h5 = self._compute_waveform_five_comps(df, self.f_final)
            if ASD is None:
                ASD = PSD**0.5
            if h1.data.length > len(ASD):
                err_msg = "waveform has length greater than ASD; cannot whiten"
                raise ValueError(err_msg)
            arr_view_h1 = h1.data.data
            arr_view_h2 = h2.data.data
            arr_view_h3 = h3.data.data
            arr_view_h4 = h4.data.data
            arr_view_h5 = h5.data.data

            # Whiten
            arr_view_h1[:] /= ASD[:h1.data.length]
            arr_view_h1[:int(self.flow / df)] = 0.
            arr_view_h1[int(self.f_final/df):h1.data.length] = 0.

            arr_view_h2[:] /= ASD[:h2.data.length]
            arr_view_h2[:int(self.flow / df)] = 0.
            arr_view_h2[int(self.f_final/df):h2.data.length] = 0.

            arr_view_h3[:] /= ASD[:h3.data.length]
            arr_view_h3[:int(self.flow / df)] = 0.
            arr_view_h3[int(self.f_final/df):h3.data.length] = 0.

            arr_view_h4[:] /= ASD[:h4.data.length]
            arr_view_h4[:int(self.flow / df)] = 0.
            arr_view_h4[int(self.f_final/df):h4.data.length] = 0.

            arr_view_h5[:] /= ASD[:h5.data.length]
            arr_view_h5[:int(self.flow / df)] = 0.
            arr_view_h5[int(self.f_final/df):h5.data.length] = 0.

            # Get normalization factors and normalize
            self._h1sigmasq[df] = compute_sigmasq(arr_view_h1, df)
            self._h2sigmasq[df] = compute_sigmasq(arr_view_h2, df)
            self._h3sigmasq[df] = compute_sigmasq(arr_view_h3, df)
            self._h4sigmasq[df] = compute_sigmasq(arr_view_h4, df)
            self._h5sigmasq[df] = compute_sigmasq(arr_view_h5, df)
            arr_view_h1[:] /= self._h1sigmasq[df]**0.5
            arr_view_h2[:] /= self._h2sigmasq[df]**0.5
            arr_view_h3[:] /= self._h3sigmasq[df]**0.5
            arr_view_h4[:] /= self._h4sigmasq[df]**0.5
            arr_view_h5[:] /= self._h5sigmasq[df]**0.5

            h1h2corr = compute_complex_correlation(arr_view_h1, arr_view_h2, df)
            h1h3corr = compute_complex_correlation(arr_view_h1, arr_view_h3, df)
            h1h4corr = compute_complex_correlation(arr_view_h1, arr_view_h4, df)
            h1h5corr = compute_complex_correlation(arr_view_h1, arr_view_h5, df)

            arr_view_h2[:] = (
                (arr_view_h2[:] - h1h2corr * arr_view_h1[:])
                / (1. - h1h2corr * h1h2corr.conj()) ** 0.5
            )

            arr_view_h3[:] = (
                (arr_view_h3[:] - h1h3corr * arr_view_h1[:])
                / (1. - h1h3corr * h1h3corr.conj()) ** 0.5
            )
            arr_view_h4[:] = (
                (arr_view_h4[:] - h1h4corr * arr_view_h1[:])
                / (1. - h1h4corr * h1h4corr.conj()) ** 0.5
            )
            arr_view_h5[:] = (
                (arr_view_h5[:] - h1h5corr * arr_view_h1[:])
                / (1. - h1h5corr * h1h5corr.conj()) ** 0.5
            )

            h2h3corr = compute_complex_correlation(arr_view_h2, arr_view_h3, df)
            h2h4corr = compute_complex_correlation(arr_view_h2, arr_view_h4, df)
            h2h5corr = compute_complex_correlation(arr_view_h2, arr_view_h5, df)

            arr_view_h3[:] = (
                (arr_view_h3[:] - h2h3corr * arr_view_h2[:])
                / (1. - h2h3corr * h2h3corr.conj()) ** 0.5
            )
            arr_view_h4[:] = (
                (arr_view_h4[:] - h2h4corr * arr_view_h2[:])
                / (1. - h2h4corr * h2h4corr.conj()) ** 0.5
            )
            arr_view_h5[:] = (
                (arr_view_h5[:] - h2h5corr * arr_view_h2[:])
                / (1. - h2h5corr * h2h5corr.conj()) ** 0.5
            )

            h3h4corr = compute_complex_correlation(arr_view_h3, arr_view_h4, df)
            h3h5corr = compute_complex_correlation(arr_view_h3, arr_view_h5, df)

            arr_view_h4[:] = (
                (arr_view_h4[:] - h3h4corr * arr_view_h3[:])
                / (1. - h3h4corr * h3h4corr.conj()) ** 0.5
            )
            arr_view_h5[:] = (
                (arr_view_h5[:] - h3h5corr * arr_view_h3[:])
                / (1. - h3h5corr * h3h5corr.conj()) ** 0.5
            )

            h4h5corr = compute_complex_correlation(arr_view_h4, arr_view_h5, df)

            arr_view_h5[:] = (
                (arr_view_h5[:] - h4h5corr * arr_view_h4[:])
                / (1. - h4h5corr * h4h5corr.conj()) ** 0.5
            )

            self._wf_h1[df] = FrequencySeries_to_COMPLEX8FrequencySeries(h1)
            self._wf_h2[df] = FrequencySeries_to_COMPLEX8FrequencySeries(h2)
            self._wf_h3[df] = FrequencySeries_to_COMPLEX8FrequencySeries(h3)
            self._wf_h4[df] = FrequencySeries_to_COMPLEX8FrequencySeries(h4)
            self._wf_h5[df] = FrequencySeries_to_COMPLEX8FrequencySeries(h5)

        return (self._wf_h1[df], self._wf_h2[df], self._wf_h3[df],
                self._wf_h4[df], self._wf_h5[df])

    def brute_match(self, other, df, workspace_cache, num_comps=None, **kwargs):

        if num_comps is None:
            num_comps = self.num_comps

        if (num_comps < 1) or (num_comps > 5):
            raise ValueError("num_comps must be between 1 and 5")

        # Template generates hp and hc
        h1, h2, h3, h4, h5 = self.get_whitened_normalized_comps(df, **kwargs)

        # Proposal generates h(t), sky loc is later discarded.
        try:
            proposal = other.get_whitened_normalized(df, **kwargs)
        except RuntimeError:
            # Waveform generation failed. Bad, but try to carry on
            return 1

        # maximize over sky position of template
        value = SBankComputeFiveCompMatch(
            h1, h2, h3, h4, h5,
            proposal, num_comps, workspace_cache[0],
            workspace_cache[1], workspace_cache[2],
            workspace_cache[3], workspace_cache[4]
        )

        return value

    def brute_comp_match(self, other, factors, df, workspace_cache, num_comps=None, **kwargs):

        if num_comps is None:
            num_comps = self.num_comps

        if (num_comps < 1) or (num_comps > 5):
            raise ValueError("num_comps must be between 1 and 5")

        # Template generates hp and hc
        h1, h2, h3, h4, h5 = self.get_whitened_normalized_comps(df, **kwargs)

        try:
            p1, p2, p3, p4, p5 = other.get_whitened_normalized_comps(df, **kwargs)
        except RuntimeError:
            # Waveform generation failed. Bad, but try to carry on
            return 1

        # maximize over sky position of template
        value = SBankComputeFiveCompFactorMatch(
            h1, h2, h3, h4, h5,
            p1, p2, p3, p4, p5,
            factors[0], factors[1],
            factors[2], factors[3],
            factors[4], num_comps,
            workspace_cache[0], workspace_cache[1],
            workspace_cache[2], workspace_cache[3],
            workspace_cache[4], workspace_cache[5],
            workspace_cache[6], workspace_cache[7],
            workspace_cache[8], workspace_cache[9],
            workspace_cache[10], workspace_cache[11],
            workspace_cache[12], workspace_cache[13],
            workspace_cache[14], workspace_cache[15],
            workspace_cache[16], workspace_cache[17],
            workspace_cache[18], workspace_cache[19],
            workspace_cache[20], workspace_cache[21],
            workspace_cache[22], workspace_cache[23],
            workspace_cache[24]
        )

        return value

    @classmethod
    def from_sim(cls, sim, bank):
        # theta = polar angle wrt overhead
        #       = pi/2 - latitude (which is 0 on the horizon)
        return cls(sim.mass1, sim.mass2, sim.spin1x, sim.spin1y, sim.spin1z,
                   sim.spin2x, sim.spin2y, sim.spin2z, np.pi/2 - sim.latitude,
                   sim.longitude, sim.inclination, sim.polarization,
                   sim.coa_phase, bank, num_comps=sim.num_comps)

    @classmethod
    def from_sngl(cls, sngl, bank):
        # FIXME: Using alpha columns to hold theta, phi, iota, psi
        return cls(sngl.mass1, sngl.mass2, sngl.spin1x, sngl.spin1y,
                   sngl.spin1z, sngl.spin2x, sngl.spin2y, sngl.spin2z,
                   sngl.alpha1, sngl.alpha2, sngl.alpha3, sngl.alpha4,
                   sngl.alpha5, bank, num_comps=sngl.alpha6)

    @classmethod
    def from_dict(cls, params, idx, bank):
        flow = float(params['f_lower'][idx])
        if not flow > 0:
            flow = None
        duration = float(params['template_duration'][idx])
        if not duration > 0:
            duration = None
        return cls(params['mass1'][idx], params['mass2'][idx],
                   params['spin1x'][idx], params['spin1y'][idx],
                   params['spin1z'][idx], params['spin2x'][idx],
                   params['spin2y'][idx], params['spin2z'][idx],
                   params['latitude'][idx], params['longitude'][idx],
                   params['inclination'][idx], params['polarization'][idx],
                   params['orbital_phase'][idx], bank,
                   flow=flow, duration=duration, num_comps=params['num_comps'][idx])

    def to_sngl(self):
        # All numerical values are initiated as 0 and all strings as ''
        row = SnglInspiralTable()
        row.mass1 = self.m1
        row.mass2 = self.m2
        row.mtotal = self.m1 + self.m2
        row.mchirp = self._mchirp
        row.eta = row.mass1 * row.mass2 / (row.mtotal * row.mtotal)
        row.tau0, row.tau3 = m1m2_to_tau0tau3(self.m1, self.m2, self.flow)
        row.template_duration = self.dur
        row.spin1x = self.spin1x
        row.spin1y = self.spin1y
        row.spin1z = self.spin1z
        row.spin2x = self.spin2x
        row.spin2y = self.spin2y
        row.spin2z = self.spin2z
        row.alpha1 = self.theta
        row.alpha2 = self.phi
        row.alpha3 = self.iota
        row.alpha4 = self.psi
        row.alpha5 = self.orb_phase
        row.alpha6 = self.num_comps
        row.sigmasq = self.sigmasq
        if self.bank.flow_column:
            setattr(row, self.bank.flow_column, self.flow)
        return row

    def to_storage_arr(self):
        """Dump the template params to a numpy array."""
        new_tmplt = super(PrecessingSpinTemplate, self).to_storage_arr()
        new_tmplt['spin1x'] = self.spin1x
        new_tmplt['spin1y'] = self.spin1y
        new_tmplt['spin2x'] = self.spin2x
        new_tmplt['spin2y'] = self.spin2y
        new_tmplt['latitude'] = self.theta
        new_tmplt['longitude'] = self.phi
        new_tmplt['polarization'] = self.psi
        new_tmplt['inclination'] = self.iota
        new_tmplt['orbital_phase'] = self.orb_phase
        new_tmplt['num_comps'] = self.num_comps
        return new_tmplt


class HigherOrderModeTemplate(PrecessingSpinTemplate):
    """Class for higher order mode templates.

    Uses maximization over sky-location and amplitude, but *not* phase.
    """

    def brute_match(self, other, df, workspace_cache, **kwargs):

        # Template generates hp and hc
        hp, hc, hphccorr = self.get_whitened_normalized_comps(df, **kwargs)

        # Proposal generates h(t), sky loc is later discarded.
        proposal = other.get_whitened_normalized(df, **kwargs)

        # maximize over sky position of template
        return SBankComputeMatchSkyLoc(hp, hc,
                                       hphccorr, proposal,
                                       workspace_cache[0],
                                       workspace_cache[1],
                                       phase_maximized=False)


class EOBNRHigherOrderModeTemplate(IMRPrecessingSpinTemplate,
                                   HigherOrderModeTemplate):
    """Class for EOBNRHM templates."""
    approximant = "EOBNRv2HM_ROM"


class IMRPhenomXHMTemplate(IMRPrecessingSpinTemplate,
                                   HigherOrderModeTemplate):
    """Class for EOBNRHM templates."""
    approximant = "IMRPhenomXHM"


class EOBNRHigherOrderModeAmpMaxTemplate(IMRPrecessingSpinTemplate):
    """Class for EOBNRHM templates."""
    approximant = "EOBNRv2HM_ROM"

    def brute_match(self, other, df, workspace_cache, **kwargs):

        tmplt = self.get_whitened_normalized(df, **kwargs)
        proposal = other.get_whitened_normalized(df, **kwargs)

        # maximize over amplitude of template only
        return SBankComputeMatch(tmplt, proposal,
                                 workspace_cache[0], phase_maximized=False)


class EOBNRHigherOrderModePhaseMaxTemplate(IMRPrecessingSpinTemplate):
    """Class for EOBNRHM templates."""
    approximant = "EOBNRv2HM_ROM"


waveforms = {
    "TaylorF2RedSpin": TaylorF2RedSpinTemplate,
    "TaylorF2": TaylorF2Template,
    "IMRPhenomB": IMRPhenomBTemplate,
    "IMRPhenomC": IMRPhenomCTemplate,
    "IMRPhenomD": IMRPhenomDTemplate,
    "IMRPhenomP": IMRPhenomPTemplate,
    "IMRPhenomPv2": IMRPhenomPv2Template,
    "IMRPhenomPv2_THA": IMRPhenomPv2THATemplate,
    "IMRPhenomPv3": IMRPhenomPv3Template,
    "SEOBNRv2": SEOBNRv2Template,
    "SEOBNRv2_ROM_DoubleSpin": SEOBNRv2ROMDoubleSpinTemplate,
    "SEOBNRv2_ROM_DoubleSpin_HI": SEOBNRv2ROMDoubleSpinHITemplate,
    "SEOBNRv4": SEOBNRv4ROMTemplate,
    "SEOBNRv4_ROM": SEOBNRv4ROMTemplate,
    "EOBNRv2": EOBNRv2Template,
    "SpinTaylorT4": SpinTaylorT4Template,
    "SpinTaylorT5": SpinTaylorT5Template,
    "SpinTaylorF2": SpinTaylorF2Template,
    "SpinTaylorT5Fourier": SpinTaylorT5FourierTemplate,
    "SEOBNRv3": SEOBNRv3Template,
    "EOBNRv2HM_ROM": EOBNRHigherOrderModeTemplate,
    "EOBNRv2HM_ROM_AmpMax": EOBNRHigherOrderModeAmpMaxTemplate,
    "EOBNRv2HM_ROM_PhaseMax": EOBNRHigherOrderModePhaseMaxTemplate,
    "IMRPhenomXHM": IMRPhenomXHMTemplate,
}
