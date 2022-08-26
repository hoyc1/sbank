# Copyright (C) 2012  Nickolas Fotopoulos
# Copyright (C) 2014-2017  Stephen Privitera
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

from __future__ import division

import bisect
from operator import attrgetter

from six.moves import range

import numpy as np

from lal.iterutils import inorder, uniq
from lal import CreateREAL8Vector
from .overlap import SBankWorkspaceCache
from .psds import get_neighborhood_ASD, get_PSD, get_neighborhood_df_fmax
from . import waveforms


class lazy_nhoods(object):
    __slots__ = ("seq", "nhood_param")

    def __init__(self, seq, nhood_param="tau0"):
        self.seq = seq
        self.nhood_param = nhood_param

    def __getitem__(self, idx):
        return getattr(self.seq[idx], self.nhood_param)

    def __len__(self):
        return len(self.seq)


class Bank(object):

    def __init__(self, noise_model, flow, use_metric=False, cache_waveforms=False, nhood_size=1.0,
                 nhood_param="tau0", coarse_match_df=None, iterative_match_df_max=None,
                 fhigh_max=None, optimize_flow=None, flow_column=None):
        self.noise_model = noise_model
        self.flow = flow
        self.use_metric = use_metric
        self.cache_waveforms = cache_waveforms
        self.coarse_match_df = coarse_match_df
        self.iterative_match_df_max = iterative_match_df_max
        self.optimize_flow = optimize_flow
        self.flow_column = flow_column

        if (
            self.coarse_match_df
            and self.iterative_match_df_max
            and self.coarse_match_df < self.iterative_match_df_max
        ):
            # If this case occurs coarse_match_df offers no improvement, turn off
            self.coarse_match_df = None

        if fhigh_max is not None:
            self.fhigh_max = (2**(np.ceil(np.log2(fhigh_max))))
        else:
            self.fhigh_max = fhigh_max

        self.nhood_size = nhood_size
        self.nhood_param = nhood_param

        self._templates = []
        self._nmatch = 0
        self._nhoods = lazy_nhoods(self._templates, self.nhood_param)
        if use_metric:
            self._moments = {}
            self.compute_match = self._metric_match
        else:
            # The max over skyloc stuff needs a second cache
            self._workspace_cache = [SBankWorkspaceCache(),
                                     SBankWorkspaceCache()]
            self.compute_match = self._brute_match

    def __len__(self):
        return len(self._templates)

    def __iter__(self):
        return iter(self._templates)

    def __repr__(self):
        return repr(self._templates)

    def insort(self, new):
        ind = bisect.bisect_left(self._nhoods, getattr(new, self.nhood_param))
        self._templates.insert(ind, new)
        new.finalize_as_template()

    @classmethod
    def merge(cls, *banks):
        if not banks:
            raise ValueError("cannot merge zero banks")
        cls_args = list(uniq((b.tmplt_class, b.noise_model, b.flow,
                              b.use_metric) for b in banks))
        if len(cls_args) > 1:
            return ValueError("bank parameters do not match")
        merged = cls(*cls_args)
        merged._templates[:] = list(
            inorder(banks, key=attrgetter(merged.nhood_param))
        )
        merged._nmatch = sum(b._nmatch for b in banks)
        return merged

    def add_from_sngls(self, sngls, tmplt_class):
        newtmplts = [tmplt_class.from_sngl(s, bank=self) for s in sngls]
        for template in newtmplts:
            # Mark all templates as seed points
            template.is_seed_point = True
        self._templates.extend(newtmplts)
        self._templates.sort(key=attrgetter(self.nhood_param))

    def add_from_hdf(self, hdf_fp):
        num_points = len(hdf_fp['mass1'])
        newtmplts = []
        for idx in range(num_points):
            if not idx % 100000:
                tmp = {}
                end_idx = min(idx+100000, num_points)
                for name in hdf_fp:
                    tmp[name] = hdf_fp[name][idx:end_idx]
            c_idx = idx % 100000
            approx = tmp['approximant'][c_idx].decode('utf-8')
            tmplt_class = waveforms.waveforms[approx]
            newtmplts.append(tmplt_class.from_dict(tmp, c_idx, self))
            newtmplts[-1].is_seed_point = True
        self._templates.extend(newtmplts)
        self._templates.sort(key=attrgetter(self.nhood_param))

    @classmethod
    def from_sngls(cls, sngls, tmplt_class, *args, **kwargs):
        bank = cls(*args, **kwargs)
        new_tmplts = [tmplt_class.from_sngl(s, bank=bank) for s in sngls]
        bank._templates.extend(new_tmplts)
        bank._templates.sort(key=attrgetter(bank.nhood_param))
        # Mark all templates as seed points
        for template in bank._templates:
            template.is_seed_point = True
        return bank

    @classmethod
    def from_sims(cls, sims, tmplt_class, *args):
        bank = cls(*args)
        new_sims = [tmplt_class.from_sim(s, bank=bank) for s in sims]
        bank._templates.extend(new_sims)
        return bank

    def _metric_match(self, tmplt, proposal, f, **kwargs):
        return tmplt.metric_match(proposal, f, **kwargs)

    def _brute_match(self, tmplt, proposal, f, **kwargs):
        match = tmplt.brute_match(proposal, f, self._workspace_cache, **kwargs)
        if not self.cache_waveforms:
            tmplt.clear()
        return match

    def tmplt_coarse(self, tmplt, proposal, f_max, **kwargs):
        match = None
        if self.coarse_match_df:
            PSD = get_PSD(self.coarse_match_df, self.flow, f_max,
                          self.noise_model)
            match = self.compute_match(tmplt, proposal,
                                       self.coarse_match_df, PSD=PSD, **kwargs)
        return match

    def tmplt_df_iter(self, tmplt, proposal, df, df_end, f_max,
                      min_match=None, **kwargs):
        match_last = 0

        while df >= df_end:

            PSD = get_PSD(df, self.flow, f_max, self.noise_model)
            match = self.compute_match(tmplt, proposal, df, PSD=PSD, **kwargs)
            if match == 0:
                match_last = -1
                df /= 2.0
                continue

            # if the result is a really bad match, trust it isn't
            # misrepresenting a good match
            if min_match and (match < min_match):
                break

            # calculation converged
            if match_last > 0 and abs(match_last - match) < 0.001:
                break

            # otherwise, refine calculation
            match_last = match
            df /= 2.0

        return match

    def covers(self, proposal, min_match, nhood=None, **kwargs):
        """
        Return (max_match, template) where max_match is either (i) the
        best found match if max_match < min_match or (ii) the match of
        the first template found with match >= min_match.  template is
        the Template() object which yields max_match.
        """
        max_match = 0
        template = None

        # find templates in the bank "near" this tmplt
        prop_nhd = getattr(proposal, self.nhood_param)
        if not nhood:
            low, high = _find_neighborhood(self._nhoods, prop_nhd,
                                           self.nhood_size)
            tmpbank = self._templates[low:high]
        else:
            tmpbank = nhood
        if not tmpbank:
            #print ("None in neighborhood")
            return (max_match, template)

        # sort the bank by its nearness to tmplt in mchirp
        # NB: This sort comes up as a dominating cost if you profile,
        # but it cuts the number of match evaluations by 80%, so turns out
        # to be worth it even for metric match, where matches are cheap.
        tmpbank.sort(key=lambda b: abs(getattr(b, self.nhood_param) - prop_nhd))

        # set parameters of match calculation that are optimized for this block
        df_end, f_max = get_neighborhood_df_fmax(tmpbank + [proposal],
                                                 self.flow)
        if self.fhigh_max:
            f_max = min(f_max, self.fhigh_max)
        if self.iterative_match_df_max is not None:
            df_start = max(df_end, self.iterative_match_df_max)
        else:
            df_start = df_end

        # Stop if the match is too low
        rough_match = min_match - 0.05

        # find and test matches
        for tmplt in tmpbank:

            self._nmatch += 1

            if self.coarse_match_df:
                # Perform a match at high df to see if point can be quickly
                # ruled out as already covering the proposal
                match = self.tmplt_coarse(tmplt, proposal, f_max, **kwargs)

                if (match > 0) and (match < rough_match):
                    continue

            match = self.tmplt_df_iter(tmplt, proposal, df_start, df_end, f_max,
                                       min_match=rough_match, **kwargs)
            if match > min_match:
                #print ("Rejecting at", match)
                return (match, tmplt)

            # record match and template params for highest match
            if match > max_match:
                max_match = match
                template = tmplt

        return (max_match, template)

    def max_match(self, proposal):
        match, best_tmplt_ind = self.argmax_match(proposal)
        if not match:
            return (0., 0)
        return match, self._templates[best_tmplt_ind]

    def argmax_match(self, proposal):
        # find templates in the bank "near" this tmplt
        prop_nhd = getattr(proposal, self.nhood_param)
        low, high = _find_neighborhood(self._nhoods, prop_nhd, self.nhood_size)
        tmpbank = self._templates[low:high]
        if not tmpbank:
            return (0., 0)

        # set parameters of match calculation that are optimized for this block
        df, ASD = get_neighborhood_ASD(tmpbank + [proposal], self.flow,
                                       self.noise_model)

        # compute matches
        matches = [self.compute_match(tmplt, proposal, df, ASD=ASD)
                   for tmplt in tmpbank]
        best_tmplt_ind = np.argmax(matches)
        self._nmatch += len(tmpbank)

        # best_tmplt_ind indexes tmpbank; add low to get index of full bank
        return matches[best_tmplt_ind], best_tmplt_ind + low

    def clear(self):
        if hasattr(self, "_workspace_cache"):
            self._workspace_cache[0] = SBankWorkspaceCache()
            self._workspace_cache[1] = SBankWorkspaceCache()

        for tmplt in self._templates:
            tmplt.clear()


class BankTHA(Bank):

    def get_factors(self, size):
        ra = np.random.uniform(0., 2. * np.pi, size=size)
        dec = np.arccos(np.random.uniform(-1., 1., size=size)) - np.pi / 2.
        psi = np.random.uniform(0., 2. * np.pi, size=size)

        phi = np.random.uniform(0., 2. * np.pi, size=size)
        alpha = np.random.uniform(0., 2. * np.pi, size=size)
        theta = np.arccos(np.random.uniform(-1., 1., size=size))

        fp = 0.5 * (1 + np.cos(ra) ** 2) * np.cos(2 * dec) * np.cos(2 * psi)
        fp -= np.cos(ra) * np.sin(2 * dec) * np.sin(2 * psi)
        fc = 0.5 * (1 + np.cos(ra) ** 2) * np.cos(2 * dec) * np.sin(2 * psi)
        fc += np.cos(ra) * np.sin(2 * dec) * np.cos(2 * psi)
        
        aplus = [(1. + np.cos(theta) ** 2.) / 2.]
        across = [np.cos(theta)]

        aplus += [2. * np.sin(theta) * np.cos(theta)]
        across += [2 * np.sin(theta)]

        aplus += [3. * np.sin(theta) ** 2.]
        across += [np.zeros(aplus[-1].shape, dtype=aplus[-1].dtype)]

        aplus += [- aplus[1]]
        across += [across[1]]

        aplus += [aplus[0]]
        across += [- across[0]]

        aplus = np.stack(aplus, axis=1)
        across = np.stack(across, axis=1)

        phik = np.stack([2. * phi + (2. - k) * alpha for k in range(5)], axis=1)
        psi2 = 2. * psi[:, None]

        a1 = aplus * np.cos(phik) * np.cos(psi2) - across * np.sin(phik) * np.sin(psi2)
        a2 = aplus * np.cos(phik) * np.sin(psi2) + across * np.sin(phik) * np.cos(psi2)
        a3 = - aplus * np.sin(phik) * np.cos(psi2) - across * np.cos(phik) * np.sin(psi2)
        a4 = - aplus * np.sin(phik) * np.sin(psi2) + across * np.cos(phik) * np.cos(psi2)

        wp = fp * np.cos(2. * psi) - fc * np.sin(2. * psi)
        wc = fp * np.sin(2. * psi) + fc * np.cos(2. * psi)

        wp = wp[:, None]
        wc = wc[:, None]

        real = wp * a1 + wc * a2
        imag = wp * a3 + wc * a4

        return (real ** 2. + imag ** 2.) ** 0.5

    def _skyaverage_match(self, tmplt, proposal, f, **kwargs):
        b = np.tan(tmplt.beta / 2.)
        bk = b ** np.arange(5)
        bk = bk / ((1 + b ** 2.) ** 2.)
        factors = self.factors * bk[None, :]

        for i in range(5):
            self.tmp_factors[i].data[:] = factors[:, i]

        match = tmplt.brute_comp_match(
            proposal, self.tmp_factors, f, self._workspace_cache, **kwargs
        )
        if not self.cache_waveforms:
            tmplt.clear()

        return match

    def __init__(self, noise_model, flow, cache_waveforms=False, nhood_size=1.0,
                 nhood_param="tau0", coarse_match_df=None, iterative_match_df_max=None,
                 fhigh_max=None, optimize_flow=None, flow_column=None, max_num_comps=5,
                 sky_draws=10000):

        super(BankTHA, self).__init__(
            noise_model, flow,
            cache_waveforms=cache_waveforms,
            nhood_size=nhood_size, nhood_param=nhood_param,
            coarse_match_df=coarse_match_df,
            iterative_match_df_max=iterative_match_df_max,
            fhigh_max=fhigh_max,
            optimize_flow=optimize_flow, flow_column=flow_column,
        )

        # We need 25 caches for harmonic banks
        self._workspace_cache = [SBankWorkspaceCache() for i in range(25)]
        self.max_num_comps = max_num_comps

        self.sky_draws = sky_draws
        self.factors = self.get_factors(sky_draws)
        self.tmp_factors = [CreateREAL8Vector(sky_draws) for i in range(5)]
        self.compute_match = self._skyaverage_match

    def insort_idx(self, new):
        ind = bisect.bisect_left(self._nhoods, getattr(new, self.nhood_param))
        self._templates.insert(ind, new)
        new.finalize_as_template()
        return ind

    def add_from_hdf(self, hdf_fp):
        num_points = len(hdf_fp['mass1'])
        newtmplts = []
        for idx in range(num_points):
            if not idx % 100000:
                tmp = {}
                end_idx = min(idx+100000, num_points)
                for name in hdf_fp:
                    tmp[name] = hdf_fp[name][idx:end_idx]
            c_idx = idx % 100000
            approx = tmp['approximant'][c_idx].decode('utf-8') + '_THA'
            tmplt_class = waveforms.waveforms[approx]
            newtmplts.append(tmplt_class.from_dict(tmp, c_idx, self))
            newtmplts[-1].is_seed_point = True
        self._templates.extend(newtmplts)
        self._templates.sort(key=attrgetter(self.nhood_param))

    def covers_harmonics(self, proposal, min_match, nhood=None, num_comps=5,
                         less_only=False, include_proposal=False, **kwargs):
        """
        Return (max_match, template) where max_match is either (i) the
        best found match if max_match < min_match or (ii) the match of
        the first template found with match >= min_match.  template is
        the Template() object which yields max_match.
        """
        max_match = 0
        template = None

        # find templates in the bank "near" this tmplt
        prop_nhd = getattr(proposal, self.nhood_param)
        if not nhood:
            low, high = _find_neighborhood(self._nhoods, prop_nhd,
                                           self.nhood_size)
            tmpbank = self._templates[low:high]
        else:
            tmpbank = nhood

        if include_proposal:
            tmpbank += [proposal]

        if not tmpbank:
            #print ("None in neighborhood")
            return max_match, template

        # sort the bank by its nearness to tmplt in mchirp
        # NB: This sort comes up as a dominating cost if you profile,
        # but it cuts the number of match evaluations by 80%, so turns out
        # to be worth it even for metric match, where matches are cheap.
        tmpbank.sort(key=lambda b: abs(getattr(b, self.nhood_param) - prop_nhd))

        # set parameters of match calculation that are optimized for this block
        df_end, f_max = get_neighborhood_df_fmax(tmpbank + [proposal],
                                                 self.flow)
        if self.fhigh_max:
            f_max = min(f_max, self.fhigh_max)
        if self.iterative_match_df_max is not None:
            df_start = max(df_end, self.iterative_match_df_max)
        else:
            df_start = df_end

        # Stop if the match is too low
        rough_match = min_match - 0.05

        # find and test matches
        for tmplt in tmpbank:

            if less_only and tmplt.num_comps < num_comps:
                continue

            self._nmatch += 1

            if self.coarse_match_df:
                # Perform a match at high df to see if point can be quickly
                # ruled out as already covering the proposal
                match = self.tmplt_coarse(tmplt, proposal, f_max,
                                          num_comps=num_comps, **kwargs)

                if (match > 0) and (match < rough_match):
                    continue

            match = self.tmplt_df_iter(tmplt, proposal, df_start, df_end, f_max,
                                       min_match=rough_match, num_comps=num_comps, **kwargs)

            if match > min_match:
                return match, tmplt

            # record match and template params for highest match
            if match > max_match:
                max_match = match
                template = tmplt

        return max_match, template

    def clear(self):
        if hasattr(self, "_workspace_cache"):
            self._workspace_cache[0] = SBankWorkspaceCache()
            self._workspace_cache[1] = SBankWorkspaceCache()
            self._workspace_cache[2] = SBankWorkspaceCache()
            self._workspace_cache[3] = SBankWorkspaceCache()
            self._workspace_cache[4] = SBankWorkspaceCache()

        for tmplt in self._templates:
            tmplt.clear()


def _find_neighborhood(tmplt_locs, prop_loc, nhood_size=0.25):
    """
    Return the min and max indices of templates that cover the given
    template at prop_loc within a parameter difference of nhood_size (seconds).
    tmplt_locs should be a sequence of neighborhood values in sorted order.
    """
    low_ind = bisect.bisect_left(tmplt_locs, prop_loc - nhood_size)
    high_ind = bisect.bisect_right(tmplt_locs, prop_loc + nhood_size)
    return low_ind, high_ind
