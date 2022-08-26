# Copyright (C) 2021 Ian Harry

# There is an overlap_cuda.py module, but that's never been hooked up
# So we just hook up the CPU module in all cases. We could add switches here
# to use the GPU if needed (although it will likely be more complicated than
# that if the waveforms were not generated on the GPU)

from .overlap_cpu import SBankCythonComputeMatch
from .overlap_cpu import SBankCythonComputeRealMatch
from .overlap_cpu import SBankCythonComputeMatchMaxSkyLoc
from .overlap_cpu import SBankCythonComputeMatchMaxSkyLocNoPhase
from .overlap_cpu import SBankCythonComputeFiveCompMatch
from .overlap_cpu import SBankCythonComputeFiveCompFactorMatch
from .overlap_cpu import SBankWorkspaceCache as CPUWorkspaceCache

# If considering enabling the GPU code, need to switch this as well.
# Currently the GPU WorkspaceCache will not work and would need some fixing.
SBankWorkspaceCache = CPUWorkspaceCache


def SBankComputeMatch(inj, tmplt, workspace_cache, phase_maximized=True):
    """
    ADD ME
    """
    min_len = tmplt.data.length
    if inj.data.length <= tmplt.data.length:
        min_len = inj.data.length
    else:
        min_len = tmplt.data.length
    assert(inj.deltaF == tmplt.deltaF)
    delta_f = inj.deltaF
    if phase_maximized:
        return SBankCythonComputeMatch(inj.data.data, tmplt.data.data, min_len,
                                       delta_f, workspace_cache)
    else:
        return SBankCythonComputeRealMatch(inj.data.data, tmplt.data.data,
                                           min_len, delta_f, workspace_cache)


def SBankComputeMatchSkyLoc(hp, hc, hphccorr, proposal, workspace_cache1,
                            workspace_cache2, phase_maximized=False):
    """
    ADD ME
    """
    assert(hp.deltaF == proposal.deltaF)
    assert(hc.deltaF == proposal.deltaF)
    assert(hp.data.length == hc.data.length)
    if proposal.data.length <= hp.data.length:
        min_len = proposal.data.length
    else:
        min_len = hp.data.length
    if phase_maximized:
        return SBankCythonComputeMatchMaxSkyLoc(hp.data.data, hc.data.data,
                                                hphccorr, proposal.data.data,
                                                min_len, hp.deltaF,
                                                workspace_cache1,
                                                workspace_cache2)
    else:
        return SBankCythonComputeMatchMaxSkyLocNoPhase(
            hp.data.data,
            hc.data.data,
            hphccorr,
            proposal.data.data,
            min_len,
            hp.deltaF,
            workspace_cache1,
            workspace_cache2
        )

def SBankComputeFiveCompMatch(h1, h2, h3, h4, h5, proposal, num_comps, workspace_cache1,
                              workspace_cache2, workspace_cache3,
                              workspace_cache4, workspace_cache5):
    """
    ADD ME
    """
    assert(1 <= num_comps <= 5)
    assert(h1.deltaF == proposal.deltaF)
    assert(h2.deltaF == proposal.deltaF)
    assert(h3.deltaF == proposal.deltaF)
    assert(h4.deltaF == proposal.deltaF)
    assert(h5.deltaF == proposal.deltaF)
    assert(h2.data.length == h1.data.length)
    assert(h3.data.length == h1.data.length)
    assert(h4.data.length == h1.data.length)
    assert(h5.data.length == h1.data.length)
    if proposal.data.length <= h1.data.length:
        min_len = proposal.data.length
    else:
        min_len = h1.data.length

    return SBankCythonComputeFiveCompMatch(h1.data.data, h2.data.data,
                                           h3.data.data, h4.data.data,
                                           h5.data.data, proposal.data.data,
                                           min_len, h1.deltaF, num_comps,
                                           workspace_cache1,
                                           workspace_cache2,
                                           workspace_cache3,
                                           workspace_cache4,
                                           workspace_cache5)

def SBankComputeFiveCompFactorMatch(h1, h2, h3, h4, h5,
                                    p1, p2, p3, p4, p5,
                                    f1, f2, f3, f4, f5,
                                    num_comps, workspace_cache1,
                                    workspace_cache2, workspace_cache3,
                                    workspace_cache4, workspace_cache5,
                                    workspace_cache6, workspace_cache7,
                                    workspace_cache8, workspace_cache9,
                                    workspace_cache10, workspace_cache11,
                                    workspace_cache12, workspace_cache13,
                                    workspace_cache14, workspace_cache15,
                                    workspace_cache16, workspace_cache17,
                                    workspace_cache18, workspace_cache19,
                                    workspace_cache20, workspace_cache21,
                                    workspace_cache22, workspace_cache23,
                                    workspace_cache24, workspace_cache25):
    """
    ADD ME
    """
    assert(1 <= num_comps <= 5)
    assert(h1.deltaF == p1.deltaF)
    assert(h2.deltaF == p1.deltaF)
    assert(h3.deltaF == p1.deltaF)
    assert(h4.deltaF == p1.deltaF)
    assert(h5.deltaF == p1.deltaF)
    assert(p2.deltaF == p1.deltaF)
    assert(p3.deltaF == p1.deltaF)
    assert(p4.deltaF == p1.deltaF)
    assert(p5.deltaF == p1.deltaF)
    assert(h2.data.length == h1.data.length)
    assert(h3.data.length == h1.data.length)
    assert(h4.data.length == h1.data.length)
    assert(h5.data.length == h1.data.length)
    assert(p2.data.length == p1.data.length)
    assert(p3.data.length == p1.data.length)
    assert(p4.data.length == p1.data.length)
    assert(p5.data.length == p1.data.length)
    if p1.data.length <= h1.data.length:
        min_len = p1.data.length
    else:
        min_len = h1.data.length
    f_len = f1.length

    return SBankCythonComputeFiveCompFactorMatch(h1.data.data, h2.data.data,
                                                 h3.data.data, h4.data.data,
                                                 h5.data.data, p1.data.data,
                                                 p2.data.data, p3.data.data,
                                                 p4.data.data, p5.data.data,
                                                 f1.data, f2.data, f3.data,
                                                 f4.data, f5.data,
                                                 min_len, f_len,
                                                 h1.deltaF, num_comps,
                                                 workspace_cache1, workspace_cache2,
                                                 workspace_cache3, workspace_cache4,
                                                 workspace_cache5, workspace_cache6,
                                                 workspace_cache7, workspace_cache8,
                                                 workspace_cache9, workspace_cache10,
                                                 workspace_cache11, workspace_cache12,
                                                 workspace_cache13, workspace_cache14,
                                                 workspace_cache15, workspace_cache16,
                                                 workspace_cache17, workspace_cache18,
                                                 workspace_cache19, workspace_cache20,
                                                 workspace_cache21, workspace_cache22,
                                                 workspace_cache23, workspace_cache24,
                                                 workspace_cache25)
