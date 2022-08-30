# Copyright (C) 2021 Ian Harry

import cython
cimport numpy

cdef extern from "overlap_cpu_lib.c":
    ctypedef struct WS
    WS *SBankCreateWorkspaceCache()
    void SBankDestroyWorkspaceCache(WS *workspace_cache)
    double _SBankComputeMatch(float complex *inj, float complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache)
    double _SBankComputeRealMatch(float complex *inj, float complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache)
    double _SBankComputeMatchMaxSkyLoc(float complex *hp, float complex *hc, const double hphccorr, float complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2)
    double _SBankComputeMatchMaxSkyLocNoPhase(float complex *hp, float complex *hc, const double hphccorr, float complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2)
    double _SBankComputeFiveCompMatch(float complex *temp_comp1, float complex *temp_comp2, float complex *temp_comp3, float complex *temp_comp4, float complex *temp_comp5, float complex *proposal, size_t min_len, double delta_f, int num_comps, WS *workspace_cache1, WS *workspace_cache2, WS *workspace_cache3, WS *workspace_cache4, WS *workspace_cache5)
    double _SBankComputeFiveCompFactorMatch(float complex *temp_comp1, float complex *temp_comp2, float complex *temp_comp3, float complex *temp_comp4, float complex *temp_comp5, float complex *prop_comp1, float complex *prop_comp2, float complex *prop_comp3, float complex *prop_comp4, float complex *prop_comp5, double *sigmasq1, double *sigmasq2, double *sigmasq3, double *sigmasq4, double *sigmasq5, double *sigmasq, double *matchsq, size_t min_len, size_t f_len, double delta_f, int num_comps, WS *workspace_cache1, WS *workspace_cache2, WS *workspace_cache3, WS *workspace_cache4, WS *workspace_cache5, WS *workspace_cache6, WS *workspace_cache7, WS *workspace_cache8, WS *workspace_cache9, WS *workspace_cache10, WS *workspace_cache11, WS *workspace_cache12, WS *workspace_cache13, WS *workspace_cache14, WS *workspace_cache15, WS *workspace_cache16, WS *workspace_cache17, WS *workspace_cache18, WS *workspace_cache19, WS *workspace_cache20, WS *workspace_cache21, WS *workspace_cache22, WS *workspace_cache23, WS *workspace_cache24, WS *workspace_cache25)

# WARNING: Handling C pointers in python gets nasty. The workspace item is
#          important for sbank's memory management and optimality. It makes
#          sense that we store this *in* python, but it needs to contain
#          actual C memory pointers. I'm not 100% sure I've understood how I've
#          solved this, but I think the main point is to have the structure
#          imported here, and then any function that needs it must use cdef.
#          It's okay to *get* the thing using the cdeffed functions below, but
#          when interacting with the C functions it must be clear what we have
#          and then the cdef interface functions are needed as well.
# https://www.mail-archive.com/cython-dev@codespeak.net/msg06363.html
cdef class SBankWorkspaceCache:
    cdef WS *workspace

    def __cinit__(self):
        self.workspace = self.__create()
        
    def __dealloc__(self):
        if self.workspace != NULL:
            SBankDestroyWorkspaceCache(self.workspace)

    cdef WS* __create(self):
        cdef WS *temp
        temp = SBankCreateWorkspaceCache()
        return temp

    cdef WS* get_workspace(self):
        return self.workspace

# As a reference for sending numpy arrays onto C++
# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html
# https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
# https://stackoverflow.com/questions/21242160/how-to-build-a-cython-wrapper-for-c-function-with-stl-list-parameter

def SBankCythonComputeMatch(
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] inj,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] tmplt,
    int min_len,
    double delta_f,
    workspace_cache
):
    cdef WS* _workspace
    # So even though workspace_cache is an instance of SBankWorkspaceCache, it
    # seems we still need to call like this. workspace_cache.get_workspace()
    # does not compile as cython is not clear it *is* a SBankWorkspaceCache
    # object.
    _workspace = SBankWorkspaceCache.get_workspace(workspace_cache)
    return _SBankComputeMatch(&inj[0], &tmplt[0], min_len, delta_f,
                              _workspace)

def SBankCythonComputeRealMatch(
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] inj,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] tmplt,
    int min_len,
    double delta_f,
    workspace_cache
):
    cdef WS* _workspace
    _workspace = SBankWorkspaceCache.get_workspace(workspace_cache)
    return _SBankComputeRealMatch(&inj[0], &tmplt[0], min_len, delta_f,
                                  _workspace)

def SBankCythonComputeMatchMaxSkyLoc(
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] hp,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] hc,
    double hphccorr,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] proposal,
    int min_len,
    double delta_f,
    workspace_cache1,
    workspace_cache2
):
    cdef WS* _workspace1
    cdef WS* _workspace2
    _workspace1 = SBankWorkspaceCache.get_workspace(workspace_cache1)
    _workspace2 = SBankWorkspaceCache.get_workspace(workspace_cache2)
    return _SBankComputeMatchMaxSkyLoc(&hp[0], &hc[0], hphccorr, &proposal[0],
                                       min_len, delta_f, _workspace1,
                                       _workspace2)

def SBankCythonComputeMatchMaxSkyLocNoPhase(
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] hp,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] hc,
    double hphccorr,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] proposal,
    int min_len,
    double delta_f,
    workspace_cache1,
    workspace_cache2
):
    cdef WS* _workspace1
    cdef WS* _workspace2
    _workspace1 = SBankWorkspaceCache.get_workspace(workspace_cache1)
    _workspace2 = SBankWorkspaceCache.get_workspace(workspace_cache2)
    return _SBankComputeMatchMaxSkyLocNoPhase(&hp[0], &hc[0], hphccorr,
                                              &proposal[0], min_len, delta_f,
                                              _workspace1, _workspace2)

def SBankCythonComputeFiveCompMatch(
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp1,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp2,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp3,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp4,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp5,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] proposal,
    int min_len,
    double delta_f,
    int num_comps,
    workspace_cache1,
    workspace_cache2,
    workspace_cache3,
    workspace_cache4,
    workspace_cache5
):
    cdef WS* _workspace1
    cdef WS* _workspace2
    cdef WS* _workspace3
    cdef WS* _workspace4
    cdef WS* _workspace5
    _workspace1 = SBankWorkspaceCache.get_workspace(workspace_cache1)
    _workspace2 = SBankWorkspaceCache.get_workspace(workspace_cache2)
    _workspace3 = SBankWorkspaceCache.get_workspace(workspace_cache3)
    _workspace4 = SBankWorkspaceCache.get_workspace(workspace_cache4)
    _workspace5 = SBankWorkspaceCache.get_workspace(workspace_cache5)

    return _SBankComputeFiveCompMatch(&temp_comp1[0], &temp_comp2[0], &temp_comp3[0], &temp_comp4[0], &temp_comp5[0], &proposal[0], min_len, delta_f, num_comps, _workspace1, _workspace2, _workspace3, _workspace4, _workspace5)

def SBankCythonComputeFiveCompFactorMatch(
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp1,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp2,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp3,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp4,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] temp_comp5,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] prop_comp1,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] prop_comp2,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] prop_comp3,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] prop_comp4,
    numpy.ndarray[numpy.complex64_t, ndim=1, mode="c"] prop_comp5,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] sigmasq1,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] sigmasq2,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] sigmasq3,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] sigmasq4,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] sigmasq5,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] sigmasq,
    numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] matchsq,
    int min_len,
    int f_len,
    double delta_f,
    int num_comps,
    workspace_cache1,
    workspace_cache2,
    workspace_cache3,
    workspace_cache4,
    workspace_cache5,
    workspace_cache6,
    workspace_cache7,
    workspace_cache8,
    workspace_cache9,
    workspace_cache10,
    workspace_cache11,
    workspace_cache12,
    workspace_cache13,
    workspace_cache14,
    workspace_cache15,
    workspace_cache16,
    workspace_cache17,
    workspace_cache18,
    workspace_cache19,
    workspace_cache20,
    workspace_cache21,
    workspace_cache22,
    workspace_cache23,
    workspace_cache24,
    workspace_cache25,
):
    cdef WS* _workspace1
    cdef WS* _workspace2
    cdef WS* _workspace3
    cdef WS* _workspace4
    cdef WS* _workspace5
    cdef WS* _workspace6
    cdef WS* _workspace7
    cdef WS* _workspace8
    cdef WS* _workspace9
    cdef WS* _workspace10
    cdef WS* _workspace11
    cdef WS* _workspace12
    cdef WS* _workspace13
    cdef WS* _workspace14
    cdef WS* _workspace15
    cdef WS* _workspace16
    cdef WS* _workspace17
    cdef WS* _workspace18
    cdef WS* _workspace19
    cdef WS* _workspace20
    cdef WS* _workspace21
    cdef WS* _workspace22
    cdef WS* _workspace23
    cdef WS* _workspace24
    cdef WS* _workspace25
    _workspace1 = SBankWorkspaceCache.get_workspace(workspace_cache1)
    _workspace2 = SBankWorkspaceCache.get_workspace(workspace_cache2)
    _workspace3 = SBankWorkspaceCache.get_workspace(workspace_cache3)
    _workspace4 = SBankWorkspaceCache.get_workspace(workspace_cache4)
    _workspace5 = SBankWorkspaceCache.get_workspace(workspace_cache5)
    _workspace6 = SBankWorkspaceCache.get_workspace(workspace_cache6)
    _workspace7 = SBankWorkspaceCache.get_workspace(workspace_cache7)
    _workspace8 = SBankWorkspaceCache.get_workspace(workspace_cache8)
    _workspace9 = SBankWorkspaceCache.get_workspace(workspace_cache9)
    _workspace10 = SBankWorkspaceCache.get_workspace(workspace_cache10)
    _workspace11 = SBankWorkspaceCache.get_workspace(workspace_cache11)
    _workspace12 = SBankWorkspaceCache.get_workspace(workspace_cache12)
    _workspace13 = SBankWorkspaceCache.get_workspace(workspace_cache13)
    _workspace14 = SBankWorkspaceCache.get_workspace(workspace_cache14)
    _workspace15 = SBankWorkspaceCache.get_workspace(workspace_cache15)
    _workspace16 = SBankWorkspaceCache.get_workspace(workspace_cache16)
    _workspace17 = SBankWorkspaceCache.get_workspace(workspace_cache17)
    _workspace18 = SBankWorkspaceCache.get_workspace(workspace_cache18)
    _workspace19 = SBankWorkspaceCache.get_workspace(workspace_cache19)
    _workspace20 = SBankWorkspaceCache.get_workspace(workspace_cache20)
    _workspace21 = SBankWorkspaceCache.get_workspace(workspace_cache21)
    _workspace22 = SBankWorkspaceCache.get_workspace(workspace_cache22)
    _workspace23 = SBankWorkspaceCache.get_workspace(workspace_cache23)
    _workspace24 = SBankWorkspaceCache.get_workspace(workspace_cache24)
    _workspace25 = SBankWorkspaceCache.get_workspace(workspace_cache25)

    return _SBankComputeFiveCompFactorMatch(&temp_comp1[0], &temp_comp2[0], &temp_comp3[0], &temp_comp4[0], &temp_comp5[0], &prop_comp1[0], &prop_comp2[0], &prop_comp3[0], &prop_comp4[0], &prop_comp5[0], &sigmasq1[0], &sigmasq2[0], &sigmasq3[0], &sigmasq4[0], &sigmasq5[0], &sigmasq[0], &matchsq[0], min_len, f_len, delta_f, num_comps, _workspace1, _workspace2, _workspace3, _workspace4, _workspace5, _workspace6, _workspace7, _workspace8, _workspace9, _workspace10, _workspace11, _workspace12, _workspace13, _workspace14, _workspace15, _workspace16, _workspace17, _workspace18, _workspace19, _workspace20, _workspace21, _workspace22, _workspace23, _workspace24, _workspace25)
