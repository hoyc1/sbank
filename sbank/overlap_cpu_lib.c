/*
 * Copyright (C) 2012  Nickolas Fotopoulos
 * Copyright (C) 2016  Stephen Privitera
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <sys/types.h>

/* ---------------- LAL STUFF NEEDED ----------------- */
typedef struct tagCOMPLEX8Vector {
     uint32_t length; /**< Number of elements in array. */
     float complex *data; /**< Pointer to the data array. */
} COMPLEX8Vector;

COMPLEX8Vector * XLALCreateCOMPLEX8Vector ( uint32_t length );
void XLALDestroyCOMPLEX8Vector ( COMPLEX8Vector * vector );

typedef struct tagCOMPLEX8FFTPlan COMPLEX8FFTPlan;

COMPLEX8FFTPlan * XLALCreateReverseCOMPLEX8FFTPlan( uint32_t size, int measurelvl );

void XLALDestroyCOMPLEX8FFTPlan( COMPLEX8FFTPlan *plan );

/* ----------------------------------------------- */

typedef struct tagWS {
    size_t n;
    COMPLEX8FFTPlan *plan;
    COMPLEX8Vector *zf;
    COMPLEX8Vector *zt;
} WS;

WS *SBankCreateWorkspaceCache(void);

void SBankDestroyWorkspaceCache(WS *workspace_cache);

double _SBankComputeMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache);

double _SBankComputeRealMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache);

double _SBankComputeMatchMaxSkyLoc(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2);

double _SBankComputeMatchMaxSkyLocNoPhase(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2);

double _SBankComputeFiveCompMatch(complex *temp_comp1, complex *temp_comp2,
                                  complex *temp_comp3, complex *temp_comp4,
                                  complex *temp_comp5, complex *proposal,
                                  size_t min_len, double delta_f, int8_t num_comps,
                                  WS *workspace_cache1,
                                  WS *workspace_cache2, WS *workspace_cache3,
                                  WS *workspace_cache4, WS *workspace_cache5);

double _SBankComputeFiveCompFactorMatch(complex *temp_comp1, complex *temp_comp2,
					complex *temp_comp3, complex *temp_comp4,
					complex *temp_comp5, complex *prop_comp1,
					complex *prop_comp2, complex *prop_comp3,
					complex *prop_comp4, complex *prop_comp5,
					double *prop_factor1, double *prop_factor2,
					double *prop_factor3, double *prop_factor4,
					double *prop_factor5, size_t min_len,
					size_t f_len, double delta_f, int8_t num_comps,
					WS *workspace_cache1,
					WS *workspace_cache2, WS *workspace_cache3,
					WS *workspace_cache4, WS *workspace_cache5,
					WS *workspace_cache6, WS *workspace_cache7,
					WS *workspace_cache8, WS *workspace_cache9,
					WS *workspace_cache10, WS *workspace_cache11,
					WS *workspace_cache12, WS *workspace_cache13,
					WS *workspace_cache14, WS *workspace_cache15,
					WS *workspace_cache16, WS *workspace_cache17,
					WS *workspace_cache18, WS *workspace_cache19,
					WS *workspace_cache20, WS *workspace_cache21,
					WS *workspace_cache22, WS *workspace_cache23,
					WS *workspace_cache24, WS *workspace_cache25);

#define MAX_NUM_WS 32  /* maximum number of workspaces */
#define CHECK_OOM(ptr, msg) if (!(ptr)) { XLALPrintError((msg)); exit(-1); }

/*
 * set up workspaces
 */

WS *SBankCreateWorkspaceCache(void) {
    WS *workspace_cache = calloc(MAX_NUM_WS, sizeof(WS));
    CHECK_OOM(workspace_cache, "unable to allocate workspace\n");
    return workspace_cache;
}

void SBankDestroyWorkspaceCache(WS *workspace_cache) {
    size_t k = MAX_NUM_WS;
    for (;k--;) {
        if (workspace_cache[k].n) {
            XLALDestroyCOMPLEX8FFTPlan(workspace_cache[k].plan);
            XLALDestroyCOMPLEX8Vector(workspace_cache[k].zf);
            XLALDestroyCOMPLEX8Vector(workspace_cache[k].zt);
        }
    }
    free(workspace_cache);
}

static WS *get_workspace(WS *workspace_cache, const size_t n) {
    if (!n) {
        fprintf(stderr, "Zero size workspace requested\n");
        abort();
    }

    /* if n already in cache, return it */
    WS *ptr = workspace_cache;
    while (ptr->n) {
        if (ptr->n == n) return ptr;
        if (++ptr - workspace_cache > MAX_NUM_WS) return NULL;  /* out of space! */
    }

    /* if n not in cache, ptr now points at first blank entry */
    ptr->zf = XLALCreateCOMPLEX8Vector(n);
    CHECK_OOM(ptr->zf->data, "unable to allocate workspace array zf\n");
    memset(ptr->zf->data, 0, n * sizeof(float complex));

    ptr->zt = XLALCreateCOMPLEX8Vector(n);
    CHECK_OOM(ptr->zf->data, "unable to allocate workspace array zt\n");
    memset(ptr->zt->data, 0, n * sizeof(float complex));

    ptr->n = n;
    ptr->plan = XLALCreateReverseCOMPLEX8FFTPlan(n, 1);
    CHECK_OOM(ptr->plan, "unable to allocate plan");

    return ptr;
}

/* by default, complex arithmetic will call built-in function __muldc3, which does a lot of error checking for inf and nan; just do it manually */
static void multiply_conjugate(float complex * restrict out, float complex *a, float complex *b, const size_t size) {
    size_t k = 0;
    for (;k < size; ++k) {
        const float ar = crealf(a[k]);
        const float br = crealf(b[k]);
        const float ai = cimagf(a[k]);
        const float bi = cimagf(b[k]);
        __real__ out[k] = ar * br + ai * bi;
        __imag__ out[k] = ar * -bi + ai * br;
    }
}

static double abs_real(const float complex x) {
    const double re = crealf(x);
    return re;
}

static double abs2(const float complex x) {
    const double re = crealf(x);
    const double im = cimagf(x);
    return re * re + im * im;
}

/* interpolate the peak with a parabolic interpolation */
static double vector_peak_interp(const double ym1, const double y, const double yp1) {
    const double dy = 0.5 * (yp1 - ym1);
    const double d2y = 2. * y - ym1 - yp1;
    return y + 0.5 * dy * dy / d2y;
}

/*
 * Returns the match for two whitened, normalized, positive-frequency
 * COMPLEX8FrequencySeries inputs.
 */
double _SBankComputeMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws = get_workspace(workspace_cache, n);
    if (!ws) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }

    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws->zf->data, inj, tmplt, min_len);
    XLALCOMPLEX8VectorFFT(ws->zt, ws->zf, ws->plan); /* plan is reverse */

    /* maximize over |z(t)|^2 */
    float complex *zdata = ws->zt->data;
    size_t k = n;
    ssize_t argmax = -1;
    double max = 0.;
    for (;k--;) {
        double temp = abs2(zdata[k]);
        if (temp > max) {
            argmax = k;
            max = temp;
        }
    }
    if (max == 0.) return 0.;

    /* refine estimate of maximum */
    double result;
    if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        result = vector_peak_interp(abs2(zdata[argmax - 1]), abs2(zdata[argmax]), abs2(zdata[argmax + 1]));

    /* compute match */
    return 4. * delta_f * sqrt(result); 
}


/*
  Compute the overlap between a normalized template waveform h and a
  normalized signal proposal maximizing over the template h's overall
  amplitude. This is the most basic match function one can compute.
*/
double _SBankComputeRealMatch(complex *inj, complex *tmplt, size_t min_len, double delta_f, WS *workspace_cache) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws = get_workspace(workspace_cache, n);
    if (!ws) {
        XLALPrintError("out of space in the workspace_cache\n");
	exit(-1);
    }

    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws->zf->data, inj, tmplt, min_len);
    XLALCOMPLEX8VectorFFT(ws->zt, ws->zf, ws->plan); /* plan is reverse */

    /* maximize over |Re z(t)| */
    float complex *zdata = ws->zt->data;
    size_t k = n;
    double max = 0.;
    for (;k--;) {
	double temp = abs_real((zdata[k]));
	if (temp > max) {
	    max = temp;
	}
    }
    return 4. * delta_f * max;
}


/*
  Compute the overlap between a template waveform h and a signal
  proposal assuming only the (2,2) mode and maximizing over the
  template h's coalescence phase, overall amplitude and effective
  polarization / sky position. The function assumes that the plus and
  cross polarization hp and hc are both normalized to unity and that
  hphccorr is the correlation between these normalized components.
 */
double _SBankComputeMatchMaxSkyLoc(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws1 = get_workspace(workspace_cache1, n);
    if (!ws1) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws2 = get_workspace(workspace_cache2, n);
    if (!ws2) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }


    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws1->zf->data, hp, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws1->zt, ws1->zf, ws1->plan); /* plan is reverse */
    multiply_conjugate(ws2->zf->data, hc, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws2->zt, ws2->zf, ws2->plan);


    /* COMPUTE DETECTION STATISTIC */

    /* First start with constant values */
    double delta = 2 * hphccorr;
    double denom = 4 - delta * delta;
    if (denom < 0)
    {
        fprintf(stderr, "DANGER WILL ROBINSON: CODE IS BROKEN!!\n");
    }

    /* Now the tricksy bit as we loop over time*/
    float complex *hpdata = ws1->zt->data;
    float complex *hcdata = ws2->zt->data;
    size_t k = n;
    /* FIXME: This is needed if we turn back on peak refinement. */
    /*ssize_t argmax = -1;*/
    double max = 0.;
    for (;k--;) {
        double complex ratio = hcdata[k] / hpdata[k];
        double ratio_real = creal(ratio);
        double ratio_imag = cimag(ratio);
        double beta = 2 * ratio_real;
        double alpha = ratio_real * ratio_real + ratio_imag * ratio_imag;
        double sqroot = alpha*alpha + alpha * (delta*delta - 2) + 1;
        sqroot += beta * (beta - delta * (1 + alpha));
        sqroot = sqrt(sqroot);
        double brckt = 2*(alpha + 1) - beta*delta + 2*sqroot;
        brckt = brckt / denom;
        double det_stat_sq = abs2(hpdata[k]) * brckt;

        if (det_stat_sq > max) {
            /*argmax = k;*/
            max = det_stat_sq;
        }
    }
    if (max == 0.) return 0.;

    /* FIXME: For now do *not* refine estimate of peak. */
    /* double result;
    if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        result = vector_peak_interp(abs2(zdata[argmax - 1]), abs2(zdata[argmax]), abs2(zdata[argmax + 1])); */

    /* Return match */
    return 4. * delta_f * sqrt(max);
}

double _SBankComputeMatchMaxSkyLocNoPhase(complex *hp, complex *hc, const double hphccorr, complex *proposal, size_t min_len, double delta_f, WS *workspace_cache1, WS *workspace_cache2) {

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws1 = get_workspace(workspace_cache1, n);
    if (!ws1) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws2 = get_workspace(workspace_cache2, n);
    if (!ws2) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }


    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
       so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws1->zf->data, hp, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws1->zt, ws1->zf, ws1->plan); /* plan is reverse */
    multiply_conjugate(ws2->zf->data, hc, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws2->zt, ws2->zf, ws2->plan);


    /* COMPUTE DETECTION STATISTIC */

    /* First start with constant values */
    double denom = 1. - (hphccorr*hphccorr);
    if (denom < 0)
    {
        fprintf(stderr, "DANGER WILL ROBINSON: CODE IS BROKEN!!\n");
    }

    /* Now the tricksy bit as we loop over time*/
    float complex *hpdata = ws1->zt->data;
    float complex *hcdata = ws2->zt->data;
    size_t k = n;
    /* FIXME: This is needed if we turn back on peak refinement. */
    /*ssize_t argmax = -1;*/
    double max = 0.;
    double det_stat_sq;

    for (;k--;) {
        det_stat_sq = creal(hpdata[k])*creal(hpdata[k]);
        det_stat_sq += creal(hcdata[k])*creal(hcdata[k]);
        det_stat_sq -= 2*creal(hpdata[k])*creal(hcdata[k])*hphccorr;

        det_stat_sq = det_stat_sq / denom;

        if (det_stat_sq > max) {
            /*argmax = k;*/
            max = det_stat_sq;
        }
    }
    if (max == 0.) return 0.;

    /* FIXME: For now do *not* refine estimate of peak. */
    /* double result;
    if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        result = vector_peak_interp(abs2(zdata[argmax - 1]), abs2(zdata[argmax])
, abs2(zdata[argmax + 1])); */

    /* Return match */
    return 4. * delta_f * sqrt(max);
}

double _SBankComputeFiveCompMatch(complex *temp_comp1, complex *temp_comp2,
                                  complex *temp_comp3, complex *temp_comp4,
                                  complex *temp_comp5, complex *proposal,
                                  size_t min_len, double delta_f, int8_t num_comps,
                                  WS *workspace_cache1,
                                  WS *workspace_cache2, WS *workspace_cache3,
                                  WS *workspace_cache4, WS *workspace_cache5)
{
    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws1 = get_workspace(workspace_cache1, n);
    if (!ws1) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws2 = get_workspace(workspace_cache2, n);
    if (!ws2) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws3 = get_workspace(workspace_cache3, n);
    if (!ws3) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws4 = get_workspace(workspace_cache4, n);
    if (!ws4) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws5 = get_workspace(workspace_cache5, n);
    if (!ws5) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }

    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
     * so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws1->zf->data, temp_comp1, proposal, min_len);
    XLALCOMPLEX8VectorFFT(ws1->zt, ws1->zf, ws1->plan); /* plan is reverse */
    if (num_comps > 1) {
        multiply_conjugate(ws2->zf->data, temp_comp2, proposal, min_len);
	XLALCOMPLEX8VectorFFT(ws2->zt, ws2->zf, ws2->plan);
    }
    if (num_comps > 2) {
        multiply_conjugate(ws3->zf->data, temp_comp3, proposal, min_len);
	XLALCOMPLEX8VectorFFT(ws3->zt, ws3->zf, ws3->plan);
    }
    if (num_comps > 3) {
        multiply_conjugate(ws4->zf->data, temp_comp4, proposal, min_len);
	XLALCOMPLEX8VectorFFT(ws4->zt, ws4->zf, ws4->plan);
    }
    if (num_comps > 4) {
        multiply_conjugate(ws5->zf->data, temp_comp5, proposal, min_len);
	XLALCOMPLEX8VectorFFT(ws5->zt, ws5->zf, ws5->plan);
    }

    /* maximize over |z(t)|^2 */
    float complex *zdata1 = ws1->zt->data;
    float complex *zdata2 = ws2->zt->data;
    float complex *zdata3 = ws3->zt->data;
    float complex *zdata4 = ws4->zt->data;
    float complex *zdata5 = ws5->zt->data;

    size_t k = n;
    ssize_t argmax = -1;
    double max = 0.;
    for (;k--;) {
      double temp = 0.;
        if (num_comps == 1) {
	    temp = abs2(zdata1[k]);
	} else if (num_comps == 2) {
	    temp = abs2(zdata1[k]) + abs2(zdata2[k]);
	} else if (num_comps == 3) {
	    temp = abs2(zdata1[k]) + abs2(zdata2[k]) + abs2(zdata3[k]);
	} else if (num_comps == 4) {
	    temp = abs2(zdata1[k]) + abs2(zdata2[k]) + abs2(zdata3[k])
	         + abs2(zdata4[k]);
	} else if (num_comps == 5) {
	    temp = abs2(zdata1[k]) + abs2(zdata2[k]) + abs2(zdata3[k])
	         + abs2(zdata4[k]) + abs2(zdata5[k]);
	}
        if (temp > max) {
            argmax = k;
            max = temp;
        }
    }
    if (max == 0.) return 0.;

    /* refine estimate of maximum */
    double result;
    double templ, tempu;
    /*if (argmax == 0 || argmax == (ssize_t) n - 1)
        result = max;
    else
        templ = abs2(zdata1[argmax - 1]) + abs2(zdata2[argmax - 1])
              + abs2(zdata3[argmax - 1]) + abs2(zdata4[argmax - 1])
              + abs2(zdata5[argmax - 1]);
        tempu = abs2(zdata1[argmax + 1]) + abs2(zdata2[argmax + 1])
              + abs2(zdata3[argmax + 1]) + abs2(zdata4[argmax + 1]) 
              + abs2(zdata5[argmax + 1]);
        result = vector_peak_interp(templ, max, tempu);
        result = max;*/

    /* compute match */
    result = max;

    return 4. * delta_f * sqrt(result);
}

double _SBankComputeFiveCompFactorMatch(complex *temp_comp1, complex *temp_comp2,
					complex *temp_comp3, complex *temp_comp4,
					complex *temp_comp5, complex *prop_comp1,
					complex *prop_comp2, complex *prop_comp3,
					complex *prop_comp4, complex *prop_comp5,
					double *prop_factor1, double *prop_factor2,
					double *prop_factor3, double *prop_factor4,
					double *prop_factor5, size_t min_len,
					size_t f_len, double delta_f, int8_t num_comps,
					WS *workspace_cache1,
					WS *workspace_cache2, WS *workspace_cache3,
					WS *workspace_cache4, WS *workspace_cache5,
					WS *workspace_cache6, WS *workspace_cache7,
					WS *workspace_cache8, WS *workspace_cache9,
					WS *workspace_cache10, WS *workspace_cache11,
					WS *workspace_cache12, WS *workspace_cache13,
					WS *workspace_cache14, WS *workspace_cache15,
					WS *workspace_cache16, WS *workspace_cache17,
					WS *workspace_cache18, WS *workspace_cache19,
					WS *workspace_cache20, WS *workspace_cache21,
					WS *workspace_cache22, WS *workspace_cache23,
					WS *workspace_cache24, WS *workspace_cache25)
{

    /* get workspace for + and - frequencies */
    size_t n = 2 * (min_len - 1);   /* no need to integrate implicit zeros */
    WS *ws1 = get_workspace(workspace_cache1, n);
    if (!ws1) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws2 = get_workspace(workspace_cache2, n);
    if (!ws2) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws3 = get_workspace(workspace_cache3, n);
    if (!ws3) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws4 = get_workspace(workspace_cache4, n);
    if (!ws4) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws5 = get_workspace(workspace_cache5, n);
    if (!ws5) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws6 = get_workspace(workspace_cache6, n);
    if (!ws6) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws7 = get_workspace(workspace_cache7, n);
    if (!ws7) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws8 = get_workspace(workspace_cache8, n);
    if (!ws8) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws9 = get_workspace(workspace_cache9, n);
    if (!ws9) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws10 = get_workspace(workspace_cache10, n);
    if (!ws10) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws11 = get_workspace(workspace_cache11, n);
    if (!ws11) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws12 = get_workspace(workspace_cache12, n);
    if (!ws12) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws13 = get_workspace(workspace_cache13, n);
    if (!ws13) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws14 = get_workspace(workspace_cache14, n);
    if (!ws14) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws15 = get_workspace(workspace_cache15, n);
    if (!ws15) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws16 = get_workspace(workspace_cache16, n);
    if (!ws16) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws17 = get_workspace(workspace_cache17, n);
    if (!ws17) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws18 = get_workspace(workspace_cache18, n);
    if (!ws18) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws19 = get_workspace(workspace_cache19, n);
    if (!ws19) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws20 = get_workspace(workspace_cache20, n);
    if (!ws20) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws21 = get_workspace(workspace_cache21, n);
    if (!ws21) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws22 = get_workspace(workspace_cache22, n);
    if (!ws22) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws23 = get_workspace(workspace_cache23, n);
    if (!ws23) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws24 = get_workspace(workspace_cache24, n);
    if (!ws24) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }
    WS *ws25 = get_workspace(workspace_cache25, n);
    if (!ws25) {
        XLALPrintError("out of space in the workspace_cache\n");
        exit(-1);
    }

    /* compute complex SNR time-series in freq-domain, then time-domain */
    /* Note that findchirp paper eq 4.2 defines a positive-frequency integral,
     * so we should only fill the positive frequencies (first half of zf). */
    multiply_conjugate(ws1->zf->data, temp_comp1, prop_comp1, min_len);
    multiply_conjugate(ws2->zf->data, temp_comp1, prop_comp2, min_len);
    multiply_conjugate(ws3->zf->data, temp_comp1, prop_comp3, min_len);
    multiply_conjugate(ws4->zf->data, temp_comp1, prop_comp4, min_len);
    multiply_conjugate(ws5->zf->data, temp_comp1, prop_comp5, min_len);
    XLALCOMPLEX8VectorFFT(ws1->zt, ws1->zf, ws1->plan);
    XLALCOMPLEX8VectorFFT(ws2->zt, ws2->zf, ws2->plan);
    XLALCOMPLEX8VectorFFT(ws3->zt, ws3->zf, ws3->plan);
    XLALCOMPLEX8VectorFFT(ws4->zt, ws4->zf, ws4->plan);
    XLALCOMPLEX8VectorFFT(ws5->zt, ws5->zf, ws5->plan);
    if (num_comps > 1) {
        multiply_conjugate(ws6->zf->data, temp_comp2, prop_comp1, min_len);
        multiply_conjugate(ws7->zf->data, temp_comp2, prop_comp2, min_len);
        multiply_conjugate(ws8->zf->data, temp_comp2, prop_comp3, min_len);
        multiply_conjugate(ws9->zf->data, temp_comp2, prop_comp4, min_len);
        multiply_conjugate(ws10->zf->data, temp_comp2, prop_comp5, min_len);
	XLALCOMPLEX8VectorFFT(ws6->zt, ws6->zf, ws6->plan);
	XLALCOMPLEX8VectorFFT(ws7->zt, ws7->zf, ws7->plan);
	XLALCOMPLEX8VectorFFT(ws8->zt, ws8->zf, ws8->plan);
	XLALCOMPLEX8VectorFFT(ws9->zt, ws9->zf, ws9->plan);
	XLALCOMPLEX8VectorFFT(ws10->zt, ws10->zf, ws10->plan);
    }
    if (num_comps > 2) {
        multiply_conjugate(ws11->zf->data, temp_comp3, prop_comp1, min_len);
        multiply_conjugate(ws12->zf->data, temp_comp3, prop_comp2, min_len);
        multiply_conjugate(ws13->zf->data, temp_comp3, prop_comp3, min_len);
        multiply_conjugate(ws14->zf->data, temp_comp3, prop_comp4, min_len);
        multiply_conjugate(ws15->zf->data, temp_comp3, prop_comp5, min_len);
	XLALCOMPLEX8VectorFFT(ws11->zt, ws11->zf, ws11->plan);
	XLALCOMPLEX8VectorFFT(ws12->zt, ws12->zf, ws12->plan);
	XLALCOMPLEX8VectorFFT(ws13->zt, ws13->zf, ws13->plan);
	XLALCOMPLEX8VectorFFT(ws14->zt, ws14->zf, ws14->plan);
	XLALCOMPLEX8VectorFFT(ws15->zt, ws15->zf, ws15->plan);
    }
    if (num_comps > 3) {
        multiply_conjugate(ws16->zf->data, temp_comp4, prop_comp1, min_len);
        multiply_conjugate(ws17->zf->data, temp_comp4, prop_comp2, min_len);
        multiply_conjugate(ws18->zf->data, temp_comp4, prop_comp3, min_len);
        multiply_conjugate(ws19->zf->data, temp_comp4, prop_comp4, min_len);
        multiply_conjugate(ws20->zf->data, temp_comp4, prop_comp5, min_len);
	XLALCOMPLEX8VectorFFT(ws16->zt, ws16->zf, ws16->plan);
	XLALCOMPLEX8VectorFFT(ws17->zt, ws17->zf, ws17->plan);
	XLALCOMPLEX8VectorFFT(ws18->zt, ws18->zf, ws18->plan);
	XLALCOMPLEX8VectorFFT(ws19->zt, ws19->zf, ws19->plan);
	XLALCOMPLEX8VectorFFT(ws20->zt, ws20->zf, ws20->plan);
    }
    if (num_comps > 4) {
        multiply_conjugate(ws21->zf->data, temp_comp5, prop_comp1, min_len);
        multiply_conjugate(ws22->zf->data, temp_comp5, prop_comp2, min_len);
        multiply_conjugate(ws23->zf->data, temp_comp5, prop_comp3, min_len);
        multiply_conjugate(ws24->zf->data, temp_comp5, prop_comp4, min_len);
        multiply_conjugate(ws25->zf->data, temp_comp5, prop_comp5, min_len);
	XLALCOMPLEX8VectorFFT(ws21->zt, ws21->zf, ws21->plan);
	XLALCOMPLEX8VectorFFT(ws22->zt, ws22->zf, ws22->plan);
	XLALCOMPLEX8VectorFFT(ws23->zt, ws23->zf, ws23->plan);
	XLALCOMPLEX8VectorFFT(ws24->zt, ws24->zf, ws24->plan);
	XLALCOMPLEX8VectorFFT(ws25->zt, ws25->zf, ws25->plan);
    }

    /* maximize over |z(t)|^2 */
    float complex *zdata1 = ws1->zt->data;
    float complex *zdata2 = ws2->zt->data;
    float complex *zdata3 = ws3->zt->data;
    float complex *zdata4 = ws4->zt->data;
    float complex *zdata5 = ws5->zt->data;
    float complex *zdata6 = ws6->zt->data;
    float complex *zdata7 = ws7->zt->data;
    float complex *zdata8 = ws8->zt->data;
    float complex *zdata9 = ws9->zt->data;
    float complex *zdata10 = ws10->zt->data;
    float complex *zdata11 = ws11->zt->data;
    float complex *zdata12 = ws12->zt->data;
    float complex *zdata13 = ws13->zt->data;
    float complex *zdata14 = ws14->zt->data;
    float complex *zdata15 = ws15->zt->data;
    float complex *zdata16 = ws16->zt->data;
    float complex *zdata17 = ws17->zt->data;
    float complex *zdata18 = ws18->zt->data;
    float complex *zdata19 = ws19->zt->data;
    float complex *zdata20 = ws20->zt->data;
    float complex *zdata21 = ws21->zt->data;
    float complex *zdata22 = ws22->zt->data;
    float complex *zdata23 = ws23->zt->data;
    float complex *zdata24 = ws24->zt->data;
    float complex *zdata25 = ws25->zt->data;

    double top = 0.;
    double bot = 0.;
    size_t l = f_len;
    for (;l--;) {
        double max_temp = 0.;
        double max_match = 0.;
        double amp1 = prop_factor1[l] * prop_factor1[l];
        double amp2 = prop_factor2[l] * prop_factor2[l];
        double amp3 = prop_factor3[l] * prop_factor3[l];
        double amp4 = prop_factor4[l] * prop_factor4[l];
        double amp5 = prop_factor5[l] * prop_factor5[l];
	double amp = amp1 + amp2 + amp3 + amp4 + amp5;
	size_t k = n;
	for (;k--;) {
	    double temp = 0.;
	    double match = 0.;
	    temp += abs2(zdata1[k]) * amp1;
	    temp += abs2(zdata2[k]) * amp2;
	    temp += abs2(zdata3[k]) * amp3;
	    temp += abs2(zdata4[k]) * amp4;
	    temp += abs2(zdata5[k]) * amp5;
	    if (num_comps > 1) {
		temp += abs2(zdata6[k]) * amp1;
		temp += abs2(zdata7[k]) * amp2;
		temp += abs2(zdata8[k]) * amp3;
		temp += abs2(zdata9[k]) * amp4;
		temp += abs2(zdata10[k]) * amp5;
	    }
	    if (num_comps > 2) {
		temp += abs2(zdata11[k]) * amp1;
		temp += abs2(zdata12[k]) * amp2;
		temp += abs2(zdata13[k]) * amp3;
		temp += abs2(zdata14[k]) * amp4;
		temp += abs2(zdata15[k]) * amp5;
	    }
	    if (num_comps > 3) {
		temp += abs2(zdata16[k]) * amp1;
		temp += abs2(zdata17[k]) * amp2;
		temp += abs2(zdata18[k]) * amp3;
		temp += abs2(zdata19[k]) * amp4;
		temp += abs2(zdata20[k]) * amp5;
	    }
	    if (num_comps > 4) {
		temp += abs2(zdata21[k]) * amp1;
		temp += abs2(zdata22[k]) * amp2;
		temp += abs2(zdata23[k]) * amp3;
		temp += abs2(zdata24[k]) * amp4;
		temp += abs2(zdata25[k]) * amp5;
	    }
	    match = temp / amp;
	    if (match > max_match) {
	        max_temp = temp;
	        max_match = match;
	    }

	}
	top += pow(max_temp, 1.5);
	bot += pow(amp, 1.5);
    }
    double result;
    result = 4. * delta_f * pow(top / bot, 1./3.);
    return result;
}
