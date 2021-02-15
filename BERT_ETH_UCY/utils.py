"""Plotting functions for visualizing distributions."""
from __future__ import division
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.collections import LineCollection
import warnings
from distutils.version import LooseVersion

from six import string_types

try:
    import statsmodels.nonparametric.api as smnp
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

from seaborn.utils import iqr, _kde_support
from seaborn.palettes import  color_palette, light_palette, dark_palette, blend_palette




def kdeplot(data, data2=None, shade=False, vertical=False, kernel="gau",
            bw="scott", gridsize=100, cut=3, clip=None, legend=True,
            cumulative=False, shade_lowest=True, cbar=False, cbar_ax=None,
            cbar_kws=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if isinstance(data, list):
        data = np.asarray(data)

    if len(data) == 0:
        return ax

    data = data.astype(np.float64)
    if data2 is not None:
        if isinstance(data2, list):
            data2 = np.asarray(data2)
        data2 = data2.astype(np.float64)

    warn = False
    bivariate = False

    if data2 is not None:
        bivariate = True
        x = data
        y = data2

    if warn:
        warn_msg = ("Passing a 2D dataset for a bivariate plot is deprecated "
                    "in favor of kdeplot(x, y), and it will cause an error in "
                    "future versions. Please update your code.")
        warnings.warn(warn_msg, UserWarning)

    if bivariate and cumulative:
        raise TypeError("Cumulative distribution plots are not"
                        "supported for bivariate distributions.")
    if bivariate:
        ax = _bivariate_kdeplot(x, y, shade, shade_lowest,
                                kernel, bw, gridsize, cut, clip, legend,
                                cbar, cbar_ax, cbar_kws, ax, **kwargs)
    else:
        pass

    return ax



def _bivariate_kdeplot(x, y, filled, fill_lowest,
                       kernel, bw, gridsize, cut, clip,
                       axlabel, cbar, cbar_ax, cbar_kws, ax, **kwargs):
    """Plot a joint KDE estimate as a bivariate contour plot."""
    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]

    x_support = np.linspace(x.min(), x.max(), gridsize)
    y_support = np.linspace(y.min(), y.max(), gridsize)
    xx=[]
    yy=[]
    z=[]
    for i in range(x.shape[0]):
        if _has_statsmodels:
            xx_, yy_, z_ = _statsmodels_bivariate_kde(x[i], y[i],x_support,y_support, bw, gridsize, cut, clip)
        xx.append(xx_)
        yy.append(yy_)
        z.append(z_)

    # Plot the contours
    xx=xx[0]
    yy=yy[0]
    zz=np.stack(z,-1)
    zz[np.isnan(zz)] = 0
    #zz=zz / (100)
    zz=zz/(zz.max((0,1))+1e-10)
    #zz=np.log(zz+1)
    z=zz.max(-1)

    n_levels = kwargs.pop("n_levels", 10) + 10

    scout, = ax.plot([], [])
    default_color = scout.get_color()
    scout.remove()

    color = kwargs.pop("color", default_color)
    cmap = kwargs.pop("cmap", None)
    if cmap is None:
        if filled:
            cmap = light_palette(color, as_cmap=True)
        else:
            cmap = dark_palette(color, as_cmap=True)
    if isinstance(cmap, string_types):
        if cmap.endswith("_d"):
            pal = ["#333333"]
            pal.extend(color_palette(cmap.replace("_d", "_r"), 2))
            cmap = blend_palette(pal, as_cmap=True)
        else:
            cmap = mpl.cm.get_cmap(cmap)

    label = kwargs.pop("label", None)

    kwargs["cmap"] = cmap
    contour_func = ax.contourf if filled else ax.contour
    cset = contour_func(xx, yy, z, n_levels, **kwargs)
    if filled and not fill_lowest:
        cset.collections[0].set_alpha(0)
    kwargs["n_levels"] = n_levels

    if cbar:
        cbar_kws = {} if cbar_kws is None else cbar_kws
        ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

    # Label the axes
    if hasattr(x, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(y, "name") and axlabel:
        ax.set_ylabel(y.name)

    if label is not None:
        legend_color = cmap(.95) if color is None else color
        if filled:
            ax.fill_between([], [], color=legend_color, label=label)
        else:
            ax.plot([], [], color=legend_color, label=label)

    return ax



def _statsmodels_bivariate_kde(x, y, x_support,y_support,bw, gridsize, cut, clip):
    """Compute a bivariate kde using statsmodels."""
    if isinstance(bw, string_types):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T, bw_method=bw)
    data_std = data.std(axis=0, ddof=1)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw_x = getattr(kde, "%s_factor" % bw)() * data_std[0]
        bw_y = getattr(kde, "%s_factor" % bw)() * data_std[1]
    elif np.isscalar(bw):
        bw_x, bw_y = bw, bw
    else:
        msg = ("Cannot specify a different bandwidth for each dimension "
               "with the scipy backend. You should install statsmodels.")
        raise ValueError(msg)
    x_support = _kde_support(data[:, 0], bw_x, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw_y, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z