import os
import pickle
import itertools
import numpy as np
import networkx as nx
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score  # sklearn ver.=1.1.1.
import statsmodels.api as sm
from fuzzywuzzy import fuzz
import re
import pandas as pd  # For table printing.
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# I/O to Rds.
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


SENT2LAB = {1: "Favorable Sentiment", 0: "Neutral Sentiment", -1: "Critical Sentiment"}
SENT2IDX = {1: 0, 0: 1, -1: 2}
SENT_COLORS = ["#504DB2", "#414042", "#B2504D"]  # POS, NEU, NEG
COSTRA_COLORS = ["#2CBEC6", "#F59448"]  # collaborators, non-collaborators


def savePKL(dir_out, fname, file):
    with open(os.path.join(dir_out, f"{fname}.pkl"), "wb") as f:
        pickle.dump(file, f)


def loadPKL(dir_in, fname):
    with open(os.path.join(dir_in, f"{fname}.pkl"), "rb") as f:
        file = pickle.load(f)
    return file


def save_Rds(dir_, fname, df):
    # Convert pandas df to R df.
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)

    # Save R df as .rds file.
    r_file = os.path.join(dir_, f"{fname}.Rds")
    robjects.r.assign("my_df_tosave", r_df)
    robjects.r(f"saveRDS(my_df_tosave, file='{r_file}')")


def read_Rds(dir_, fname):
    # Load as R df from .rds file.
    r_file = os.path.join(dir_, f"{fname}.Rds")
    robjects.r(f"df_to_load <- readRDS('{r_file}')")
    r_df = robjects.r["df_to_load"]

    # Convert R df to pandas df.
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_df)

    return df


def get_mean_with_spread(nparr, axis=None, use_SEM=False):
    if use_SEM:
        denom = np.sqrt(nparr.size) if axis is None else np.sqrt(nparr.shape[axis])
    else:
        denom = 1
    return np.mean(nparr, axis=axis), np.std(nparr, axis=axis, ddof=1) / denom


def set_yticks(ax, major_len=6.5, minor_len=4, major_width=1.5, minor_width=1, labelsize=10, alt=0):
    if ax.get_yscale() == "log":
        # Log scale: decades as majors, 2–9 as minors.
        # Major ticks at each power of 10.
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=15))
        # ax.yaxis.set_major_formatter(mticker.ScalarFormatter())  # Tick label 1,10,100.
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())  # Tick label 10^0, 10^1, 10^2.
        # Minor ticks between decades (2,3,...9).
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        yticks = ax.get_yticks()
        major_ticks = yticks[alt % 2 :: 2]
        minor_ticks = yticks[(alt + 1) % 2 :: 2]
        ax.yaxis.set_major_locator(mticker.FixedLocator(major_ticks))
        ax.yaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())  # Hide minor tick labels.
    ax.tick_params(axis="y", which="major", length=major_len, width=major_width, labelsize=labelsize)
    ax.tick_params(axis="y", which="minor", length=minor_len, width=minor_width)


def set_xticks(ax, major_len=6.5, minor_len=4, major_width=1.5, minor_width=1, labelsize=10, alt=0):
    if ax.get_xscale() == "log":
        # Log scale: decades as majors, 2–9 as minors.
        # Major ticks at each power of 10.
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=15))
        # ax.yaxis.set_major_formatter(mticker.ScalarFormatter())  # Tick label 1,10,100.
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())  # Tick label 10^0, 10^1, 10^2.
        # Minor ticks between decades (2,3,...9).
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        xticks = ax.get_xticks()
        major_ticks = xticks[alt % 2 :: 2]
        minor_ticks = xticks[(alt + 1) % 2 :: 2]
        ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
        ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())  # Hide minor tick labels.
    ax.tick_params(axis="x", which="major", length=major_len, width=major_width, labelsize=labelsize)
    ax.tick_params(axis="x", which="minor", length=minor_len, width=minor_width)


def assert_in_xylims(x, y, xlim=None, ylim=None):
    # Check if all the points (center, not accounting for error bars) are inside the limits
    # so that they are visible in the figures.
    if xlim is not None:
        for x_ in x:
            assert xlim[0] < x_ < xlim[1], f"xlim={xlim} doesn't cover all the points, missing x={x_}."

    if ylim is not None:
        for y_ in y:
            assert ylim[0] < y_ < ylim[1], f"ylim={ylim} doesn't cover all the points, missing y={y_}."


def get_xylims_noerr(x, y, margin=0.05):
    # Get xylims such that all points specified by x,y coordinates are covered.
    # margin is percentage of width/height beyond the points boundary.
    dx = (np.max(x) - np.min(x)) * margin
    dy = (np.max(y) - np.min(y)) * margin
    xlim = (np.min(x) - dx, np.max(x) + dx)
    ylim = (np.min(y) - dy, np.max(y) + dy)
    return xlim, ylim


def print_sigfig(num, sig=2):  # Show whole number if integer, or 2 decimal points if non-integer non "0.", otherwise 2 sig figs.
    if float(num).is_integer():
        return f"{num:.0f}"
    elif abs(num) >= 1:
        return f"{num:.{sig}f}"
    else:
        return f"{num:.{sig}g}"


def pval_print(pval):
    if pval >= 0.001:
        return f"{pval:.3g}"
    else:
        return f"{pval:.3e}"


def pval_star(pval, star=False, show_insig=True, show_p=True, nature_style=True):
    if nature_style:
        if pval >= 0.001:
            return f"{'p='*show_p}{print_sigfig(pval, sig=2)}"
        else:
            return f"{'p'*show_p}<0.001"

    if star:
        return "" + "*" * (pval < 0.05) + "*" * (pval < 0.01) + "*" * (pval < 0.001) + "*" * (pval < 0.0001)
    else:
        if pval >= 0.05:
            if show_insig:
                return f"{'p='*show_p}{pval:.2f}"
            else:
                return f"{'p'*show_p}≥0.05"
        elif 0.01 <= pval < 0.05:
            return f"{'p'*show_p}<0.05"
        elif 0.001 <= pval < 0.01:
            return f"{'p'*show_p}<0.01"
        elif 0.0001 <= pval < 0.001:
            return f"{'p'*show_p}<0.001"
        else:
            return f"{'p'*show_p}<0.0001"


def calculate_stats(x, y, yerr, spearman=True):
    """Calculate several stats for the linear relationship:
    1) Weighted linear regression
        adjusted R^2, coef_arr, F-test score and pval
        coef_arr:
            row: factor/predictor
            col: coef, t, p>|t|, CI_lower, CI_upper
    2) Spearman: size and pval

    for tmp_fit, see:
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
    beta = tmp_fit.params  # intercept (beta0), beta1

    Args:
        yerr: Standard deviation of y.
            weights = 1/yvar, yvar is variance of y, so yvar = yerr ** 2
    """
    stat_dict = dict()
    tmp_fit = sm.WLS(y, sm.add_constant(x, has_constant="add"), weights=1 / (yerr**2)).fit()
    if spearman:
        tmp_spear = stats.spearmanr(x, y)
        stat_dict["spearman"] = tmp_spear.correlation
        stat_dict["spearmanp"] = tmp_spear.pvalue

    stat_dict["rsqr_a"] = tmp_fit.rsquared_adj
    stat_dict["fs"] = (np.round(tmp_fit.df_model).astype(int), np.round(tmp_fit.df_resid).astype(int))
    stat_dict["f"] = tmp_fit.fvalue
    stat_dict["fp"] = tmp_fit.f_pvalue
    stat_dict["coef_arr"] = np.array(
        [tmp_fit.params, tmp_fit.tvalues, tmp_fit.pvalues, tmp_fit.conf_int()[:, 0], tmp_fit.conf_int()[:, 1]]
    ).T

    stat_dict["yhat"] = tmp_fit.fittedvalues
    # stat_dict["cov"] = tmp_fit.cov_params()
    stat_dict["resid"] = tmp_fit.resid
    stat_dict["df_model"] = tmp_fit.df_model  # k-1 (i.e., p)
    stat_dict["df_resid"] = tmp_fit.df_resid  # N-k (k=p+1), p is num of regressor; +1 cuz constant.
    return stat_dict


def calculate_WLS_CI_UNUSED(X, y, yerr):
    slr_stats = calculate_stats(X, y, yerr, spearman=False)
    # Confidence interval (95%) for the (weighted) linear fit:
    n = len(y)
    s_err = np.sqrt(np.sum(slr_stats["resid"] ** 2) / (n - 2))  # standard deviation of the error (residuals)
    t = stats.t.ppf(0.975, n - 2)
    ci = t * s_err * np.sqrt(1 / n + (X - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2))

    return slr_stats, ci


def get_list_from_stat_dict(stat_dict):
    comp0 = stat_dict["coef_arr"][1, 0]  # Slope.
    comp0l = stat_dict["coef_arr"][1, 3]  # 95% confidence interval, lower.
    comp0u = stat_dict["coef_arr"][1, 4]  # 95% confidence interval, upper.
    comp1 = stat_dict["rsqr_a"]
    comp2 = f"F({stat_dict['fs'][0]},{stat_dict['fs'][1]})={print_sigfig(stat_dict['f'])}"
    comp3 = pval_star(stat_dict["fp"], star=False, show_insig=True, show_p=False)
    comp4 = stat_dict["spearman"]
    comp5 = pval_star(stat_dict["spearmanp"], star=False, show_insig=True, show_p=False)
    return [
        f"{print_sigfig(comp0)}",
        f"[{print_sigfig(comp0l)},{print_sigfig(comp0u)}]",
        f"{print_sigfig(comp1)}",
        comp2,
        comp3,
        f"{print_sigfig(comp4)}",
        comp5,
    ]


def print_stats(rmrs, grps, x_factor=None, x_lab=None, sent=-1, use_SEM=False, err_max=np.inf):
    data = []
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1, ddof=1) / denom_sem

    if x_factor is None:
        ####### bias vs. mean.
        x, xerr, y, yerr = crit_mean, crit_err, bias_m, bias_e
        m_ = (xerr <= err_max) & (yerr <= err_max)
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Bias vs. Mean'] + get_list_from_stat_dict(stat_dict))

    if x_factor is not None:
        ####### mean, collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd1_m, crit_cd1_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (Collab) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))
        ####### mean, non-collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd2p_m, crit_cd2p_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (Non-Collab) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))
        ####### mean, both collab and non-collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (All) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))

        ####### bias.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), bias_m, bias_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Bias vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))

    columns = ["Type", "Slope", "95% CI", "Adjusted R-squared", "F-test Statistic", "F-test p-value"]
    columns.extend(["Spearman Correlation", "Spearman p-value"])
    df = pd.DataFrame(data=data, columns=columns)
    # df.reset_index(drop=True, inplace=True)  # Remove index column.
    return df


def print_stats_lite(rmrs, grps, x_factor, x_lab=None, sent=-1, use_SEM=False):
    data = []
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem

    ####### mean, both collab and non-collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
    x, xerr, y, yerr, text = x, xerr, y, yerr, grps
    stat_dict = calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True)
    data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (All) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))

    columns = ["Type", "Slope", "95% CI", "Adjusted R-squared", "F-test Statistic", "F-test p-value"]
    columns.extend(["Spearman Correlation", "Spearman p-value"])
    df = pd.DataFrame(data=data, columns=columns)
    # df.reset_index(drop=True, inplace=True)  # Remove index column.
    return df


def calculate_WLS_stats(x, y, yerr, x_pred=None, spearman=True):
    """Calculate several stats for the linear relationship:
    Combines calculate_stats() and calculate_WLS_CI().
    1) Weighted linear regression
        fitted model, predictions + 95% CIs for mean response.
        adjusted R^2, coef_arr, F-test score and pval
        coef_arr:
            row: factor/predictor
            col: coef, t, p>|t|, CI_lower, CI_upper
    2) Spearman: size and pval

    for tmp_fit, see:
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
    beta = tmp_fit.params  # intercept (beta0), beta1

    Args:
        yerr: Standard deviation of y.
            weights = 1/yvar, yvar is variance of y, so yvar = yerr ** 2
    """
    stat_dict = dict()
    tmp_fit = sm.WLS(y, sm.add_constant(x, has_constant="add"), weights=1 / (yerr**2)).fit()
    if spearman:
        tmp_spear = stats.spearmanr(x, y)
        stat_dict["spearman"] = tmp_spear.correlation
        stat_dict["spearmanp"] = tmp_spear.pvalue

    stat_dict["model"] = tmp_fit
    stat_dict["x_pred"] = x if x_pred is None else x_pred
    pred_summary = tmp_fit.get_prediction(sm.add_constant(stat_dict["x_pred"])).summary_frame(alpha=0.05)
    stat_dict["y_pred"] = pred_summary["mean"]
    stat_dict["y_lower"] = pred_summary["mean_ci_lower"]
    stat_dict["y_upper"] = pred_summary["mean_ci_upper"]

    stat_dict["rsqr_a"] = tmp_fit.rsquared_adj
    stat_dict["fs"] = (np.round(tmp_fit.df_model).astype(int), np.round(tmp_fit.df_resid).astype(int))
    stat_dict["f"] = tmp_fit.fvalue
    stat_dict["fp"] = tmp_fit.f_pvalue
    stat_dict["coef_arr"] = np.array(
        [tmp_fit.params, tmp_fit.tvalues, tmp_fit.pvalues, tmp_fit.conf_int()[:, 0], tmp_fit.conf_int()[:, 1]]
    ).T

    stat_dict["yhat"] = tmp_fit.fittedvalues
    # stat_dict["cov"] = tmp_fit.cov_params()
    stat_dict["resid"] = tmp_fit.resid
    stat_dict["df_model"] = tmp_fit.df_model  # k-1 (i.e., p)
    stat_dict["df_resid"] = tmp_fit.df_resid  # N-k (k=p+1), p is num of regressor; +1 cuz constant.
    return stat_dict


def plot_scatter_and_fit(
    ax,
    x,
    y,
    yerr=None,
    text_arr=None,
    text_idx=None,
    color_="grey",
    show_scatter=True,
    show_stats=True,
    caps=False,
    alpha_err=0.33,
    xerr=None,
    alpha_pt=1,
    xlim=None,
):
    """

    Args:
        text_arr (np.arr): Text to annotate with; same length as x/y.
        text_idx (idx for x/y): Corresponding index for x/y to annotate; this can be shorter in length.
        xlim (len-2 list): if xlim is None, then line and shade cover xlim instead of x.
    """
    if show_scatter:
        ax.scatter(x, y, color=color_, marker=".", s=50, edgecolors="none", alpha=alpha_pt, zorder=0)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            linestyle="None",
            marker=None,
            markersize=2.5,
            elinewidth=0.75,
            color=color_,
            zorder=0,
            alpha=alpha_err,
        )
    # Weighted linear fit (or if yerr is None, it's OLS).
    sor_m = np.argsort(x)
    x_pred = np.linspace(xlim[0], xlim[1], 100) if xlim is not None else x[sor_m]
    if yerr is None:
        stat_dict = calculate_WLS_stats(x[sor_m], y[sor_m], 1.0, x_pred=x_pred, spearman=False)
    else:
        stat_dict = calculate_WLS_stats(x[sor_m], y[sor_m], yerr[sor_m], x_pred=x_pred, spearman=False)
    # Plot fit line and region.
    ax.plot(x_pred, stat_dict["y_pred"], color=color_, alpha=1)
    ax.fill_between(x_pred, stat_dict["y_lower"], stat_dict["y_upper"], color=color_, alpha=0.2, zorder=0, edgecolor=None)
    # Show stats on the plot.
    if show_stats:
        styles_txt = dict(fontsize=8, fontweight="normal", horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)
        xcoor, ycoor = 0.975, 0.975
        # if color_ != "grey":  # For collaborator/non-collaborator. Deprecated.
        #     xcoor, ycoor = 0.05 + 0.47 * int(color_ == "#F59448"), 0.95
        fval = stat_dict["f"]
        f_pval = stat_dict["fp"]
        s = stat_dict["coef_arr"][1, 0]
        s_ci = stat_dict["coef_arr"][1, 3:5]
        n_numer = np.round(stat_dict["df_model"]).astype(int)
        n_denom = np.round(stat_dict["df_resid"]).astype(int)
        txt_sig = (
            rf"$F_{{{n_numer},{n_denom}}}={print_sigfig(fval)}$  ${pval_star(f_pval)}$ ${'R_a'}^2={print_sigfig(stat_dict['rsqr_a'])}$"
        )
        ax.text(xcoor, ycoor, txt_sig, color=color_, **styles_txt)
        # txt_effect = rf"$s={s:.2g}$  $95\%$CI$=[{s_ci[0]:.2g},{s_ci[1]:.2g}]$"
        # ax.text(xcoor, ycoor - 0.06, txt_effect, color=color_, **styles_txt)

    styles_txt = dict(
        fontsize=8, fontweight="normal", horizontalalignment="center", verticalalignment="bottom", xytext=(0, 2), textcoords="offset points"
    )
    color_2 = "black" if color_ == "grey" else color_
    if text_idx is not None:
        for idx, txt in zip(text_idx, text_arr[text_idx]):
            txt = txt.capitalize() if caps else txt
            ax.annotate(txt, (x[idx], y[idx]), color=color_2, zorder=30, **styles_txt)


def plot_scatter_no_fit(
    ax, x, y, xerr, yerr, text_arr=None, text_idx=None, color_="grey", show_scatter=True, caps=False, alpha_err=0.33, alpha_pt=1
):
    """Same as plot_scatter_and_fit() but no fitting."""
    if show_scatter:
        ax.scatter(x, y, color=color_, marker=".", s=50, edgecolors="none", alpha=alpha_pt, zorder=0)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            linestyle="None",
            marker=None,
            markersize=2.5,
            elinewidth=0.75,
            color=color_,
            zorder=0,
            alpha=alpha_err,
        )

    styles_txt = dict(
        fontsize=8, fontweight="normal", horizontalalignment="center", verticalalignment="bottom", xytext=(0, 2), textcoords="offset points"
    )
    color_2 = "black" if color_ == "grey" else color_
    if text_idx is not None:
        for idx, txt in zip(text_idx, text_arr[text_idx]):
            txt = txt.capitalize() if caps else txt
            ax.annotate(txt, (x[idx], y[idx]), color=color_2, zorder=30, **styles_txt)


def plot_overview_2_subplots_no_fit(
    rmrs, grps, grps_subset, dir_, xlims=None, ylims=None, sent=-1, caps=False, alpha_err=0.33, use_SEM=False, err_max=np.inf, by=None
):
    # err_max: Maximum standard deviation allowed for points to be considered.
    if not np.isinf(err_max):
        print("DEBUG: err_max is not np.inf.")
    if xlims is not None:
        xlim1, xlim2 = xlims
    else:
        xlims = [None, None]
    if ylims is not None:
        ylim1, ylim2 = ylims
    else:
        ylims = [None, None]
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1, ddof=1) / denom_sem

    fig, axes = plt.subplots(1, 2, figsize=(3.41 * 2, 3.41 * 1))
    ax = axes[0]
    ####### Make non-collab vs. collab subplot.
    x, xerr, y, yerr = crit_cd1_m, crit_cd1_e, crit_cd2p_m, crit_cd2p_e
    m_ = (xerr <= err_max) & (yerr <= err_max)
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    ax.scatter(x, y, color="grey", marker=".", s=50, edgecolors="none", alpha=1, zorder=0)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle="None", marker=None, markersize=2.5, elinewidth=0.75, color="grey", alpha=alpha_err)
    styles_txt = dict(horizontalalignment="center", verticalalignment="bottom", xytext=(0, 2), textcoords="offset points")
    for idx, txt in zip(text_idx, text[text_idx]):
        txt = txt.capitalize() if caps else txt
        ax.annotate(txt, (x[idx], y[idx]), color="black", zorder=30, **styles_txt)
    #### Miscellanious.
    ax.set_xlabel(f"{SENT2LAB[sent]} to Collaborators")
    ax.set_ylabel(f"{SENT2LAB[sent]} to Non-Collaborators")
    if xlims[0] is None:
        # xlim1 = ax.get_xlim()
        xlim1, _ = get_xylims_noerr(x, y, margin=0.05)
    if ylims[0] is None:
        # ylim1 = ax.get_ylim()
        _, ylim1 = get_xylims_noerr(x, y, margin=0.05)
    assert_in_xylims(x, y, xlim=xlim1, ylim=ylim1)
    smol_, big_ = np.max([xlim1[0], ylim1[0]]), np.min([xlim1[1], ylim1[1]])
    ax.plot([smol_, big_], [smol_, big_], c="black", alpha=0.5, linestyle=":")  # y=x line.
    ax.plot([0, 0], ylim1, color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.plot(xlim1, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim1)

    ####### Make bias vs. mean subplot.
    ax = axes[1]
    x, xerr, y, yerr = crit_mean, crit_err, bias_m, bias_e
    m_ = (xerr <= err_max) & (yerr <= err_max)
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_no_fit(ax, x, y, xerr, yerr, text, text_idx, color_="grey", show_scatter=True, caps=caps, alpha_err=alpha_err)
    #### Miscellanious.
    ax.set_xlabel(f"{SENT2LAB[sent]} Mean")
    ax.set_ylabel(f"{SENT2LAB[sent]} Bias")
    if xlims[1] is None:
        # xlim2 = ax.get_xlim()
        xlim2, _ = get_xylims_noerr(x, y, margin=0.05)
    if ylims[1] is None:
        # ylim2 = ax.get_ylim()
        _, ylim2 = get_xylims_noerr(x, y, margin=0.05)
    assert_in_xylims(x, y, xlim=xlim2, ylim=ylim2)
    ax.plot(xlim2, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim2)
    ax.set_ylim(ylim2)

    for ax in axes:
        set_xticks(ax)
        set_yticks(ax)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} Mean-Bias {SENT2LAB[sent]}.svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def plot_overview_2_subplots(
    rmrs, grps, grps_subset, dir_, xlims=None, ylims=None, sent=-1, caps=False, alpha_err=0.33, use_SEM=False, err_max=np.inf, by=None
):
    # err_max: Maximum standard deviation allowed for points to be considered.
    if not np.isinf(err_max):
        print("DEBUG: err_max is not np.inf.")
    if xlims is not None:
        xlim1, xlim2 = xlims
    else:
        xlims = [None, None]
    if ylims is not None:
        ylim1, ylim2 = ylims
    else:
        ylims = [None, None]
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1, ddof=1) / denom_sem

    fig, axes = plt.subplots(1, 2, figsize=(3.41 * 2, 3.41 * 1))
    ax = axes[0]
    ####### Make non-collab vs. collab subplot.
    x, xerr, y, yerr = crit_cd1_m, crit_cd1_e, crit_cd2p_m, crit_cd2p_e
    m_ = (xerr <= err_max) & (yerr <= err_max)
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    ax.scatter(x, y, color="grey", marker=".", s=50, edgecolors="none", alpha=1, zorder=0)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle="None", marker=None, markersize=2.5, elinewidth=0.75, color="grey", alpha=alpha_err)
    styles_txt = dict(horizontalalignment="center", verticalalignment="bottom", xytext=(0, 2), textcoords="offset points")
    for idx, txt in zip(text_idx, text[text_idx]):
        txt = txt.capitalize() if caps else txt
        ax.annotate(txt, (x[idx], y[idx]), color="black", zorder=30, **styles_txt)
    #### Miscellanious.
    ax.set_xlabel(f"{SENT2LAB[sent]} to Collaborators")
    ax.set_ylabel(f"{SENT2LAB[sent]} to Non-Collaborators")
    if xlims[0] is None:
        # xlim1 = ax.get_xlim()
        xlim1, _ = get_xylims_noerr(x, y, margin=0.05)
    if ylims[0] is None:
        # ylim1 = ax.get_ylim()
        _, ylim1 = get_xylims_noerr(x, y, margin=0.05)
    assert_in_xylims(x, y, xlim=xlim1, ylim=ylim1)
    smol_, big_ = np.max([xlim1[0], ylim1[0]]), np.min([xlim1[1], ylim1[1]])
    ax.plot([smol_, big_], [smol_, big_], c="black", alpha=0.5, linestyle=":")  # y=x line.
    ax.plot([0, 0], ylim1, color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.plot(xlim1, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim1)

    ####### Make bias vs. mean subplot.
    ax = axes[1]
    x, xerr, y, yerr = crit_mean, crit_err, bias_m, bias_e
    m_ = (xerr <= err_max) & (yerr <= err_max)
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_and_fit(ax, x, y, yerr, text, text_idx, color_="grey", show_scatter=True, caps=caps, alpha_err=alpha_err, xerr=xerr)
    #### Miscellanious.
    ax.set_xlabel(f"{SENT2LAB[sent]} Mean")
    ax.set_ylabel(f"{SENT2LAB[sent]} Bias")
    if xlims[1] is None:
        # xlim2 = ax.get_xlim()
        xlim2, _ = get_xylims_noerr(x, y, margin=0.05)
    if ylims[1] is None:
        # ylim2 = ax.get_ylim()
        _, ylim2 = get_xylims_noerr(x, y, margin=0.05)
    assert_in_xylims(x, y, xlim=xlim2, ylim=ylim2)
    ax.plot(xlim2, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim2)
    ax.set_ylim(ylim2)

    for ax in axes:
        set_xticks(ax)
        set_yticks(ax)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} Mean-Bias {SENT2LAB[sent]}.svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def plot_effects_2_subplots(
    rmrs,
    grps,
    grps_subset,
    x_factor,
    x_lab,
    dir_,
    xlim=None,
    ylims=None,
    sent=-1,
    caps=False,
    alpha_err=0.33,
    use_SEM=False,
    err_max=np.inf,
    by=None,
    main_only=True,  # Towards all, meaning no collab vs. non-collab subplot.
    mean_only=True,  # Sentiment mean only, no sentiment bias.
):
    # err_max: Maximum standard deviation allowed for points to be considered.
    if not np.isinf(err_max):
        print("DEBUG: err_max is not np.inf.")
    if ylims is not None:
        ylim1, ylim2 = ylims  # ylim1 for mean plot, ylim2 for bias plot.
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
    # Sentiment bias.
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1, ddof=1) / denom_sem

    if mean_only:
        fig, ax = plt.subplots(1, 1, figsize=(3.41 * 1, 3.41 * 1))
    else:  # Include bias plot.
        fig, axes = plt.subplots(1, 2, figsize=(3.41 * 2, 3.41 * 1))
        ax = axes[0]
    x_all, y_all = np.array([]), np.array([])
    ####### Make mean subplot; collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd1_m, crit_cd1_e
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    if not main_only:
        plot_scatter_and_fit(
            ax, x, y, yerr, None, None, color_=COSTRA_COLORS[0], show_scatter=True, caps=caps, alpha_err=0.1, alpha_pt=0.2, xlim=xlim
        )
        x_all = np.concatenate((x_all, x))
        y_all = np.concatenate((y_all, y))
    ####### Make mean subplot; non-collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd2p_m, crit_cd2p_e
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    if not main_only:
        plot_scatter_and_fit(
            ax, x, y, yerr, None, None, color_=COSTRA_COLORS[1], show_scatter=True, caps=caps, alpha_err=0.1, alpha_pt=0.2, xlim=xlim
        )
        x_all = np.concatenate((x_all, x))
        y_all = np.concatenate((y_all, y))
    ####### Make mean subplot; both collab and non-collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_and_fit(
        ax, x, y, yerr, text, text_idx, color_=SENT_COLORS[SENT2IDX[sent]], show_scatter=True, caps=caps, alpha_err=alpha_err, xlim=xlim
    )
    x_all = np.concatenate((x_all, x))
    y_all = np.concatenate((y_all, y))
    #### Miscellanious.
    ax.set_xlabel(f"{x_lab}")
    ax.set_ylabel(f"{SENT2LAB[sent]} Mean")
    if xlim is None:
        xlim, _ = get_xylims_noerr(x_all, y_all, margin=0.05)
    if ylims is None:
        _, ylim1 = get_xylims_noerr(x_all, y_all, margin=0.05)
    assert_in_xylims(x_all, y_all, xlim=xlim, ylim=ylim1)
    ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim)
    ax.set_ylim(ylim1)

    ####### Make bias subplot.
    if not mean_only:
        ax = axes[1]
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), bias_m, bias_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        text_idx = np.in1d(text, grps_subset).nonzero()[0]
        plot_scatter_and_fit(
            ax, x, y, yerr, text, text_idx, color_=SENT_COLORS[SENT2IDX[sent]], show_scatter=True, caps=caps, alpha_err=alpha_err, xlim=xlim
        )
        #### Miscellanious.
        ax.set_xlabel(f"{x_lab}")
        ax.set_ylabel(f"{SENT2LAB[sent]} Bias")
        if xlim is None:
            xlim, _ = get_xylims_noerr(x, y, margin=0.05)
        if ylims is None:
            _, ylim2 = get_xylims_noerr(x, y, margin=0.05)
        assert_in_xylims(x, y, xlim=xlim, ylim=ylim2)
        ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
        ax.set_xlim(xlim)
        ax.set_ylim(ylim2)

    if mean_only:
        set_xticks(ax)
        set_yticks(ax)
    else:
        for ax in axes:
            set_xticks(ax)
            set_yticks(ax)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} {SENT2LAB[sent]} ({x_lab}).svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def plot_effects_2_lite_NEG_POS(
    rmrs,
    grps,
    grps_subset,
    x_factor,
    x_lab,
    dir_,
    xlim=None,
    ylims=None,
    caps=False,
    alpha_err=0.33,
    use_SEM=False,
    err_max=np.inf,
    by=None,
):
    # err_max: Maximum standard deviation allowed for points to be considered.
    if not np.isinf(err_max):
        print("DEBUG: err_max is not np.inf.")
    if ylims is None:
        ylims = [None, None]  # ylims[0] for sent=-1, ylims[1] for sent=1.
    fig, axes = plt.subplots(1, 2, figsize=(3.41 * 2, 3.41 * 1))
    for si, sent in enumerate([-1, 1]):
        denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
        # Sentiment across collab & non-collab.
        crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
        crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
        # Sentiment for collab.
        crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
        crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
        # Sentiment for non-collab.
        crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
        crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1, ddof=1) / denom_sem
        # Sentiment bias.
        bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
        bias_m = np.nanmean(bias, axis=-1)
        bias_e = np.nanstd(bias, axis=-1, ddof=1) / denom_sem

        ax = axes[si]
        x_all, y_all = np.array([]), np.array([])
        ####### Make mean subplot; both collab and non-collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        text_idx = np.in1d(text, grps_subset).nonzero()[0]
        x_all = np.concatenate((x_all, x))
        y_all = np.concatenate((y_all, y))
        if xlim is None:
            xlim, _ = get_xylims_noerr(x_all, y_all, margin=0.05)
        xlim_nonce = [xlim[0] + (xlim[1] - xlim[0]) * 0.02, xlim[1] - (xlim[1] - xlim[0]) * 0.02]
        plot_scatter_and_fit(
            ax,
            x,
            y,
            yerr,
            text,
            text_idx,
            color_=SENT_COLORS[SENT2IDX[sent]],
            show_scatter=True,
            caps=caps,
            alpha_err=alpha_err,
            xlim=xlim_nonce,
        )
        #### Miscellanious.
        ax.set_xlabel(f"{x_lab}")
        ax.set_ylabel(f"{SENT2LAB[sent]} Mean")
        if ylims[si] is None:
            _, ylim = get_xylims_noerr(x_all, y_all, margin=0.05)
        else:
            ylim = ylims[si]
        assert_in_xylims(x_all, y_all, xlim=xlim, ylim=ylim)
        ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for ax in axes:
        set_xticks(ax)
        set_yticks(ax)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} ({x_lab}).svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def plot_effects_2_lite_NEG_POS_simulation(
    rmrs,
    grps,
    grps_subset,
    x_factor,
    x_lab,
    dir_,
    xlim=None,
    ylims=None,
    caps=False,
    alpha_err=0.33,
    use_SEM=False,
    err_max=np.inf,
    by=None,
):
    # err_max: Maximum standard deviation allowed for points to be considered.
    if not np.isinf(err_max):
        print("DEBUG: err_max is not np.inf.")
    if ylims is None:
        ylims = [None, None]  # ylims[0] for sent=-1, ylims[1] for sent=1.
    fig, axes = plt.subplots(1, 2, figsize=(3.41 * 2, 3.41 * 1))
    sent2idx_sim = {1: 0, -1: 1}
    sent_colors_sim = ["#504DB2", "#B2504D"]  # POS, NEU, NEG
    for si, sent in enumerate([-1, 1]):
        denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
        # Sentiment across collab & non-collab.
        crit_mean = np.nanmean(rmrs[:, sent2idx_sim[sent], :], axis=-1)
        crit_err = np.nanstd(rmrs[:, sent2idx_sim[sent], :], axis=-1, ddof=1) / denom_sem

        ax = axes[si]
        x_all, y_all = np.array([]), np.array([])
        ####### Make mean subplot; both collab and non-collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        text_idx = np.in1d(text, grps_subset).nonzero()[0]
        x_all = np.concatenate((x_all, x))
        y_all = np.concatenate((y_all, y))
        if xlim is None:
            xlim, _ = get_xylims_noerr(x_all, y_all, margin=0.05)
        xlim_nonce = [xlim[0] + (xlim[1] - xlim[0]) * 0.02, xlim[1] - (xlim[1] - xlim[0]) * 0.02]
        plot_scatter_and_fit(
            ax,
            x,
            y,
            yerr,
            text,
            text_idx,
            color_=sent_colors_sim[sent2idx_sim[sent]],
            show_scatter=True,
            caps=caps,
            alpha_err=alpha_err,
            xlim=xlim_nonce,
        )
        #### Miscellanious.
        ax.set_xlabel(f"{x_lab}")
        ax.set_ylabel(f"{SENT2LAB[sent]} Mean")
        if ylims[si] is None:
            _, ylim = get_xylims_noerr(x_all, y_all, margin=0.05)
        else:
            ylim = ylims[si]
        assert_in_xylims(x_all, y_all, xlim=xlim, ylim=ylim)
        ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for ax in axes:
        set_xticks(ax)
        set_yticks(ax)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} ({x_lab}).svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def plot_covariates_corr(
    grpa2score, grpb2score, grpsa, grpsb, paper2grpsa, paper2grpsb, xy_labs, dir_, aggregate=True, use_SEM=False, alpha_err=0.33
):
    # If aggregate is False, each point is a paper (with unique grpa grpb combination);
    # If aggregate is True, each point is grpa's papers, with y-axis being those papers' grpb score mean and SEM. In other words, we aggregate grpb into mean and SEM per grpa.
    # ASSUME paper2grpsa and paper2grpsb share the same keys.
    # grpsa and grpsb are department and country, and the points are papers.
    # Turn grpsa and grpsb into sets since we only need to make sure it's the groups we considered in the main plots; orders don't matter.
    grpsa = set(grpsa)
    grpsb = set(grpsb)

    if aggregate:
        grpa2paper = reverse_dict_list(paper2grpsa)
        xyyerr = [
            (
                sa,
                *get_mean_with_spread(
                    np.array(flatten_list([[grpb2score[gb] for gb in paper2grpsb[p] if gb in grpsb] for p in grpa2paper[ga]])),
                    axis=None,
                    use_SEM=use_SEM,
                ),
            )
            for ga, sa in grpa2score.items()
        ]
        x = np.array([x for x, _, _ in xyyerr])
        y = np.array([y for _, y, _ in xyyerr])
        yerr = np.array([yerr for _, _, yerr in xyyerr])
    else:
        xy = [
            [
                (grpa2score[ga], grpb2score[gb])
                for ga, gb in itertools.product([t for t in paper2grpsa[p] if t in grpsa], [t for t in paper2grpsb[p] if t in grpsb])
            ]
            for p in paper2grpsa
        ]
        xy = flatten_list(xy)
        x = np.array([x for x, _ in xy])
        y = np.array([y for _, y in xy])
        yerr = None

    fig, ax = plt.subplots(1, 1, figsize=(3.41 * 1, 3.41 * 1))
    plot_scatter_and_fit(ax, x, y, yerr, None, None, color_="grey", show_scatter=True, alpha_err=alpha_err)
    ax.set_xlabel(f"{xy_labs[0]}")
    ax.set_ylabel(f"{xy_labs[1]}")
    xlim, _ = get_xylims_noerr(x, y, margin=0.05)
    _, ylim = get_xylims_noerr(x, y, margin=0.05)
    assert_in_xylims(x, y, xlim=xlim, ylim=ylim)
    ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    set_xticks(ax)
    set_yticks(ax)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{xy_labs}.svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def cosine_sim(vec1, vec2):
    if (type(vec1) == float and np.isnan(vec1)) or (type(vec2) == float and np.isnan(vec2)):
        return np.nan
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_collab_distance(g_coau, aui, auj, weight=None):
    try:
        return nx.shortest_path_length(g_coau, source=aui, target=auj, weight=weight)
    except nx.NetworkXNoPath:
        return np.inf
    except nx.NodeNotFound:
        raise Exception(f"Either {aui} or {auj} is not found in the coauthorship network.")
    except Exception:
        raise


def reverse_dict_list(A2B):
    # input a dict: key is A, val (B) is a list and returns a dict whose key is B, and val (A) is a list
    B2A = dict()
    for a in A2B:
        for b in A2B[a]:
            if b not in B2A:
                B2A[b] = [a]
            else:
                B2A[b].append(a)
    return B2A


def reverse_dict_val(A2B):
    # input a dict: key is A, val (B) is a single val and returns a dict whose key is B, and val (A) is a list
    B2A = dict()
    for a, b in A2B.items():
        if b not in B2A:
            B2A[b] = [a]
        else:
            B2A[b].append(a)
    return B2A


def sorted_dict(dict_, reverse=False, key=None):
    if key is None:
        key = lambda d: d[1]  # Sort by val.
    return dict(sorted(dict_.items(), key=key, reverse=reverse))


def flatten_list(ll):  # ll is list of list
    return [x for xx in ll for x in xx]


def has_keyword_any(str_, kws):
    for kw in kws:
        if kw in str_:
            return True
    return False


def get_IRR(row2rate_a, row2rate_b, labels=None):
    """IRR
    Args:
        row2rate_a: reference/true labels.
        labels (list): List of labels in row2rate_a. If None, default to [-1, 1, 0].
    """
    if labels is None:
        labels = [-1, 1, 0]
    rows = list(row2rate_a.keys())
    la = [row2rate_a[r] for r in rows]
    lb = [row2rate_b[r] for r in rows]
    # No meaningful spear or ck if there's only 1 label in la.
    spear = None if len(labels) == 1 else stats.spearmanr(la, lb)
    ck = None if len(labels) == 1 else cohen_kappa_score(la, lb)
    # average=None: F1 for each class is returned.
    f1 = f1_score(la, lb, labels=labels, average=None, sample_weight=None, zero_division="warn")

    agree_ratios = {x: [0, la.count(x)] for x in labels}
    for i in rows:
        if row2rate_a[i] == row2rate_b[i]:
            agree_ratios[row2rate_a[i]][0] += 1
    for k in agree_ratios.keys():
        agree_ratios[k].insert(0, agree_ratios[k][0] / agree_ratios[k][1])
    return {"spear": spear, "ck": ck, "F1": f1, "agree_ratios": agree_ratios}


def fuzz_check(text_1, text_2):

    r = fuzz.ratio(text_1, text_2)
    pr = fuzz.partial_ratio(text_1, text_2)

    return max(r, pr)
