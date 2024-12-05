import os
import numpy as np
import networkx as nx
from scipy import stats
import statsmodels.api as sm
from fuzzywuzzy import fuzz
import re
import pandas as pd  # For table printing.
import matplotlib as mpl
import matplotlib.pyplot as plt

SENT2LAB = {1: "Favorable Sentiment", 0: "Neutral Sentiment", -1: "Critical Sentiment"}
SENT2IDX = {1: 0, 0: 1, -1: 2}
SENT_COLORS = ["#504DB2", "#414042", "#B2504D"]  # POS, NEU, NEG
COSTRA_COLORS = ["#2CBEC6", "#F59448"]  # collaborators, non-collaborators


def fuzz_check(text_1, text_2):

    r = fuzz.ratio(text_1, text_2)
    pr = fuzz.partial_ratio(text_1, text_2)

    return max(r, pr)


def pval_print(pval):
    if pval >= 0.0001:
        return f"{pval:.3g}"
    else:
        return f"{pval:.3e}"


def pval_star(pval, star=False, show_insig=True, show_p=True):
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


def get_list_from_stat_dict(stat_dict):
    comp0 = stat_dict["coef_arr"][1, 0]  # Slope.
    comp0l = stat_dict["coef_arr"][1, 3]  # 95% confidence interval, lower.
    comp0u = stat_dict["coef_arr"][1, 4]  # 95% confidence interval, upper.
    comp1 = stat_dict["rsqr_a"]
    comp2 = f"F({stat_dict['fs'][0]},{stat_dict['fs'][1]})={stat_dict['f']:.2f}"
    comp3 = pval_star(stat_dict["fp"], star=False, show_insig=True, show_p=False)
    comp4 = stat_dict["spearman"]
    comp5 = pval_star(stat_dict["spearmanp"], star=False, show_insig=True, show_p=False)
    return [f"{comp0:.2f}", f"[{comp0l:.2f},{comp0u:.2f}]", f"{comp1:.3f}", comp2, comp3, f"{comp4:.2f}", comp5]


def print_stats(rmrs, grps, x_factor=None, x_lab=None, sent=-1, use_SEM=False, err_max=np.inf):
    data = []
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1) / denom_sem

    if x_factor is None:
        ####### bias vs. mean.
        x, xerr, y, yerr = crit_mean, crit_err, bias_m, bias_e
        m_ = (xerr <= err_max) & (yerr <= err_max)
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_stats(x, y, yerr)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Bias vs. Mean'] + get_list_from_stat_dict(stat_dict))

    if x_factor is not None:
        ####### mean, collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd1_m, crit_cd1_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_stats(x, y, yerr)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (Collab) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))
        ####### mean, non-collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd2p_m, crit_cd2p_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_stats(x, y, yerr)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (Non-Collab) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))
        ####### mean, both collab and non-collab.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_stats(x, y, yerr)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Mean (All) vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))

        ####### bias.
        x, xerr, y, yerr = x_factor, np.ones_like(x_factor), bias_m, bias_e
        m_ = yerr <= err_max
        x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
        stat_dict = calculate_stats(x, y, yerr)
        data.append([f'{SENT2LAB[sent].replace(" Sentiment", "")} Bias vs. {x_lab}'] + get_list_from_stat_dict(stat_dict))

    columns = ["Type", "Slope", "95% CI", "Adjusted R-squared", "F-test Statistic", "F-test p-value"]
    columns.extend(["Spearman Correlation", "Spearman p-value"])
    df = pd.DataFrame(data=data, columns=columns)
    # df.reset_index(drop=True, inplace=True)  # Remove index column.
    return df


def calculate_WLS_CI(X, y, yerr):
    slr_stats = calculate_stats(X, y, yerr, spearman=False)
    # Confidence interval (95%) for the (weighted) linear fit:
    n = len(y)
    s_err = np.sqrt(np.sum(slr_stats["resid"] ** 2) / (n - 2))  # standard deviation of the error (residuals)
    t = stats.t.ppf(0.975, n - 2)
    ci = t * s_err * np.sqrt(1 / n + (X - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2))

    return slr_stats, ci


def plot_scatter_and_fit(
    ax,
    x,
    y,
    yerr,
    text_arr=None,
    text_idx=None,
    color_="grey",
    show_scatter=True,
    show_stats=True,
    caps=False,
    alpha_err=0.33,
    xerr=None,
    alpha_pt=1,
):
    """

    Args:
        text_arr (np.arr): Text to annotate with; same length as x/y.
        text_idx (idx for x/y): Corresponding index for x/y to annotate; this can be shorter in length.
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
    # Weighted linear fit.
    sor_m = np.argsort(x)
    slr_stats, ci95 = calculate_WLS_CI(x[sor_m], y[sor_m], yerr[sor_m])
    ax.plot(x[sor_m], slr_stats["yhat"], color=color_, alpha=1)
    ax.fill_between(x[sor_m], slr_stats["yhat"] + ci95, slr_stats["yhat"] - ci95, color=color_, alpha=0.2, zorder=0, edgecolor=None)
    # Show stats on the plot.
    if show_stats:
        styles_txt = dict(fontsize=7, fontweight="normal", horizontalalignment="left", transform=ax.transAxes)
        if color_ == "grey":
            xcoor, ycoor = 0.05, 0.1
        else:
            xcoor, ycoor = 0.05 + 0.47 * int(color_ == "#F59448"), 0.95
        fval = slr_stats["f"]
        f_pval = slr_stats["fp"]
        slope = slr_stats["coef_arr"][1, 0]
        n_numer = np.round(slr_stats["df_model"]).astype(int)
        n_denom = np.round(slr_stats["df_resid"]).astype(int)
        txt_sig = rf"$F({n_numer},{n_denom})={fval:.2f}$  ${pval_star(f_pval)}$"
        txt_effect = rf"$s={slope:.2f}$   ${'R_a'}^2={slr_stats['rsqr_a']:.3f}$"
        ax.text(xcoor, ycoor, txt_sig, color=color_, **styles_txt)
        ax.text(xcoor, ycoor - 0.04, txt_effect, color=color_, **styles_txt)

    styles_txt = dict(horizontalalignment="center", verticalalignment="bottom", xytext=(0, 2), textcoords="offset points")
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

    styles_txt = dict(horizontalalignment="center", verticalalignment="bottom", xytext=(0, 2), textcoords="offset points")
    color_2 = "black" if color_ == "grey" else color_
    if text_idx is not None:
        for idx, txt in zip(text_idx, text_arr[text_idx]):
            txt = txt.capitalize() if caps else txt
            ax.annotate(txt, (x[idx], y[idx]), color=color_2, zorder=30, **styles_txt)


def assert_in_xylims(x, y, xlim=None, ylim=None):
    # Check if all the points (center, not accounting for error bars) are inside the limits
    # so that they are visible in the figures.
    if xlim is not None:
        for x_ in x:
            assert xlim[0] < x_ < xlim[1], f"xlim={xlim} doesn't cover all the points, missing x={x_}."

    if ylim is not None:
        for y_ in y:
            assert ylim[0] < y_ < ylim[1], f"ylim={ylim} doesn't cover all the points, missing y={y_}."


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
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1) / denom_sem

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
        xlim1 = ax.get_xlim()
    if ylims[0] is None:
        ylim1 = ax.get_ylim()
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
        xlim2 = ax.get_xlim()
    if ylims[1] is None:
        ylim2 = ax.get_ylim()
    assert_in_xylims(x, y, xlim=xlim2, ylim=ylim2)
    ax.plot(xlim2, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim2)
    ax.set_ylim(ylim2)

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
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1) / denom_sem

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
        xlim1 = ax.get_xlim()
    if ylims[0] is None:
        ylim1 = ax.get_ylim()
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
        xlim2 = ax.get_xlim()
    if ylims[1] is None:
        ylim2 = ax.get_ylim()
    assert_in_xylims(x, y, xlim=xlim2, ylim=ylim2)
    ax.plot(xlim2, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim2)
    ax.set_ylim(ylim2)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} Mean-Bias {SENT2LAB[sent]}.svg"), bbox_inches="tight", transparent=True)
    fig.clf()  # Clear figure.
    plt.close(fig=fig)  # Close figure.


def plot_effects_2_subplots(
    rmrs, grps, grps_subset, x_factor, x_lab, dir_, ylims=None, sent=-1, caps=False, alpha_err=0.33, use_SEM=False, err_max=np.inf, by=None
):
    # err_max: Maximum standard deviation allowed for points to be considered.
    if not np.isinf(err_max):
        print("DEBUG: err_max is not np.inf.")
    if ylims is not None:
        ylim1, ylim2 = ylims
    denom_sem = np.sqrt(rmrs.shape[-1]) if use_SEM else 1
    # Sentiment across collab & non-collab.
    crit_mean = np.nanmean(rmrs[0, :, SENT2IDX[sent], :], axis=-1)
    crit_err = np.nanstd(rmrs[0, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for collab.
    crit_cd1_m = np.nanmean(rmrs[1, :, SENT2IDX[sent], :], axis=-1)
    crit_cd1_e = np.nanstd(rmrs[1, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment for non-collab.
    crit_cd2p_m = np.nanmean(rmrs[2, :, SENT2IDX[sent], :], axis=-1)
    crit_cd2p_e = np.nanstd(rmrs[2, :, SENT2IDX[sent], :], axis=-1) / denom_sem
    # Sentiment bias.
    bias = rmrs[2, :, SENT2IDX[sent], :] - rmrs[1, :, SENT2IDX[sent], :]
    bias_m = np.nanmean(bias, axis=-1)
    bias_e = np.nanstd(bias, axis=-1) / denom_sem

    fig, axes = plt.subplots(1, 2, figsize=(3.41 * 2, 3.41 * 1))
    ax = axes[0]
    ####### Make mean subplot; collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd1_m, crit_cd1_e
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_and_fit(ax, x, y, yerr, None, None, color_=COSTRA_COLORS[0], show_scatter=True, caps=caps, alpha_err=0.1, alpha_pt=0.2)
    ####### Make mean subplot; non-collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_cd2p_m, crit_cd2p_e
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_and_fit(ax, x, y, yerr, None, None, color_=COSTRA_COLORS[1], show_scatter=True, caps=caps, alpha_err=0.1, alpha_pt=0.2)
    ####### Make mean subplot; both collab and non-collab.
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), crit_mean, crit_err
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_and_fit(ax, x, y, yerr, text, text_idx, color_="grey", show_scatter=True, caps=caps, alpha_err=alpha_err)
    #### Miscellanious.
    ax.set_xlabel(f"{x_lab}")
    ax.set_ylabel(f"{SENT2LAB[sent]} Mean")
    xlim = ax.get_xlim()
    if ylims is None:
        ylim1 = ax.get_ylim()
    assert_in_xylims(x, y, xlim=xlim, ylim=ylim1)
    ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim)
    ax.set_ylim(ylim1)

    ####### Make bias subplot.
    ax = axes[1]
    x, xerr, y, yerr = x_factor, np.ones_like(x_factor), bias_m, bias_e
    m_ = yerr <= err_max
    x, xerr, y, yerr, text = x[m_], xerr[m_], y[m_], yerr[m_], grps[m_]
    text_idx = np.in1d(text, grps_subset).nonzero()[0]
    plot_scatter_and_fit(ax, x, y, yerr, text, text_idx, color_="grey", show_scatter=True, caps=caps, alpha_err=alpha_err)
    #### Miscellanious.
    ax.set_xlabel(f"{x_lab}")
    ax.set_ylabel(f"{SENT2LAB[sent]} Bias")
    xlim = ax.get_xlim()
    if ylims is None:
        ylim2 = ax.get_ylim()
    assert_in_xylims(x, y, xlim=xlim, ylim=ylim2)
    ax.plot(xlim, [0, 0], color="grey", alpha=0.5, zorder=1, linestyle=":")  # Baseline (indistinguishable from null).
    ax.set_xlim(xlim)
    ax.set_ylim(ylim2)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(dir_, f"{by} {SENT2LAB[sent]} ({x_lab}).svg"), bbox_inches="tight", transparent=True)
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
