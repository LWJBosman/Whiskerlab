import operator
import timeit

import numpy as np
import pandas as pd
import statistics
from bokeh.io import curdoc
from bokeh.layouts import row
from scipy import stats, optimize, random
from bokeh.plotting import figure, output_file, show, save
from bokeh.models.widgets import CheckboxGroup, Tabs
from bokeh.models import ColumnDataSource, WidgetBox, Panel, Whisker, Range1d, LinearAxis, LegendItem, \
    Legend, LinearColorMapper, ColorBar, BasicTicker
from bokeh.transform import transform
from scipy.stats import gaussian_kde, norm
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from Python import limbrecursionfuncs


class Fromfileplotter(object):
    def __init__(self):
        pass

    def plot_touchdurationdistribution(self, database, min_dur, max_dur, sample, sessionnr=None, subject_ids=None,
                                       durationlimit=None, touchlimit=None, durationlimit_lower=None,
                                       touchlimit_lower=None, walkside=None,
                                       rungpick=None, directionpick=None, trialselection=None, comment=None):

        data_all = []
        for t in trialselection:
            moves, _, _ = limbrecursionfuncs.getsteppatternfromfile(t)

            for m in moves:
                if m.limb == 1 or m.limb == 2:
                    touchduration = m[1].t_fly_start - m[1].t_start
                    data_all.append(touchduration)

        # histogram fitting
        bandwith = max_dur - min_dur
        hist_all, edges_all = np.histogram(data_all, density=True, bins=round(bandwith / 6))
        hist_relevant, edges_relevant = [], []
        for i in range(len(hist_all)):
            hist_relevant.append(hist_all[i])
            if len(edges_relevant) == 0 or not edges_relevant[-1] == edges_all[i]:
                edges_relevant.append(edges_all[i])  # add left edge if not already in there
            edges_relevant.append(edges_all[i + 1])  # add next right edge

        # plotting
        norm_params = stats.norm.fit(data_all)
        mu = norm_params[0]
        sigma = norm_params[1]
        median = statistics.median(data_all)
        # maxwell_params = stats.maxwell.fit(data_x)
        # gamma_params = stats.gamma.fit(data_x)
        # expon_params = stats.expon.fit(data_x)
        lognorm_params = stats.lognorm.fit(data_all, loc=0)
        x = np.linspace(0, max_dur, 1000)
        # other distribtions that aren't really relevant
        # pdf_normal = stats.norm.pdf(x, *norm_params)
        # pdf_maxwell = stats.maxwell.pdf(x, *maxwell_params)
        # pdf_gamma = stats.gamma.pdf(x, *gamma_params)
        # pdf_expon = stats.expon.pdf(x, *expon_params)
        # arbitrarily poked the scale and loc parameters to nicely fit data
        #

        # pdf_lognorm = stats.lognorm.pdf(x, lognorm_params[0] * v, lognorm_params[1] + v1,
        #                                 lognorm_params[2] + v2)
        p = figure(title="touchdurationdistribution | nr of touches: %i " % len(data_all),
                   x_axis_label='duration', y_axis_label=' frequency ',
                   x_range=(0, 600), y_range=(0, 0.02), plot_width=1000, plot_height=1000)
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        legend = Legend(items=[
            LegendItem(label="Nr of touches: %s " % len(data_all), index=2),
        ])
        p.add_layout(legend)

        # p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="black")
        # p.quad(bottom=0, top=hist_redu, left=edges_redu[:-1], right=edges_redu[1:], fill_color="green",
        #        line_color="black")
        # p.quad(bottom=0, top=hist_all, left=edges_all[:-1], right=edges_all[1:], fill_color="grey",
        #        line_color="black")
        p.quad(bottom=0, top=hist_relevant, left=edges_relevant[:-1], right=edges_relevant[1:],
               fill_color="blue",
               line_color="black")
        p.line([min_dur, min_dur], [0.01, -0.003], line_color="black", line_dash="dashed", legend="cutoff")
        p.line([max_dur, max_dur], [0.01, -0.003], line_color="black", line_dash="dashed", legend="cutoff")
        # p.line([median, median], [0, 1], line_color="red", legend="median = %i" % median)
        # p.line([mu + sigma, mu + sigma], [0, 0.015], line_color="yellow", legend="SD = %i" % sigma)
        # p.line([mu - sigma, mu - sigma], [0, 0.015], line_color="yellow")
        # p.line(x, pdf_normal, line_color="#D95B43", line_width=3, alpha=0.7, legend="PDF Normal")
        # p.line(x, pdf_maxwell, line_color="purple", line_width=3, alpha=0.7, legend="PDF Maxwell")
        # p.line(x, pdf_gamma, line_color="orange", line_width=3, alpha=0.7, legend="PDF Gamma")
        # p.line(x, pdf_expon, line_color="pink", line_width=3, alpha=0.7, legend="PDF Expon")
        # p.line(x, pdf_lognorm, line_color="cyan", line_width=3, alpha=0.7, legend="PDF Lognorm")
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "35px"
        output_file("ERCC1_%s_touchdurationdistribution_limbdetect_ses%s.html" % sessionnr, comment)
        show(p)

    def plot_frontstepdurationdistribution(self, database, min_dur, max_dur, sample,
                                           durationlimit=None, touchlimit=None, durationlimit_lower=None,
                                           touchlimit_lower=None,
                                           sessionnr=None, mouse_ids=None, directionpick=None, trialselection=None,
                                           comment=None):

        step_durations_all = []
        for t in trialselection:
            moves, _, _ = limbrecursionfuncs.getsteppatternfromfile(t)

            for m in moves:
                if m.limb == 1 or m.limb == 2:
                    touchduration = m[1].t_end - m[1].t_start
                    step_durations_all.append(touchduration)

        # plotting
        step_durations_all = list(filter(lambda x2: x2 < max_dur + 250, step_durations_all))
        gaussianmix = mixture.GaussianMixture(n_components=1, covariance_type='full')
        data_array = np.asarray(step_durations_all).reshape(-1, 1)
        mvg_params1 = gaussianmix.fit_predict(data_array)

        p = figure(title="Front step durations histogram | cutoffs: %i, %i " %
                         (min_dur, max_dur),
                   x_axis_label='time (ms)',
                   y_axis_label='p',
                   plot_height=800,
                   plot_width=800,
                   x_range=(0, 800), y_range=(0, 0.006))
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        legend = Legend(items=[
            LegendItem(label="Nr of steps: %s " % len(step_durations_all), index=0),
            LegendItem(label="Nr of trials: %s " % len(trialselection), index=1)
        ])
        p.add_layout(legend)
        p.legend.label_text_font_size = "30px"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        p.title.text_font_size = '20pt'

        hist, edges = np.histogram(step_durations_all, density=True, bins=80)
        median = statistics.median(step_durations_all)
        norm_params = stats.norm.fit(step_durations_all)
        mu = norm_params[0]
        lognorm_params = stats.lognorm.fit(step_durations_all, loc=0)  # loc (location parameter) shifts distribution
        # left/right, fixes problem of bad fit when initialized to 0, but is not proper value
        x = np.linspace(0, max_dur + 500, 1000)
        # arbitrarily poked the scale and loc parameters to nicely fit data
        pdf_lognorm = stats.lognorm.pdf(x, lognorm_params[0], lognorm_params[1] + 75, lognorm_params[2] - 90)
        p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="black")
        p.line(x, pdf_lognorm, line_color="cyan", line_width=3, alpha=0.7, legend="PDF Lognorm")
        # p.line([median, median], [0, 0.01], line_color="red", legend="median = %i" % median)
        # p.line([mu, mu], [0, 0.01], line_color="red", legend="mean = %i" % mu)
        p.legend.click_policy = "hide"
        output_file("ERCC1_%s_stepdurationdistribution_limbdetect_ses%s.html" % sessionnr, comment)
        show(p)

    def plot_frontsteplengthdistribution(self, database, min_dur, max_dur, subjectrange_db1,
                                         subjectrange_db1_2, subjectrange_db2=None, subjectrange_db2_2=None,
                                         sessionnr=None, durationlimit=None, touchlimit=None,
                                         control=None, mutant=None, directionpick=None, db2=None,
                                         trialselection_control=None, trialselection_mutant=None):
        p = figure(
            title="Front step length limbdetect | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
                  % (min_dur, max_dur,),  # len(step_lengths), sum(step_lengths) / len(step_lengths)),
            x_axis_label='length (rungs)',
            y_axis_label='occurence',
            y_range=(0, 1),
            x_range=(0.5, 8.5),
            plot_height=1000,
            plot_width=1000)
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        # legend = Legend(items=[
        #     LegendItem(label="Nr of steps: %s " % len(step_lengths), index=0),
        #     LegendItem(label="Nr of trials: %s " % len(trialselection), index=1)
        # ])
        # p.add_layout(legend)
        # p.legend.label_text_font_size = "25px"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        p.title.text_font_size = '15pt'

        for r in range(2):
            if r == 0:
                # CONTROL
                trialselection = trialselection_control
                movegraph = - 0.46
                color = 'blue'
                lgnd = 'control'
            if r == 1:
                # MUTANTS
                trialselection = trialselection_mutant
                movegraph = - 0.01
                color = 'red'
                lgnd = 'mutant'
            LF_step_lengths = [[] for _ in range(len(trialselection))]
            RF_step_lengths = [[] for _ in range(len(trialselection))]
            LH_step_lengths = [[] for _ in range(len(trialselection))]
            RH_step_lengths = [[] for _ in range(len(trialselection))]

            for t in trialselection:
                moves, _, _ = limbrecursionfuncs.getsteppatternfromfile(t)

                for m in moves:
                    if 8 >= m[1].rung_end - m[1].rung_start > 0:
                        if m[1].limb == 1:
                            touchduration = m[1].rung_end - m[1].rung_start
                            LF_step_lengths[trialselection.index(t)].append(touchduration)
                        if m[1].limb == 2:
                            touchduration = m[1].rung_end - m[1].rung_start
                            RF_step_lengths[trialselection.index(t)].append(touchduration)
                        if m[1].limb == 3:
                            touchduration = m[1].rung_end - m[1].rung_start
                            LH_step_lengths[trialselection.index(t)].append(touchduration)
                        if m[1].limb == 4:
                            touchduration = m[1].rung_end - m[1].rung_start
                            RH_step_lengths[trialselection.index(t)].append(touchduration)

            # plotting
            # for s, s1, s2, s3 in LF_step_lengths, RF_step_lengths, LH_step_lengths, RH_step_lengths:
            #     s = list(filter(lambda x2: x2 < 9, s))
            #     s1 = list(filter(lambda x2: x2 < 9, s1))
            #     s2 = list(filter(lambda x2: x2 < 9, s2))
            #     s3 = list(filter(lambda x2: x2 < 9, s3))

            LF_stepsize_averages = [[[] for _ in range(8)] for _ in range(len(trialselection))]
            RF_stepsize_averages = [[[] for _ in range(8)] for _ in range(len(trialselection))]
            LH_stepsize_averages = [[[] for _ in range(8)] for _ in range(len(trialselection))]
            RH_stepsize_averages = [[[] for _ in range(8)] for _ in range(len(trialselection))]
            for i in range(1, 9):
                for j in range(len(trialselection)):
                    if len(LF_step_lengths) > 0:
                        LF_stepsize_averages[j][i - 1] = LF_step_lengths[j].count(i) / len(LF_step_lengths[j])
                    else:
                        LF_stepsize_averages[j][i - 1] = 0
                    if len(RF_step_lengths) > 0:
                        RF_stepsize_averages[j][i - 1] = RF_step_lengths[j].count(i) / len(RF_step_lengths[j])
                    else:
                        RF_stepsize_averages[j][i - 1] = 0
                    if len(LH_step_lengths) > 0:
                        LH_stepsize_averages[j][i - 1] = LH_step_lengths[j].count(i) / len(LH_step_lengths[j])
                    else:
                        LH_stepsize_averages[j][i - 1] = 0
                    if len(RH_step_lengths) > 0:
                        RH_stepsize_averages[j][i - 1] = RH_step_lengths[j].count(i) / len(RH_step_lengths[j])
                    else:
                        RH_stepsize_averages[j][i - 1] = 0

            LF_standard_deviations = [[] for _ in range(8)]
            LF_means = [[] for _ in range(8)]
            LF_steplen_array = [[] for _ in range(len(LF_stepsize_averages))]
            RF_standard_deviations = [[] for _ in range(8)]
            RF_means = [[] for _ in range(8)]
            RF_steplen_array = [[] for _ in range(len(RF_stepsize_averages))]
            LH_standard_deviations = [[] for _ in range(8)]
            LH_means = [[] for _ in range(8)]
            LH_steplen_array = [[] for _ in range(len(LH_stepsize_averages))]
            RH_standard_deviations = [[] for _ in range(8)]
            RH_means = [[] for _ in range(8)]
            RH_steplen_array = [[] for _ in range(len(RH_stepsize_averages))]
            # for i in range(0, 8):
            #     for j in range(len(stepsize_averages)):
            #         steplen_array[j] = [stepsize_averages[j][i]]
            #     standard_deviations[i] = np.std(steplen_array)
            #     means[i] = np.mean(steplen_array)
            # for i in range(0, 8):
            #     counter = 0
            #     while counter < len(stepsize_averages):
            #         if abs(stepsize_averages[counter][i] - means[i]) > 3 * standard_deviations[i]:
            #             del stepsize_averages[counter]
            #             continue
            #         counter += 1
            # steplen_array = [[] for _ in range(len(stepsize_averages))]
            linewidth = 10
            for i in range(0, 8):
                for j in range(len(LF_stepsize_averages)):
                    LF_steplen_array[j] = LF_stepsize_averages[j][i]
                LF_standard_deviations[i] = np.std(LF_steplen_array)
                LF_means[i] = np.mean(LF_steplen_array)
                p.line([i + 1 + movegraph + 0.1, i + 1 + movegraph + 0.1], [0, LF_means[i]], alpha=1,
                       line_color=color, line_width=linewidth, legend=lgnd)
                p.circle([i + 1 + movegraph + 0.1], [0.95], alpha=1, legend='LF',
                         line_color='red', fill_color='red', size=10)
                p.line([i + 1 + movegraph + 0.1, i + 1 + movegraph + 0.1], [LF_means[i] - LF_standard_deviations[i],
                                                                            LF_means[i] + LF_standard_deviations[i]],
                       alpha=1,
                       line_color='black', line_width=3)
                for j in range(len(RF_stepsize_averages)):
                    RF_steplen_array[j] = RF_stepsize_averages[j][i]
                RF_standard_deviations[i] = np.std(RF_steplen_array)
                RF_means[i] = np.mean(RF_steplen_array)
                p.line([i + 1 + movegraph + 0.2, i + 1 + movegraph + 0.2], [0, RF_means[i]], alpha=1,
                       line_color=color, line_width=linewidth, legend=lgnd)
                p.circle([i + 1 + movegraph + 0.2], [0.95], alpha=1, legend='RF',
                         line_color='green', fill_color='green', size=10)
                p.line([i + 1 + movegraph + 0.2, i + 1 + movegraph + 0.2], [RF_means[i] - RF_standard_deviations[i],
                                                                            RF_means[i] + RF_standard_deviations[i]],
                       alpha=1,
                       line_color='black', line_width=3)
                for j in range(len(LH_stepsize_averages)):
                    LH_steplen_array[j] = LH_stepsize_averages[j][i]
                LH_standard_deviations[i] = np.std(LH_steplen_array)
                LH_means[i] = np.mean(LH_steplen_array)
                p.line([i + 1 + movegraph + 0.3, i + 1 + movegraph + 0.3], [0, LH_means[i]], alpha=1,
                       line_color=color, line_width=linewidth, legend=lgnd)
                p.circle([i + 1 + movegraph + 0.3], [0.95], alpha=1, legend='LH',
                         line_color='purple', fill_color='purple', size=10)
                p.line([i + 1 + movegraph + 0.3, i + 1 + movegraph + 0.3], [LH_means[i] - LH_standard_deviations[i],
                                                                            LH_means[i] + LH_standard_deviations[i]],
                       alpha=1,
                       line_color='black', line_width=3)
                for j in range(len(RH_stepsize_averages)):
                    RH_steplen_array[j] = RH_stepsize_averages[j][i]
                RH_standard_deviations[i] = np.std(RH_steplen_array)
                RH_means[i] = np.mean(RH_steplen_array)
                p.line([i + 1 + movegraph + 0.4, i + 1 + movegraph + 0.4], [0, RH_means[i]], alpha=1,
                       line_color=color, line_width=linewidth, legend=lgnd)
                p.circle([i + 1 + movegraph + 0.4], [0.95], alpha=1, legend='RH',
                         line_color='orange', fill_color='orange', size=10)
                p.line([i + 1 + movegraph + 0.4, i + 1 + movegraph + 0.4], [RH_means[i] - RH_standard_deviations[i],
                                                                            RH_means[i] + RH_standard_deviations[i]],
                       alpha=1,
                       line_color='black', line_width=3)
                for ii in LF_stepsize_averages:
                    p.circle([x + movegraph + 0.1 for x in range(1, 9)], ii,
                             line_color='black', fill_color=color, line_alpha=0.1, fill_alpha=0.1, size=5)
                for ii in RF_stepsize_averages:
                    p.circle([x + movegraph + 0.2 for x in range(1, 9)], ii,
                             line_color='black', fill_color=color, line_alpha=0.1, fill_alpha=0.1, size=5)
                for ii in LH_stepsize_averages:
                    p.circle([x + movegraph + 0.3 for x in range(1, 9)], ii,
                             line_color='black', fill_color=color, line_alpha=0.1, fill_alpha=0.1, size=5)
                for ii in RH_stepsize_averages:
                    p.circle([x + movegraph + 0.4 for x in range(1, 9)], ii,
                             line_color='black', fill_color=color, line_alpha=0.1, fill_alpha=0.1, size=5)

        p.legend.click_policy = "hide"
        output_file("ERCC1steplengthdistribution_limbdetect_ses%s.html" % sessionnr)
        show(p)

    def plotlowervsallsteps_v2(self, database, min_dur, max_dur, subjectrange_db1,
                               subjectrange_db1_2=None, subjectrange_db2=None, subjectrange_db2_2=None,
                               sessionnrs=None, durationlimit=None, touchlimit=None,
                               control=None, mutant=None, directionpick=None, db2=None, func=None):

        # p = figure(
        #     title="Lower steps over sessions, medians and IQR ranges | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
        #           % (min_dur, max_dur,),  # len(step_lengths), sum(step_lengths) / len(step_lengths)),
        #     x_axis_label='session',
        #     y_axis_label='percentage',
        #     y_range=(-1, 35),
        #     x_range=(0, 23),
        #     plot_height=1000,
        #     plot_width=1000)
        #

        p2 = figure(
            title="Lower steps over sessions, means and sems | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
                  % (min_dur, max_dur,),  # len(step_lengths), sum(step_lengths) / len(step_lengths)),
            x_axis_label='session',
            y_axis_label='percentage',
            y_range=(-1, 35),
            x_range=(0, 23),
            plot_height=1000,
            plot_width=1000)

        def processdata(moves, sessionnr):
            tolowerstep_lengths_thistrial = []
            allstep_lengths_thistrial = []

            for m in moves:
                m = m[1]
                step_length = m.rung_end - m.rung_start
                # TODO: below only correct for direction 1
                if (m.mouseside == 0 and m.rung_end % 2 == 0) or (m.mouseside == 1 and m.rung_end % 2 == 1):
                    # Step to lower rung
                    lowstep = True
                else:
                    # Step to higher rung
                    lowstep = False
                if lowstep:
                    tolowerstep_lengths[sessionnr - 1].append(step_length)
                    tolowerstep_lengths_thistrial.append(step_length)
                allstep_lengths[sessionnr - 1].append(step_length)
                allstep_lengths_thistrial.append(step_length)

            if len(allstep_lengths_thistrial) > 0:
                lowerpercentages[sessionnr - 1].append(
                    len(tolowerstep_lengths_thistrial) / len(allstep_lengths_thistrial) * 100)

        for r in range(2):
            tolowerstep_lengths = [[] for _ in sessionnrs]
            allstep_lengths = [[] for _ in sessionnrs]
            lowerpercentages = [[] for _ in sessionnrs]
            if r == 0:
                # CONTROL
                control = True
                mutant = False
                color = 'blue'
                lgnd = 'control'
            if r == 1:
                # MUTANTS
                control = False
                mutant = True
                color = 'red'
                lgnd = 'mutant'
            for ses in sessionnrs:
                trialselection = func(ses, control=control, mutant=mutant)

                for t_id in trialselection:
                    moves, _, _ = limbrecursionfuncs.getsteppatternfromfile(t_id)
                    processdata(moves, ses)

            means = [0 for _ in sessionnrs]
            medians = [0 for _ in sessionnrs]
            iqr3 = [0 for _ in sessionnrs]
            iqr1 = [0 for _ in sessionnrs]
            sd = [0 for _ in sessionnrs]
            sem = [0 for _ in sessionnrs]

            for i in range(len(sessionnrs)):
                df_percentages = pd.DataFrame(lowerpercentages[i], dtype='float')
                means[i] = df_percentages.mean()
                medians[i] = df_percentages.median()
                iqr3[i] = df_percentages.quantile(0.75)
                iqr1[i] = df_percentages.quantile(0.25)
                sd[i] = df_percentages.std()
                sem[i] = df_percentages.sem()
                # p.line(x=[i + 1, i + 1], y=[iqr3[i], iqr1[i]], line_color=color, legend=lgnd)
                # p.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[iqr3[i], iqr3[i]], line_color=color, legend=lgnd)
                # p.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[iqr1[i], iqr1[i]], line_color=color, legend=lgnd)
                p2.line(x=[i + 1, i + 1], y=[means[i] + sem[i], means[i] - sem[i]], line_color=color, legend=lgnd)
                p2.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[means[i] + sem[i], means[i] + sem[i]], line_color=color,
                        legend=lgnd)
                p2.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[means[i] - sem[i], means[i] - sem[i]], line_color=color,
                        legend=lgnd)
            #
            # p.line(x=sessionnrs, y=medians, line_color=color, legend=lgnd)
            p2.line(x=sessionnrs, y=means, line_color=color, legend=lgnd)

        # p.legend.click_policy = "hide"
        p2.legend.click_policy = "hide"
        output_file("ERCC1_means_sems_limbdetect.html")

        # show(p)
        show(p2)
        pass

    def plotstepdurationoversessions(self, database, min_dur, max_dur, subjectrange_db1,
                                     subjectrange_db1_2=None, subjectrange_db2=None, subjectrange_db2_2=None,
                                     sessionnrs=None, durationlimit=None, touchlimit=None,
                                     control=None, mutant=None, directionpick=None, db2=None, func=None):

        stepdurations = []

        # p = figure(
        #     title="Step durations over sessions, medians and IQRs | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
        #           % (min_dur, max_dur,),  # len(step_lengths), sum(step_lengths) / len(step_lengths)),
        #     x_axis_label='session',
        #     y_axis_label='percentage',
        #     y_range=(100, 350),
        #     x_range=(0, 20),
        #     plot_height=1000,
        #     plot_width=1000)
        #
        p2 = figure(
            title="Step durations over sessions, means and sems | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
                  % (min_dur, max_dur,),  # len(step_lengths), sum(step_lengths) / len(step_lengths)),
            x_axis_label='session',
            y_axis_label='percentage',
            y_range=(100, 350),
            x_range=(0, 20),
            plot_height=1000,
            plot_width=1000)

        def processdata(moves, sessionnr):
            for m in moves:
                m = m[1]
                duration = m.stepduration
                # TODO: below only correct for direction 1
                if (m.mouseside == 0 and m.rung_end % 2 == 0) or (m.mouseside == 1 and m.rung_end % 2 == 1):
                    # Step to lower rung
                    lowstep = True
                else:
                    # Step to higher rung
                    lowstep = False
                if duration > max_dur:
                    continue
                stepdurations[sessionnr - 1].append(duration)

        for r in range(2):
            stepdurations = [[] for _ in sessionnrs]
            if r == 0:
                # CONTROL
                control = True
                mutant = False
                color = 'blue'
                lgnd = 'control'
            if r == 1:
                # MUTANTS
                control = False
                mutant = True
                color = 'red'
                lgnd = 'mutant'
            for ses in sessionnrs:
                trialselection = func(ses, control=control, mutant=mutant)

                for t_id in trialselection:
                    moves, _, _ = limbrecursionfuncs.getsteppatternfromfile(t_id)
                    processdata(moves, ses)

            means = [0 for _ in sessionnrs]
            medians = [0 for _ in sessionnrs]
            iqr3 = [0 for _ in sessionnrs]
            iqr1 = [0 for _ in sessionnrs]
            sd = [0 for _ in sessionnrs]
            sem = [0 for _ in sessionnrs]

            for i in range(len(sessionnrs)):
                df_percentages = pd.DataFrame(stepdurations[i], dtype='float')
                means[i] = df_percentages.mean()
                medians[i] = df_percentages.median()
                iqr3[i] = df_percentages.quantile(0.75)
                iqr1[i] = df_percentages.quantile(0.25)
                sd[i] = df_percentages.std()
                sem[i] = df_percentages.sem()
                # p.line(x=[i + 1, i + 1], y=[iqr3[i], iqr1[i]], line_color=color, legend=lgnd)
                # p.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[iqr3[i], iqr3[i]], line_color=color, legend=lgnd)
                # p.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[iqr1[i], iqr1[i]], line_color=color, legend=lgnd)
                p2.line(x=[i + 1, i + 1], y=[means[i] + sem[i], means[i] - sem[i]], line_color=color, legend=lgnd)
                p2.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[means[i] + sem[i], means[i] + sem[i]], line_color=color,
                        legend=lgnd)
                p2.line(x=[i + 1 - 0.1, i + 1 + 0.1], y=[means[i] - sem[i], means[i] - sem[i]], line_color=color,
                        legend=lgnd)

            # p.line(x=sessionnrs, y=medians, line_color=color, legend=lgnd)
            p2.line(x=sessionnrs, y=means, line_color=color, legend=lgnd)
        # 
        # p.legend.click_policy = "hide"
        # output_file("ERCC1_plotstepdrationsaccross_MEDIAN_IQRs_sessions_limbdetect.html")
        p2.legend.click_policy = "hide"
        output_file("ERCC1_plotstepdrationsaccross_MEAN_SEMs_sessions_limbdetect.html")
        # 
        # show(p)
        show(p2)
        pass

    def plot_fronthindstepdurationdistribution(self, database, min_dur=None, max_dur=None, subjectrange=None,
                                               sessionnr=None, durationlimit=None, touchlimit=None,
                                               control=None, mutant=None, directionpick=None,
                                               trialselection_control=None, trialselection_mutant=None):

        p = figure(x_axis_label='duration', y_axis_label=' frequency ',
                   x_range=(0, 600), y_range=(0, 0.0075), plot_width=1000, plot_height=1000)
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.xaxis.major_label_text_font_size = "25pt"
        p.yaxis.major_label_text_font_size = "25pt"

        legend = Legend(items=[])
        p.add_layout(legend)

        for r in range(2):
            frontsteps = []
            hindsteps = []
            if r == 0:
                # CONTROL
                trialselection = trialselection_control
                color1 = 'blue'
                color2 = 'green'
                lgnd = 'control'
            if r == 1:
                # MUTANTS
                trialselection = trialselection_mutant
                color1 = 'red'
                color2 = 'purple'
                lgnd = 'mutant'
            for t in trialselection:
                moves, _, _ = limbrecursionfuncs.getsteppatternfromfile(t)

                for m in moves:
                    m = m[1]
                    stepduration = m.stepduration
                    if stepduration > max_dur or stepduration <= 0:
                        continue
                    if m.limb == 1 or m.limb == 2:
                        frontsteps.append(stepduration)
                    else:
                        hindsteps.append(stepduration)

            hist_fronts, edges_fronts = np.histogram(frontsteps, density=True, bins=60)
            hist_followups, edges_followups = np.histogram(hindsteps, density=True, bins=60)
            lognorm_params_fronts = stats.lognorm.fit(frontsteps, loc=0)
            x = np.linspace(0, 600, 1000)
            lognorm_params_followups = stats.lognorm.fit(hindsteps, loc=0)
            pdf_lognorm_fronts = stats.lognorm.pdf(x, lognorm_params_fronts[0], lognorm_params_fronts[1],
                                                   lognorm_params_fronts[2])
            pdf_lognorm_followups = stats.lognorm.pdf(x, lognorm_params_followups[0], lognorm_params_followups[1],
                                                      lognorm_params_followups[2])
            p.quad(bottom=0, top=hist_fronts, left=edges_fronts[:-1], right=edges_fronts[1:], fill_color=color1,
                   line_color=color1, fill_alpha=0.1, legend='Front step %s' % lgnd, line_alpha=0.1)
            p.quad(bottom=0, top=hist_followups, left=edges_followups[:-1], right=edges_followups[1:], fill_color=color2,
                   line_color=color2, fill_alpha=0.1, legend='Hind step %s' % lgnd, line_alpha=0.1)
            p.line(x, pdf_lognorm_fronts, line_color=color1, line_width=3, alpha=1, legend='Front step %s' % lgnd)
            p.line(x, pdf_lognorm_followups, line_color=color2, line_width=3, alpha=1, legend='Hind step %s' % lgnd)

        p.legend.click_policy = "hide"
        p.title.text = ("front/hindstepsanalysis | nr of touches: %i " % (len(frontsteps) + len(hindsteps)))
        output_file("ERCC1front_hindstepanalysis_%s_ses_%s.html" % (lgnd, sessionnr))
        show(p)
