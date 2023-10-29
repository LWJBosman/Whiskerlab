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

plasma256 = [
    '#0C0786', '#100787', '#130689', '#15068A', '#18068B', '#1B068C', '#1D068D', '#1F058E', '#21058F', '#230590',
    '#250591', '#270592',
    '#290593', '#2B0594', '#2D0494', '#2F0495', '#310496', '#330497', '#340498', '#360498', '#380499', '#3A049A',
    '#3B039A', '#3D039B',
    '#3F039C', '#40039C', '#42039D', '#44039E', '#45039E', '#47029F', '#49029F', '#4A02A0', '#4C02A1', '#4E02A1',
    '#4F02A2', '#5101A2',
    '#5201A3', '#5401A3', '#5601A3', '#5701A4', '#5901A4', '#5A00A5', '#5C00A5', '#5E00A5', '#5F00A6', '#6100A6',
    '#6200A6', '#6400A7',
    '#6500A7', '#6700A7', '#6800A7', '#6A00A7', '#6C00A8', '#6D00A8', '#6F00A8', '#7000A8', '#7200A8', '#7300A8',
    '#7500A8', '#7601A8',
    '#7801A8', '#7901A8', '#7B02A8', '#7C02A7', '#7E03A7', '#7F03A7', '#8104A7', '#8204A7', '#8405A6', '#8506A6',
    '#8607A6', '#8807A5',
    '#8908A5', '#8B09A4', '#8C0AA4', '#8E0CA4', '#8F0DA3', '#900EA3', '#920FA2', '#9310A1', '#9511A1', '#9612A0',
    '#9713A0', '#99149F',
    '#9A159E', '#9B179E', '#9D189D', '#9E199C', '#9F1A9B', '#A01B9B', '#A21C9A', '#A31D99', '#A41E98', '#A51F97',
    '#A72197', '#A82296',
    '#A92395', '#AA2494', '#AC2593', '#AD2692', '#AE2791', '#AF2890', '#B02A8F', '#B12B8F', '#B22C8E', '#B42D8D',
    '#B52E8C', '#B62F8B',
    '#B7308A', '#B83289', '#B93388', '#BA3487', '#BB3586', '#BC3685', '#BD3784', '#BE3883', '#BF3982', '#C03B81',
    '#C13C80', '#C23D80',
    '#C33E7F', '#C43F7E', '#C5407D', '#C6417C', '#C7427B', '#C8447A', '#C94579', '#CA4678', '#CB4777', '#CC4876',
    '#CD4975', '#CE4A75',
    '#CF4B74', '#D04D73', '#D14E72', '#D14F71', '#D25070', '#D3516F', '#D4526E', '#D5536D', '#D6556D', '#D7566C',
    '#D7576B', '#D8586A',
    '#D95969', '#DA5A68', '#DB5B67', '#DC5D66', '#DC5E66', '#DD5F65', '#DE6064', '#DF6163', '#DF6262', '#E06461',
    '#E16560', '#E26660',
    '#E3675F', '#E3685E', '#E46A5D', '#E56B5C', '#E56C5B', '#E66D5A', '#E76E5A', '#E87059', '#E87158', '#E97257',
    '#EA7356', '#EA7455',
    '#EB7654', '#EC7754', '#EC7853', '#ED7952', '#ED7B51', '#EE7C50', '#EF7D4F', '#EF7E4E', '#F0804D', '#F0814D',
    '#F1824C', '#F2844B',
    '#F2854A', '#F38649', '#F38748', '#F48947', '#F48A47', '#F58B46', '#F58D45', '#F68E44', '#F68F43', '#F69142',
    '#F79241', '#F79341',
    '#F89540', '#F8963F', '#F8983E', '#F9993D', '#F99A3C', '#FA9C3B', '#FA9D3A', '#FA9F3A', '#FAA039', '#FBA238',
    '#FBA337', '#FBA436',
    '#FCA635', '#FCA735', '#FCA934', '#FCAA33', '#FCAC32', '#FCAD31', '#FDAF31', '#FDB030', '#FDB22F', '#FDB32E',
    '#FDB52D', '#FDB62D',
    '#FDB82C', '#FDB92B', '#FDBB2B', '#FDBC2A', '#FDBE29', '#FDC029', '#FDC128', '#FDC328', '#FDC427', '#FDC626',
    '#FCC726', '#FCC926',
    '#FCCB25', '#FCCC25', '#FCCE25', '#FBD024', '#FBD124', '#FBD324', '#FAD524', '#FAD624', '#FAD824', '#F9D924',
    '#F9DB24', '#F8DD24',
    '#F8DF24', '#F7E024', '#F7E225', '#F6E425', '#F6E525', '#F5E726', '#F5E926', '#F4EA26', '#F3EC26', '#F3EE26',
    '#F2F026', '#F2F126',
    '#F1F326', '#F0F525', '#F0F623', '#EFF821']
randcolor = [np.random.randint(len(plasma256)) for _ in range(37)]


def random_color():
    rgbl = list(np.random.choice(range(256), size=3))
    return tuple(rgbl)


def rewritesession(subjectids, session):
    # muscle
    newids = [0 for _ in subjectids]
    sessions = [0 for _ in subjectids]
    for s in subjectids:
        if s == 987:
            if session <= 5:
                newids[subjectids.index(s)] = 959
                sessions[subjectids.index(s)] = session - 0
            elif session <= 10:
                newids[subjectids.index(s)] = 965
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 987
                sessions[subjectids.index(s)] = session - 10
        if s == 988:
            if session <= 5:
                newids[subjectids.index(s)] = 960
                sessions[subjectids.index(s)] = session - 0
            elif session <= 10:
                newids[subjectids.index(s)] = 966
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 988
                sessions[subjectids.index(s)] = session - 10
        if s == 989:
            if session <= 5:
                newids[subjectids.index(s)] = 961
                sessions[subjectids.index(s)] = session - 0
            elif session <= 10:
                newids[subjectids.index(s)] = 967
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 989
                sessions[subjectids.index(s)] = session - 10
        if s == 990:
            if session <= 5:
                newids[subjectids.index(s)] = 962
                sessions[subjectids.index(s)] = session - 0
            elif session <= 10:
                newids[subjectids.index(s)] = 968
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 990
                sessions[subjectids.index(s)] = session - 10
        if s == 991:
            if session <= 5:
                newids[subjectids.index(s)] = 963
                sessions[subjectids.index(s)] = session - 0
            elif session <= 10:
                newids[subjectids.index(s)] = 969
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 991
                sessions[subjectids.index(s)] = session - 10
        if s == 992:
            if session <= 5:
                newids[subjectids.index(s)] = 964
                sessions[subjectids.index(s)] = session - 0
            elif session <= 10:
                newids[subjectids.index(s)] = 970
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 992
                sessions[subjectids.index(s)] = session - 10
        if s == 999:
            if session <= 5:
                newids[subjectids.index(s)] = 983
                sessions[subjectids.index(s)] = session - 0
            else:
                newids[subjectids.index(s)] = 999
                sessions[subjectids.index(s)] = session - 5
        if s == 1003:
            if session <= 5:
                newids[subjectids.index(s)] = 984
                sessions[subjectids.index(s)] = session - 0
            elif session <= 9:
                newids[subjectids.index(s)] = 1000
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 1003
                sessions[subjectids.index(s)] = session - 9
        if s == 1004:
            if session <= 5:
                newids[subjectids.index(s)] = 985
                sessions[subjectids.index(s)] = session - 0
            elif session <= 9:
                newids[subjectids.index(s)] = 1001
                sessions[subjectids.index(s)] = session - 5
            else:
                newids[subjectids.index(s)] = 1004
                sessions[subjectids.index(s)] = session - 9
        if s == 1002:
            if session <= 5:
                newids[subjectids.index(s)] = 986
                sessions[subjectids.index(s)] = session - 0
            else:
                newids[subjectids.index(s)] = 1002
                sessions[subjectids.index(s)] = session - 5
        if s == 1005:
            if session <= 5:
                newids[subjectids.index(s)] = 995
                sessions[subjectids.index(s)] = session - 0
            else:
                newids[subjectids.index(s)] = 1005
                sessions[subjectids.index(s)] = session - 5
        if s == 1006:
            if session <= 5:
                newids[subjectids.index(s)] = 996
                sessions[subjectids.index(s)] = session - 0
            else:
                newids[subjectids.index(s)] = 1006
                sessions[subjectids.index(s)] = session - 5
        if s == 1007:
            if session <= 5:
                newids[subjectids.index(s)] = 997
                sessions[subjectids.index(s)] = session - 0
            else:
                newids[subjectids.index(s)] = 1007
                sessions[subjectids.index(s)] = session - 5
        if s == 1008:
            if session <= 5:
                newids[subjectids.index(s)] = 998
                sessions[subjectids.index(s)] = session - 0
            else:
                newids[subjectids.index(s)] = 1008
                sessions[subjectids.index(s)] = session - 5
        if s == 1009:
            newids[subjectids.index(s)] = 1009
            sessions[subjectids.index(s)] = session + 5
        if s == 1010:
            newids[subjectids.index(s)] = 1010
            sessions[subjectids.index(s)] = session + 5
        if s == 1011:
            newids[subjectids.index(s)] = 1011
            sessions[subjectids.index(s)] = session + 5
        if s == 1012:
            newids[subjectids.index(s)] = 1012
            sessions[subjectids.index(s)] = session + 5
        if s == 1013:
            newids[subjectids.index(s)] = 1013
            sessions[subjectids.index(s)] = session + 5
        if s == 1014:
            newids[subjectids.index(s)] = 1014
            sessions[subjectids.index(s)] = session + 5
        if s == 1015:
            newids[subjectids.index(s)] = 1015
            sessions[subjectids.index(s)] = session + 5
        if s == 1016:
            newids[subjectids.index(s)] = 1016
            sessions[subjectids.index(s)] = session + 5
        if s == 1017:
            newids[subjectids.index(s)] = 1017
            sessions[subjectids.index(s)] = session + 5
        if s == 1018:
            newids[subjectids.index(s)] = 1018
            sessions[subjectids.index(s)] = session + 5
    # 959 -> 965 -> 987
    # 960 -> 966 -> 988
    # 961 -> 967 -> 989
    # 962 -> 968 -> 990
    # 963 -> 969 -> 991
    # 964 -> 970 -> 992
    #
    # 983 -> 999
    # 984 -> 1000 -> 1003
    # 985 -> 1001 -> 1004
    # 986 -> 1002
    #
    # 995 -> 1005
    # 996 -> 1006
    # 997 -> 1007
    # 998 -> 1008
    combo = [[0, 0, 0] for s in subjectids]
    for i in range(len(subjectids)):
        combo[i] = (subjectids[i], newids[i], sessions[i])
    return combo
    pass


class Plotter(object):
    def __init__(self):
        pass

    def heatmap(self, title='test', df=None, x_range=None, y_range=None, horizontal=None, vertical=None,
                colorlimit=None):
        global matrix
        if df is None:
            if y_range is None:
                y_range = [0, 50]
            if x_range is None:
                x_range = [0, 50]
            x = np.linspace(x_range[0], x_range[1], 50)
            y = np.linspace(y_range[0], y_range[1], 50)

            matrixrowsize = len(x) * len(y)
            matrix = np.empty([matrixrowsize, 3])
            # for i in range(height):
            #     matrix[(i*width):(i*width+width)][0] = x
            #     matrix[(i*width):(i*width+width)][1] = y[i]

            for m in range(len(y)):
                for n in range(len(x)):
                    matrix[m * len(x) + n][0] = x[n]
                    matrix[m * len(x) + n][1] = y[m]
                    matrix[m * len(x) + n][2] = np.random.randint(0, 1000)

            df = pd.DataFrame(matrix, columns=['x', 'y', 'value'])

        # matrix[2][0] = 8
        # print(matrix)
        width = (x_range[1] - x_range[0]) / horizontal
        height = (y_range[1] - y_range[0]) / vertical
        source = ColumnDataSource(data=df, )
        maxcolor = df.max(axis=0)
        maxcolor = maxcolor[2]
        mapper = LinearColorMapper(palette=plasma256, low=0, high=maxcolor)  # colorlimit/(horizontal*vertical/15))
        p = figure(
            title=title,
            x_axis_label='duration',
            y_axis_label='nr of touches')
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        # p.line([0, 1], [0, 1],  line_color="green", line_width=3, alpha=0.7)
        p.rect(x="x", y="y", width=width, height=height, source=source, line_color='black', line_alpha=0.3,
               fill_color=transform("value", mapper))
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0), ticker=BasicTicker(desired_num_ticks=10))
        p.add_layout(color_bar, 'right')

        return p

    def multimodalgaussianfitontouchdistribution(self, database, min_dur, max_dur, sample, sessionnr, subject_ids=None,
                                                 durationlimit=None, touchlimit=None, walkside=None,
                                                 rungpick=None, db2=None):
        control_errc1 = [189, 194, 195, 197, 199, 200, 202, 204]
        mutant_errc1 = [190, 191, 193, 196, 198, 201, 203, 205]
        if subject_ids:
            # switch trialselection in both querybuilders on if used
            trialselection = database.filterusefultrials(subject_ids, min_dur, max_dur, sessionnr,
                                                         durationlimit_upper=durationlimit,
                                                         touchlimit_upper=touchlimit)[0]
        trialselection2 = None
        if db2:
            trialselection2 = db2.filterusefultrials(subject_ids, min_dur, max_dur, sessionnr,
                                                     durationlimit_upper=durationlimit, touchlimit_upper=touchlimit)[0]

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]
        min_d = 0
        max_d = 0
        maximum = 750
        data_x = [[] for _ in range(3)]
        x = [[] for _ in range(3)]
        pdf_lognorm = [[] for _ in range(3)]

        p = figure(
            x_axis_label='duration', y_axis_label=' frequency ',
            # x_axis_type='log',
            x_range=(0, maximum), y_range=(0, 0.02),
            plot_width=1000, plot_height=1000)
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.extra_y_ranges = {"% of touches": Range1d(start=0, end=100)}
        p.add_layout(LinearAxis(y_range_name="% of touches"), 'right')

        for i in range(3):
            if i == 0:
                min_d = 0
                max_d = min_dur + 0
            if i == 1:
                min_d = min_dur - 0
                max_d = max_dur + 0
            if i == 2:
                min_d = max_dur - 0
                max_d = maximum
            query = database.query_builder(outputlist,
                                           jointables=['session', 'trial', 'touch'],
                                           touchlimits=[min_d, max_d],
                                           sessionpick=None,
                                           valuelimits1=('touch', 'trial_id', 156110, 170000),
                                           var_eq1=None,
                                           var_eq2=None,  # ['session', 'sessionnr', sessionnr, '='],
                                           vars_comp1=None,
                                           order1=None,
                                           walkside=walkside,
                                           subjectrange=None,
                                           rungpick=rungpick)
            data = database.query(query)
            data2 = None
            if trialselection2:
                query = db2.query_builder(outputlist,
                                          jointables=['session', 'trial', 'touch'],
                                          touchlimits=[min_d, max_d],
                                          sessionpick=None,
                                          valuelimits1=('touch', 'trial_id', 156110, 157000),
                                          var_eq1=None,
                                          var_eq2=None,  # ['session', 'sessionnr', sessionnr, '='],
                                          vars_comp1=None,
                                          order1=None,
                                          walkside=walkside,
                                          trialselection=None,
                                          rungpick=rungpick)
                data2 = db2.query(query)
            if data2:
                for d in data2:
                    data[data2.index(d)].extend(d)
            if not data:
                continue
            for d in range(len(data[0])):
                data_x[i].append(data[3][d] - data[2][d])

            lognorm_params = stats.lognorm.fit(data_x[i], loc=0)

            if i == 0:
                x[i] = np.linspace(0, maximum, 1000)
                pdf_lognorm[i] = stats.lognorm.pdf(x[i], lognorm_params[0], lognorm_params[1],
                                                   lognorm_params[2])
            if i == 1:
                x[i] = np.linspace(0, maximum, 1000)
                pdf_lognorm[i] = stats.lognorm.pdf(x[i], lognorm_params[0], lognorm_params[1],
                                                   lognorm_params[2])
            if i == 2:
                x[i] = np.linspace(0, maximum, 1000)
                pdf_lognorm[i] = stats.lognorm.pdf(x[i], lognorm_params[0] * 5, lognorm_params[1] - 250,
                                                   lognorm_params[2])

        bandwith = round(0.2 * (maximum))
        data_all = data_x[0] + data_x[1] + data_x[2]
        hist, edges = np.histogram(data_all, density=True, bins=bandwith)
        p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:], fill_color="grey",
               line_color="black", alpha=0.5)

        kde = gaussian_kde(data_all, 0.1)
        log_dens = kde.evaluate(x[0])
        gaussianmix = mixture.GaussianMixture(n_components=3, covariance_type='full')
        start = timeit.default_timer()
        data_all.sort()
        end = timeit.default_timer()
        print(end - start)
        data_array = np.asarray(data_all).reshape(-1, 1)
        mvg_params1 = gaussianmix.fit_predict(data_array)
        mvg_probs = gaussianmix.predict_proba(data_array)
        cl1 = []
        cl2 = []
        cl3 = []
        for tc, label in zip(data_all, mvg_params1):
            if label == 0:
                cl1.append(tc)
            if label == 1:
                cl2.append(tc)
            if label == 2:
                cl3.append(tc)
        print(len(cl1))
        print(len(cl2))
        print(len(cl3))
        lognorm_paramscl1 = stats.lognorm.fit(cl1, loc=0)
        lognorm_paramscl2 = stats.lognorm.fit(cl2, loc=0)
        lognorm_paramscl3 = stats.lognorm.fit(cl3, loc=0)
        norm_paramscl1 = stats.norm.fit(cl1, loc=0)
        norm_paramscl2 = stats.norm.fit(cl2, loc=0)
        norm_paramscl3 = stats.norm.fit(cl3, loc=0)
        pdf_normcl1 = stats.norm.pdf(x[0], norm_paramscl1[0], norm_paramscl1[1], )
        pdf_normcl2 = stats.norm.pdf(x[1], norm_paramscl2[0], norm_paramscl2[1], )
        pdf_normcl3 = stats.norm.pdf(x[2], norm_paramscl3[0], norm_paramscl3[1], )
        # p.line(data_all, [i[0] for i in mvg_probs], line_color="green", line_width=3, alpha=0.7, legend="p MMG cl1")
        # p.line(data_all, [i[1] for i in mvg_probs], line_color="yellow", line_width=3, alpha=0.7, legend="p MMG cl2")
        # p.line(data_all, [i[2] for i in mvg_probs], line_color="orange", line_width=3, alpha=0.7, legend="p MMG cl3")
        # p.line(x[0], pdf_normcl1, line_color="green", line_width=3, alpha=0.7, legend="PDF normal MMG cl1")
        # p.line(x[1], pdf_normcl2, line_color="yellow", line_width=3, alpha=0.7, legend="PDF normal MMG cl2")
        # p.line(x[2], pdf_normcl3, line_color="orange", line_width=3, alpha=0.7, legend="PDF normal MMG cl3")
        pdf_lognormcl1 = stats.lognorm.pdf(x[0], lognorm_paramscl1[0], lognorm_paramscl1[1],
                                           lognorm_paramscl1[2])
        pdf_lognormcl2 = stats.lognorm.pdf(x[1], lognorm_paramscl2[0], lognorm_paramscl2[1],
                                           lognorm_paramscl2[2])
        pdf_lognormcl3 = stats.lognorm.pdf(x[2], lognorm_paramscl3[0], lognorm_paramscl3[1],
                                           lognorm_paramscl3[2])
        mean_cl1 = np.mean(cl1)
        mean_cl2 = np.mean(cl2)
        mean_cl3 = np.mean(cl3)
        means = [mean_cl1, mean_cl2, mean_cl3]
        means_sorted = sorted(means)
        colors = [None, None, None]
        colors[means.index(means_sorted[0])] = 'green'
        colors[means.index(means_sorted[1])] = 'orange'
        colors[means.index(means_sorted[2])] = 'purple'
        if means_sorted[1] == mean_cl1:
            i = 0
            while i < len(cl1) and cl1[i] < means_sorted[0]:
                cl1.remove(cl1[i])
        if means_sorted[1] == mean_cl2:
            i = 0
            while i < len(cl2) and cl2[i] < means_sorted[0]:
                cl2.remove(cl2[i])
        if means_sorted[1] == mean_cl3:
            i = 0
            while i < len(cl3) and cl3[i] < means_sorted[0]:
                cl3.remove(cl3[i])

        p.line(x[0], pdf_lognormcl1, line_color=colors[0], line_width=3, alpha=0.7, legend="PDF Lognorm MMG cl1")
        p.line(x[1], pdf_lognormcl2, line_color=colors[1], line_width=3, alpha=0.7, legend="PDF Lognorm MMG cl2")
        p.line(x[2], pdf_lognormcl3, line_color=colors[2], line_width=3, alpha=0.7, legend="PDF Lognorm MMG cl3")
        p.line([min(cl1), max(cl1)], [0.011, 0.011], line_color=colors[0],
               line_width=3, alpha=0.7, legend="PDF Lognorm MMG cl1")
        p.line([min(cl2), max(cl2)], [0.0115, 0.0115], line_color=colors[1],
               line_width=3, alpha=0.7, legend="PDF Lognorm MMG cl2")
        p.line([min(cl3), max(cl3)], [0.0105, 0.0105], line_color=colors[2],
               line_width=3, alpha=0.7, legend="PDF Lognorm MMG cl3")

        hist, edges = np.histogram(data_all, density=False, bins=bandwith)
        cumulative_all = np.cumsum(hist) / len(data_all) * 100
        p.line(edges[:-1], cumulative_all, line_color='purple', line_dash='dashed',
               legend="total % of touches", y_range_name="% of touches")
        hist, edges = np.histogram(data_x[1] + data_x[2], density=False, bins=bandwith)
        cumulative_rel = np.cumsum(hist) / len(data_x[1] + data_x[2]) * 100
        # p.line(edges[:-1], cumulative_rel, line_color='purple',
        #        legend="%% of touches > %s ms" % (min_dur), y_range_name="% of touches")
        p.line(np.linspace(0, max_d, max_d), [95 for _ in range(maximum)], line_color='black', line_dash='dashed',
               y_range_name='% of touches')

        # p.line(x[0], log_dens, line_color="red", line_width=3, alpha=0.7, legend="KDE gaussian")

        # for i in range(3):
        #     p.line(x[i], pdf_lognorm[i], line_color="cyan", line_width=3, alpha=0.7, legend="PDF Lognorm")

        p.title.text = ("multimodal gaussian curve fit | nr of gaussians: %i " % len(data_x))
        p.xaxis.axis_label_text_font_size = "20pt"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "15px"
        p.xaxis.major_label_text_font_size = "20pt"
        p.yaxis.major_label_text_font_size = "10pt"
        show(p)

    def createcheckbox(self, labels, active):

        graph_checkboxgroup = CheckboxGroup(labels=labels,
                                            active=active)
        graph_checkboxgroup.on_change('active', self.updategraph)

        return graph_checkboxgroup

    def init_checkbox(self, checkbox_group, labeldictionary, figure):
        var_selection = [checkbox_group.label[i] for i in checkbox_group.active]

        return var_selection

    def updategraph(self, checkboxgroup, data_dic, source):
        # Get the list of lines/bins for the graph
        sessions_to_plot = [checkboxgroup.labels[i] for i in checkboxgroup.active]
        # Make a new dataset based on the selected carriers
        print("were here")
        data_dic = {}
        new_source = ColumnDataSource(data=data_dic)
        # Update the source used in the quad glpyhs
        source.data.update(new_source.data)

    def plot_trialdurationmean(self, database, min_dur, max_dur, subjectrange,
                               maxsessionnr=None, durationlimit=None, touchlimit=None,
                               control=None, mutant=None, directionpick=None, db2=None):
        p = figure(
            title="Trial duration distribution",
            x_axis_label='session',
            y_axis_label='average duration of 42 trials in a session(ms)',
            plot_width=1000,
            plot_height=700,
            x_range=(0.5, maxsessionnr + 0.5),
            y_range=(0, 6000))
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.extra_y_ranges = {"nroftouches": Range1d(start=0, end=13000)}
        p.add_layout(LinearAxis(y_range_name="nroftouches"), 'right')
        legend = Legend(items=[])
        p.add_layout(legend)
        p.legend.location = "top_left"
        p.xaxis.ticker = list(range(1, maxsessionnr + 1))
        p.xaxis.major_label_text_font_size = "20pt"
        p.yaxis.major_label_text_font_size = "10pt"

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]
        durations = [[] for _ in range(maxsessionnr)]
        touchesnrs = [0 for _ in range(maxsessionnr)]
        sessionrange = range(1, maxsessionnr + 1)
        max_ses = 0  # max_ses keeps track of the highest session number for plotting purposes
        usefultrials = []

        def processdata(trials, db, max_ses, nr):
            for t_id in trials:
                query = db.query_builder(
                    [('state', 'trial_id'), ('state', 'state_duration'), ('session', 'sessionnr')],
                    jointables=['session', 'trial', 'state'],
                    touchlimits=None,
                    sessionpick=None,
                    valuelimits1=None,
                    var_eq1=['state', 'state', 6, '='],
                    var_eq2=['state', 'trial_id', t_id, '='],
                    vars_comp1=None,
                    order1=None)

                data = db.query(query)
                for d in range(len(data[0])):
                    trial_id = data[0][d]
                    if not (d + 1) == len(data[0]) and data[0][
                        d + 1] == trial_id:  # check for next index if trial_id is same
                        continue
                    if not data[1][d] > 10000:
                        # remove absurd long runs (probably experimental fail)to keep logic trials means
                        print(data[2][d])
                        durations[nr - 1].append(data[1][d])
                        # add duration to list [n] for session number [n]
                    if nr > max_ses:
                        max_ses = nr

            return max_ses

        for nr in range(1, maxsessionnr + 1):
            # if nr > 11:               # for the ercc experiment
            #     durationlimit = 6000
            # if nr > 14:
            #     durationlimit = 7000
            # if nr > 17:
            #     durationlimit = 8000
            subjectsessioncombo = rewritesession(subjectrange, nr)
            usefultrials, dt = database.filterusefultrials(subjectrange, min_dur=min_dur, max_dur=max_dur,
                                                           sessionnr=nr,
                                                           durationlimit_upper=durationlimit,
                                                           touchlimit_upper=touchlimit,
                                                           control=None,
                                                           mutant=None, directionpick=directionpick,
                                                           customsubsescombo=subjectsessioncombo,
                                                           nofilter=True, noremoval=True)[0:2]
            # usefultrials2 = None
            # if db2:
            #     usefultrials2, dt = db2.filterusefultrials(subjectrange, min_dur=min_dur, max_dur=max_dur,
            #                                                sessionnr=nr,
            #                                                durationlimit=durationlimit, touchlimit=touchlimit,
            #                                                control=control,
            #                                                mutant=mutant, directionpick=directionpick)[0:2]

            query = database.query_builder(outputlist,
                                           jointables=['session', 'trial', 'touch'],
                                           touchlimits=[min_dur, max_dur],
                                           sessionpick=None,
                                           valuelimits1=('touch', 'trial_id', 0, 244641),
                                           var_eq1=None,
                                           var_eq2=None,  # ['session', 'sessionnr', sessionnr, '='],
                                           vars_comp1=None,
                                           order1=None,
                                           walkside=None,
                                           trialselection=usefultrials,
                                           rungpick=None)

            data = database.query(query)

            # query = db2.query_builder(outputlist,
            #                           jointables=['session', 'trial', 'touch'],
            #                           touchlimits=[min_dur, max_dur],
            #                           sessionpick=None,
            #                           valuelimits1=('touch', 'trial_id', 0, 244641),
            #                           var_eq1=None,
            #                           var_eq2=None,  # ['session', 'sessionnr', sessionnr, '='],
            #                           vars_comp1=None,
            #                           order1=None,
            #                           walkside=None,
            #                           trialselection=usefultrials2,
            #                           rungpick=None)
            #
            # data2 = db2.query(query)

            touchesnrs[nr - 1] = len(data[0])
            max_ses = processdata(usefultrials, database, max_ses, nr)

        for d in range(max_ses):
            df = pd.DataFrame(durations[d])
            p.quad(bottom=0, top=df.mean(), left=d + 1 - 0.25, right=d + 1 + 0.25,
                   fill_color="blue",
                   line_color="black",
                   alpha=1)
            p.line([d + 1, d + 1], [df.mean() - df.std(), df.mean() + df.std()], line_color='black', line_width=6)
        p.line(x=sessionrange, y=touchesnrs, line_color='red',
               legend="nr of touches", y_range_name='nroftouches')
        show(p)

    def plot_frontstepdurationvsrung(self, database, min_dur, max_dur, subjectrange,
                                     sessionnr=None, durationlimit=None, touchlimit=None,
                                     control=None, mutant=None):
        # This plot will show the step duration of a mouse, where the step is allocated to the rung where it is started.

        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"  # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"  # Balb/CBYJ females
        # customwhere = " session.subject_id in ('96') "  # ,'97','98','99','100','101')"  # C57Bl6 females

        trialselection = database.filterusefultrials(subjectrange, min_dur, max_dur, sessionnr,
                                                     durationlimit_upper=durationlimit, touchlimit_upper=touchlimit)[0]

        p = figure(
            x_axis_label='rung',
            y_axis_label='duration (ms)',
            x_range=(0, 38),
            y_range=(-200, 800),
            plot_width=1000,
            plot_height=700)
        # p.extra_y_ranges = {"runtime": Range1d(start=0, end=10000)}
        # p.add_layout(LinearAxis(y_range_name="runtime"), 'right')

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]

        # variables needed for calculating and plotting:
        step_durations = [[] for _ in range(37)]
        timestamps = [[] for _ in range(37)]
        timestamps_means = [0 for _ in range(37)]
        step_means = [0 for _ in range(37)]
        step_medians = [0 for _ in range(37)]
        step_sd_upper = [0 for _ in range(37)]
        step_sd_lower = [0 for _ in range(37)]

        for t_id in trialselection:
            query = database.query_builder(outputlist, jointables=['trial', 'touch'],
                                           touchlimits=[min_dur, max_dur],
                                           valuelimits1=None,
                                           range1=None,
                                           vars_comp1=None,
                                           vars_comp2=None,
                                           var_eq1=['trial', 'id', t_id, '='],
                                           order1=['touch', 'touch_begin', 'ASC'],
                                           customwhere=None,
                                           rungpick=None,
                                           directionpick=None)
            data = database.query(query)
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))
            farthest_rung_side_0 = 0  # to keep track of where the mouse should go next with its front paw to
            # discard hind paw touches
            farthest_rung_side_1 = 0
            start_step_0 = 0  # start time of the step
            start_step_1 = 0
            # for the runtime plot:
            farthest_rung = 0

            skipfirst_0 = True
            # in order to skip the first calculation of step duration, which is the step out of the box
            skipfirst_1 = True

            for d in range(
                    len(data[0])):  # loop calculating time between touches + touch time, thus forming a step time
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        step_duration = data[2][d] - start_step_0
                        start_step_0 = data[2][d]
                        if skipfirst_0:
                            skipfirst_0 = False
                            farthest_rung_side_0 = data[1][d]
                            continue
                        if 750 > step_duration > 50:
                            # ^filter function in other step duration histogram plot
                            # filter out abnormal long steps which are likely to be pauses
                            step_durations[farthest_rung_side_0 - 1].append(step_duration)
                            source = ColumnDataSource(
                                data={'rung': [farthest_rung_side_0],
                                      'step_duration': [step_durations[farthest_rung_side_0 - 1][-1]]})
                            p.circle(source=source, x='rung', y='step_duration',
                                     line_color='blue',  # plasma256[randcolor[farthest_rung_side_1]],
                                     fill_color='blue',  # plasma256[randcolor[farthest_rung_side_1]],
                                     alpha=0.4, size=5)
                        farthest_rung_side_0 = data[1][d]
                        if step_duration < 0:
                            print(step_duration, t_id, farthest_rung_side_0)  # this should not happen
                        # calculated step lengths that exceed a certain
                        # amount of time must be discarded  #
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        step_duration = data[2][d] - start_step_1
                        start_step_1 = data[2][d]
                        if skipfirst_1:
                            skipfirst_1 = False
                            farthest_rung_side_1 = data[1][d]
                            continue
                        if 750 > step_duration > 50:
                            # ^filter function in other step duration histogram plot
                            # filter out abnormal long steps which are likely to be pauses
                            step_durations[farthest_rung_side_1 - 1].append(step_duration)
                            source = ColumnDataSource(
                                data={'rung': [farthest_rung_side_1],
                                      'step_duration': [step_durations[farthest_rung_side_1 - 1][-1]]})
                            p.circle(source=source, x='rung', y='step_duration',
                                     line_color='blue',  # plasma256[randcolor[farthest_rung_side_1]],
                                     fill_color='blue',  # plasma256[randcolor[farthest_rung_side_1]],
                                     alpha=0.4, size=5)
                        farthest_rung_side_1 = data[1][d]
                        if step_duration < 0:
                            print(step_duration, t_id, farthest_rung_side_1)  # this should not happen
                if max(farthest_rung_side_0, farthest_rung_side_1) > farthest_rung:
                    farthest_rung = max(farthest_rung_side_0, farthest_rung_side_1)
                    timestamps[farthest_rung - 1].append(data[3][d])

        size = 0
        for i in range(37):
            df = pd.DataFrame(step_durations[i])
            size = size + len(step_durations[i])
            print(size)
            df1 = pd.DataFrame(timestamps[i])
            timestamps_means[i] = df1.mean()  # mean of the time the mouse reaches a rung
            step_means[i] = df.mean()
            step_medians[i] = df.median()
            step_sd_upper[i] = step_means[i] + df.std()
            step_sd_lower[i] = step_means[i] - df.std()

        data_dic = {'rungs': range(1, 38), 'step means': step_means,
                    'step medians': step_medians,
                    'step SD_upper': step_sd_upper,
                    'step SD_lower': step_sd_lower,
                    'timestamps_means': timestamps_means}
        source = ColumnDataSource(data=data_dic)
        p.line(source=source, x='rungs', y='step medians',
               line_color='orange',  # plasma256[randcolor[farthest_rung_side_1]],
               legend='median')

        # plotting:
        # colors = ["red", "olive", "darkred", "goldenrod", "skyblue", "orange", "salmon"]
        #
        # p.circle(source=source, x='rungs', y='step means_%i' % (j + 1), line_color=colors[j], alpha=0.7,
        #          legend="mean ses. %i" % sessionnrs[j])
        # p.line(source=source, x='rungs', y='timestamps_means_%i' % (j + 1), line_color=colors[j],
        #        legend="progression ses. %i" % sessionnrs[j], y_range_name='runtime')
        p.add_layout(
            Whisker(source=source, base='rungs', upper='step SD_upper',
                    lower='step SD_lower', line_alpha=1, line_width=2)
        )
        p.legend.click_policy = "hide"
        p.title.text = ("Average front step duration distribution over different rungs | cutoffs: %i, %i | nr of "
                        "steps: %i" %
                        (min_dur, max_dur, size))
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        p.title.text_font_size = '14pt'
        show(p)

    def plotdata(self, min_dur, max_dur):
        p = figure(
            title="Lower steps over sessions | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
                  % (min_dur, max_dur,),  # len(step_lengths), sum(step_lengths) / len(step_lengths)),
            x_axis_label='session nr',
            y_axis_label='percentage (%)',
            y_range=(0, 20),
            x_range=(0, 15),
            plot_height=1000,
            plot_width=1000)
        p.xaxis.axis_label_text_font_size = "22pt"
        p.yaxis.axis_label_text_font_size = "22pt"
        p.xaxis.major_label_text_font_size = "22pt"
        p.yaxis.major_label_text_font_size = "22pt"
        p.title.text_font_size = '0pt'

        # TODO: change below when different file should be processed
        df = pd.read_excel("muscledisease_trialselectionfilter_ADJUSTED.xlsx", index_col=0)

        subjects = np.array(df.index.values)
        columns = [[] for _ in range(len(df.iloc[0]))]
        columns_mut = [[] for _ in range(len(df.iloc[0]))]
        columns_wt = [[] for _ in range(len(df.iloc[0]))]
        means_mut = [0 for _ in range(len(df.iloc[0]))]
        means_wt = [0 for _ in range(len(df.iloc[0]))]
        medians_mut = [0 for _ in range(len(df.iloc[0]))]
        medians_wt = [0 for _ in range(len(df.iloc[0]))]
        iqr3_mut = [0 for _ in range(len(df.iloc[0]))]
        iqr3_wt = [0 for _ in range(len(df.iloc[0]))]
        iqr1_mut = [0 for _ in range(len(df.iloc[0]))]
        iqr1_wt = [0 for _ in range(len(df.iloc[0]))]
        sd_mut = [0 for _ in range(len(df.iloc[0]))]
        sd_wt = [0 for _ in range(len(df.iloc[0]))]
        sem_mut = [0 for _ in range(len(df.iloc[0]))]
        sem_wt = [0 for _ in range(len(df.iloc[0]))]

        for i in range(len(df.iloc[0])):
            columns[i] = df.iloc[:, i]
            for j in columns[i].iteritems():
                if j[0] in [989, 990, 991, 999, 1002, 1003, 1004, 1005, 1008, 1009, 1011, 1013]:
                    if j[1] > 0:
                        columns_mut[i].append(j[1])
                else:
                    if j[1] > 0:
                        columns_wt[i].append(j[1])
            df_mut = pd.DataFrame(columns_mut[i], dtype='float')
            df_wt = pd.DataFrame(columns_wt[i], dtype='float')
            means_mut[i] = df_mut.mean()
            means_wt[i] = df_wt.mean()
            medians_mut[i] = df_mut.median()
            medians_wt[i] = df_wt.median()
            iqr3_mut[i] = df_mut.quantile(0.75)
            iqr3_wt[i] = df_wt.quantile(0.75)
            iqr1_mut[i] = df_mut.quantile(0.25)
            iqr1_wt[i] = df_wt.quantile(0.25)
            sd_mut[i] = df_mut.std()
            sd_wt[i] = df_wt.std()
            sem_mut[i] = df_mut.sem()
            sem_wt[i] = df_wt.sem()

        for i in subjects:
            row = df.loc[i]
            row = row[row > 0]
            if i in [190, 191, 193, 196, 198, 201, 203, 205]:
                p.line(x=range(1, len(row) + 1), y=row, line_color='red', alpha=.1, legend='subject: %s' % i)
            else:
                p.line(x=range(1, len(row) + 1), y=row, line_color='blue', alpha=.1, legend='subject: %s' % i)
        # MEDIANS AND IQR RANGE:
        p.line(x=range(1, len(df.iloc[0]) + 1), y=medians_mut, line_color='red', alpha=1, legend='mut')
        p.line(x=range(1, len(df.iloc[0]) + 1), y=medians_wt, line_color='blue', alpha=1, legend='wt')
        for i in range(1, 13):
            p.line(x=[i, i], y=[iqr3_mut[i - 1], iqr1_mut[i - 1]], line_color='red', legend='mut')
            p.line(x=[i - 0.1, i + 0.1], y=[iqr3_mut[i - 1], iqr3_mut[i - 1]], line_color='red', legend='mut')
            p.line(x=[i - 0.1, i + 0.1], y=[iqr1_mut[i - 1], iqr1_mut[i - 1]], line_color='red', legend='mut')
            p.line(x=[i, i], y=[iqr3_wt[i - 1], iqr1_wt[i - 1]], line_color='blue', legend='wt')
            p.line(x=[i - 0.1, i + 0.1], y=[iqr3_wt[i - 1], iqr3_wt[i - 1]], line_color='blue', legend='wt')
            p.line(x=[i - 0.1, i + 0.1], y=[iqr1_wt[i - 1], iqr1_wt[i - 1]], line_color='blue', legend='wt')
        # MEANS AND SD
        # p.line(x=range(1, len(df.iloc[0]) + 1), y=means_mut, line_color='red', alpha=1, legend='mut')
        # p.line(x=range(1, len(df.iloc[0]) + 1), y=means_wt, line_color='blue', alpha=1, legend='wt')
        # for i in range(1, 13):
        #     p.line(x=[i, i], y=[means_mut[i - 1] + sd_mut[i - 1], means_mut[i - 1] - sd_mut[i - 1]],
        #            line_color='red', legend='mut')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_mut[i - 1] + sd_mut[i - 1], means_mut[i - 1] + sd_mut[i - 1]],
        #            line_color='red', legend='mut')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_mut[i - 1] - sd_mut[i - 1], means_mut[i - 1] - sd_mut[i - 1]],
        #            line_color='red', legend='mut')
        #     p.line(x=[i, i], y=[means_wt[i - 1] + sd_wt[i - 1], means_wt[i - 1] - sd_wt[i - 1]],
        #            line_color='blue', legend='wt')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_wt[i - 1] + sd_wt[i - 1], means_wt[i - 1] + sd_wt[i - 1]],
        #            line_color='blue', legend='wt')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_wt[i - 1] - sd_wt[i - 1], means_wt[i - 1] - sd_wt[i - 1]],
        #            line_color='blue', legend='wt')
        # MEANS AND SEM
        # for i in range(1, 13):
        #     p.line(x=[i, i], y=[means_mut[i - 1] + sem_mut[i - 1], means_mut[i - 1] - sem_mut[i - 1]],
        #            line_color='red', legend='mut')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_mut[i - 1] + sem_mut[i - 1], means_mut[i - 1] + sem_mut[i - 1]],
        #            line_color='red', legend='mut')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_mut[i - 1] - sem_mut[i - 1], means_mut[i - 1] - sem_mut[i - 1]],
        #            line_color='red', legend='mut')
        #     p.line(x=[i, i], y=[means_wt[i - 1] + sem_wt[i - 1], means_wt[i - 1] - sem_wt[i - 1]],
        #            line_color='blue', legend='wt')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_wt[i - 1] + sem_wt[i - 1], means_wt[i - 1] + sem_wt[i - 1]],
        #            line_color='blue', legend='wt')
        #     p.line(x=[i - 0.1, i + 0.1], y=[means_wt[i - 1] - sem_wt[i - 1], means_wt[i - 1] - sem_wt[i - 1]],
        #            line_color='blue', legend='wt')

        p.legend.click_policy = "hide"
        output_file("MUSCLEDisease_allsubjects_medians_IQRs.html")
        show(p)
        pass

    def plotlowervsallsteps(self, database, min_dur, max_dur, subjectrange_db1,
                            subjectrange_db1_2=None, subjectrange_db2=None, subjectrange_db2_2=None,
                            sessionnrs=None, durationlimit=None, touchlimit=None, durationlimit_lower=None,
                            touchlimit_lower=None,
                            control=None, mutant=None, directionpick=None, db2=None):
        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend'),
                      ('session', 'subject_id'),
                      ('session', 'sessionnr')]
        tolowerstep_lengths = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        tolowerstep_lengths_short = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        tolowerstep_lengths_long = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        allstep_lengths = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        allstep_lengths_short = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        allstep_lengths_long = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        lowerpercentages = [[[subject, 0] for subject in subjectrange_db1] for _ in sessionnrs]
        lowerpercentages_short_allsteps = [[[subject, 0] for subject in subjectrange_db1] for _ in sessionnrs]
        lowerpercentages_short_allshortsteps = [[[subject, 0] for subject in subjectrange_db1] for _ in sessionnrs]
        lowerpercentages_long_allsteps = [[[subject, 0] for subject in subjectrange_db1] for _ in sessionnrs]
        lowerpercentages_long_alllongsteps = [[[subject, 0] for subject in subjectrange_db1] for _ in sessionnrs]
        percentages = [[[subject, [0, 0, 0, 0, 0]] for subject in subjectrange_db1] for _ in sessionnrs]

        def processdata(data, sessionnr):
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))

            farthest_rung_side_0 = 0  # reset farthest rung to 0 every data query loop
            farthest_rung_side_1 = 0
            skipfirst_0 = True  # in order to skip the first calculation of step duration, which is the step out of
            # the box
            skipfirst_1 = True
            for d in range(len(data[0])):  # loop calculating length touches
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        previous_rung_side_0 = farthest_rung_side_0
                        farthest_rung_side_0 = data[1][d]
                        step_length = farthest_rung_side_0 - previous_rung_side_0
                        if data[1][d] % 2 == 0:
                            # Step to lower rung
                            lowstep = True
                        else:
                            # Step to higher rung
                            lowstep = False
                        if skipfirst_0:
                            skipfirst_0 = False
                            continue
                        if lowstep:
                            tolowerstep_lengths[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                            if step_length <= 2:
                                tolowerstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                    step_length)
                            else:
                                tolowerstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                    step_length)
                            pass
                        allstep_lengths[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(step_length)
                        if step_length <= 2:
                            allstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        else:
                            allstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_0)
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        previous_rung_side_1 = farthest_rung_side_1
                        farthest_rung_side_1 = data[1][d]
                        step_length = farthest_rung_side_1 - previous_rung_side_1
                        if data[1][d] % 2 == 1:
                            # Step to lower rung
                            lowstep = True
                        else:
                            # Step to higher rung
                            lowstep = False
                        if skipfirst_1:
                            skipfirst_1 = False
                            continue
                        if lowstep:
                            tolowerstep_lengths[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                            if step_length <= 2:
                                tolowerstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                    step_length)
                            else:
                                tolowerstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                    step_length)
                            pass
                        allstep_lengths[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(step_length)
                        if step_length <= 2:
                            allstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        else:
                            allstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_1)

        for ses in sessionnrs:
            trialselection = database.filterusefultrials(subjectrange_db1, min_dur, max_dur, ses,
                                                         durationlimit_upper=durationlimit, touchlimit_upper=touchlimit,
                                                         durationlimit_lower=durationlimit_lower,
                                                         touchlimit_lower=touchlimit_lower,
                                                         directionpick=directionpick, nofilter=False, noremoval=False)[
                0]

            for t_id in trialselection:
                query = database.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                               touchlimits=[min_dur, max_dur],
                                               valuelimits1=None,
                                               range1=None,
                                               vars_comp1=None,
                                               vars_comp2=None,
                                               var_eq1=['trial', 'id', t_id, '='],
                                               order1=['touch', 'touch_begin', 'ASC'],
                                               customwhere=None,
                                               rungpick=None)
                data = database.query(query)
                if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                    continue
                processdata(data, ses)

        for sub in subjectrange_db1:
            for ses in sessionnrs:
                if len(allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) == 0:
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][0] = 0
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][1] = 0
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][3] = 0
                else:
                    lowerpercentages[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][0] = \
                        lowerpercentages[ses - 1][subjectrange_db1.index(sub)][1]
                    lowerpercentages_short_allsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][1] = \
                        lowerpercentages_short_allsteps[ses - 1][subjectrange_db1.index(sub)][1]
                    lowerpercentages_long_allsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][3] = \
                        lowerpercentages_long_allsteps[ses - 1][subjectrange_db1.index(sub)][1]
                if len(allstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) == 0:
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][2] = 0
                else:
                    lowerpercentages_short_allshortsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][2] = \
                        lowerpercentages_short_allshortsteps[ses - 1][subjectrange_db1.index(sub)][1]
                if len(allstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) == 0:
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][4] = 0
                else:
                    lowerpercentages_long_alllongsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][4] = \
                        lowerpercentages_long_alllongsteps[ses - 1][subjectrange_db1.index(sub)][1]

        df = pd.DataFrame([[p[sub_idx][1][0] for p in percentages] for sub_idx in range(len(subjectrange_db1))],
                          columns=list(range(1, 20)), index=[p[0] for p in percentages[0]], )
        df.to_excel("ERCC1_trialselectionfilter.xlsx")

        pass

    def plottouchdurationoversessions(self, database, min_dur, max_dur, subjectrange_db1,
                                      subjectrange_db1_2=None, subjectrange_db2=None, subjectrange_db2_2=None,
                                      sessionnrs=None, durationlimit=None, touchlimit=None, durationlimit_lower=None,
                                      touchlimit_lower=None,
                                      control=None, mutant=None, directionpick=None, db2=None):
        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend'),
                      ('session', 'subject_id'),
                      ('session', 'sessionnr')]
        touchdurations = [[[subject, []] for subject in subjectrange_db1] for _ in sessionnrs]
        stepdurations = [[[subject, 0] for subject in subjectrange_db1] for _ in sessionnrs]

        def processdata(data, sessionnr):
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))

            farthest_rung_side_0 = 0  # reset farthest rung to 0 every data query loop
            farthest_rung_side_1 = 0
            skipfirst_0 = True  # in order to skip the first calculation of step duration, which is the step out of
            # the box
            skipfirst_1 = True
            for d in range(len(data[0])):  # loop calculating length touches
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        previous_rung_side_0 = farthest_rung_side_0
                        farthest_rung_side_0 = data[1][d]
                        touchduration = data[3][d] - data[2][d]
                        if data[1][d] % 2 == 0:
                            # Step to lower rung
                            lowstep = True
                        else:
                            # Step to higher rung
                            lowstep = False
                        if skipfirst_0:
                            skipfirst_0 = False
                            continue
                        if lowstep:
                            pass
                        [sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(step_length)
                        if step_length <= 2:
                            allstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        else:
                            allstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_0)
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        previous_rung_side_1 = farthest_rung_side_1
                        farthest_rung_side_1 = data[1][d]
                        step_length = farthest_rung_side_1 - previous_rung_side_1
                        if data[1][d] % 2 == 1:
                            # Step to lower rung
                            lowstep = True
                        else:
                            # Step to higher rung
                            lowstep = False
                        if skipfirst_1:
                            skipfirst_1 = False
                            continue
                        if lowstep:
                            tolowerstep_lengths[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                            if step_length <= 2:
                                tolowerstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                    step_length)
                            else:
                                tolowerstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                    step_length)
                            pass
                        allstep_lengths[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(step_length)
                        if step_length <= 2:
                            allstep_lengths_short[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        else:
                            allstep_lengths_long[sessionnr - 1][subjectrange_db1.index(data[5][d])][1].append(
                                step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_1)

        for ses in sessionnrs:
            trialselection = database.filterusefultrials(subjectrange_db1, min_dur, max_dur, ses,
                                                         durationlimit_upper=durationlimit, touchlimit_upper=touchlimit,
                                                         durationlimit_lower=durationlimit_lower,
                                                         touchlimit_lower=touchlimit_lower,
                                                         directionpick=directionpick, nofilter=False, noremoval=False)[
                0]

            for t_id in trialselection:
                query = database.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                               touchlimits=[min_dur, max_dur],
                                               valuelimits1=None,
                                               range1=None,
                                               vars_comp1=None,
                                               vars_comp2=None,
                                               var_eq1=['trial', 'id', t_id, '='],
                                               order1=['touch', 'touch_begin', 'ASC'],
                                               customwhere=None,
                                               rungpick=None)
                data = database.query(query)
                if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                    continue
                processdata(data, ses)

        for sub in subjectrange_db1:
            for ses in sessionnrs:
                if len(allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) == 0:
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][0] = 0
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][1] = 0
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][3] = 0
                else:
                    lowerpercentages[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][0] = \
                        lowerpercentages[ses - 1][subjectrange_db1.index(sub)][1]
                    lowerpercentages_short_allsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][1] = \
                        lowerpercentages_short_allsteps[ses - 1][subjectrange_db1.index(sub)][1]
                    lowerpercentages_long_allsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][3] = \
                        lowerpercentages_long_allsteps[ses - 1][subjectrange_db1.index(sub)][1]
                if len(allstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) == 0:
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][2] = 0
                else:
                    lowerpercentages_short_allshortsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths_short[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][2] = \
                        lowerpercentages_short_allshortsteps[ses - 1][subjectrange_db1.index(sub)][1]
                if len(allstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) == 0:
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][4] = 0
                else:
                    lowerpercentages_long_alllongsteps[ses - 1][subjectrange_db1.index(sub)][1] = \
                        len(tolowerstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) / len(
                            allstep_lengths_long[ses - 1][subjectrange_db1.index(sub)][1]) * 100
                    percentages[ses - 1][subjectrange_db1.index(sub)][1][4] = \
                        lowerpercentages_long_alllongsteps[ses - 1][subjectrange_db1.index(sub)][1]

        df = pd.DataFrame([[p[sub_idx][1][0] for p in percentages] for sub_idx in range(len(subjectrange_db1))],
                          columns=list(range(1, 20)), index=[p[0] for p in percentages[0]], )
        df.to_excel("ERCC1_trialselectionfilter_steplen.xlsx")

        pass

    def plotlowervsallsteps_v2(self, database, min_dur, max_dur, subjectrange_db1,
                               subjectrange_db1_2=None, subjectrange_db2=None, subjectrange_db2_2=None,
                               sessionnrs=None, durationlimit=None, touchlimit=None,
                               control=None, mutant=None, directionpick=None, db2=None, func=None):
        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend'),
                      ('session', 'subject_id'),
                      ('session', 'sessionnr')]

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

        def processdata(data, sessionnr):
            tolowerstep_lengths_thistrial = []
            allstep_lengths_thistrial = []
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))

            farthest_rung_side_0 = 0  # reset farthest rung to 0 every data query loop
            farthest_rung_side_1 = 0
            skipfirst_0 = True  # in order to skip the first calculation of step duration, which is the step out of
            # the box
            skipfirst_1 = True
            for d in range(len(data[0])):  # loop calculating length touches
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        previous_rung_side_0 = farthest_rung_side_0
                        farthest_rung_side_0 = data[1][d]
                        step_length = farthest_rung_side_0 - previous_rung_side_0
                        if data[1][d] % 2 == 0:
                            # Step to lower rung
                            lowstep = True
                        else:
                            # Step to higher rung
                            lowstep = False
                        if skipfirst_0:
                            skipfirst_0 = False
                            continue
                        if lowstep:
                            tolowerstep_lengths[sessionnr - 1].append(step_length)
                            tolowerstep_lengths_thistrial.append(step_length)
                        allstep_lengths[sessionnr - 1].append(step_length)
                        allstep_lengths_thistrial.append(step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_0)
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        previous_rung_side_1 = farthest_rung_side_1
                        farthest_rung_side_1 = data[1][d]
                        step_length = farthest_rung_side_1 - previous_rung_side_1
                        if data[1][d] % 2 == 1:
                            # Step to lower rung
                            lowstep = True
                        else:
                            # Step to higher rung
                            lowstep = False
                        if skipfirst_1:
                            skipfirst_1 = False
                            continue
                        if lowstep:
                            tolowerstep_lengths[sessionnr - 1].append(step_length)
                            tolowerstep_lengths_thistrial.append(step_length)
                        allstep_lengths[sessionnr - 1].append(step_length)
                        allstep_lengths_thistrial.append(step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_1)
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
                    query = database.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                                   touchlimits=[min_dur, max_dur],
                                                   valuelimits1=None,
                                                   range1=None,
                                                   vars_comp1=None,
                                                   vars_comp2=None,
                                                   var_eq1=['trial', 'id', t_id, '='],
                                                   order1=['touch', 'touch_begin', 'ASC'],
                                                   customwhere=None,
                                                   rungpick=None)
                    data = database.query(query)
                    if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                        continue
                    processdata(data, ses)

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

            # p.line(x=sessionnrs, y=medians, line_color=color, legend=lgnd)
            p2.line(x=sessionnrs, y=means, line_color=color, legend=lgnd)
        #
        # p.legend.click_policy = "hide"
        p2.legend.click_policy = "hide"
        output_file("ERCC1_means_sems.html")
        # show(p)
        show(p2)
        pass

    def plot_touchdurationvsrung(self, database, min_dur, max_dur, sample, sessionnrs=None, mouse_ids=None,
                                 durationlimit_upper=None, durationlimit_lower=None, touchlimit_upper=None,
                                 touchlimit_lower=None, directionpick=None, filename=None, rungpick=None,
                                 disregardsubjectlimit=None):
        # shows the touch duration against the different rungs to find out what distance the mouse needs before it
        # has reached its constant walk.

        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"  # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"  # Balb/CBYJ females
        # customwhere = "session.subject_id in ('96','97','98','99','100','101')"  # C57Bl6 females
        sessionnrs = sessionnrs

        # variables:
        rung_means = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_Q1s = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_Q3s = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_Q1s_list = [[[] for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_Q3s_list = [[[] for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_Q1s_median = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_Q3s_median = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_IQR = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_9quantile = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_91quantile = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_SDs = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        rung_medians = [[0 for _ in range(37)] for _ in range(len(sessionnrs))]
        data_dic = {'rungs': range(1, 38)}
        nroftouches = 0

        p = figure(title="Average touch duration distribution over all rungs | cutoffs: %i, %i " % (min_dur, max_dur),
                   x_axis_label='rung',
                   y_axis_label='duration (ms)',
                   x_range=(0, 38),
                   y_range=(0, 1400),
                   plot_width=500,
                   plot_height=1000)

        p.xaxis.axis_label_text_font_size = "18pt"
        p.yaxis.axis_label_text_font_size = "18pt"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        p.title.text_font_size = '0pt'

        # p1 = figure(x_axis_label='rung',
        #             y_axis_label='duration (ms)',
        #             x_range=(0, 38),
        #             y_range=(0, 450))
        #
        # p2 = figure(title="Touch duration distribution discontinuous | cutoffs: %i, %i " % (min_dur, max_dur),
        #             x_axis_label='rung',
        #             y_axis_label='duration (ms)',
        #             x_range=(0, 38),
        #             y_range=(0, 350))

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend'),
                      ('session', 'sessionnr')]

        loops = len(sessionnrs)
        for m in mouse_ids:
            for nr in sessionnrs:
                j = nr - 1
                usefultrials = database.filterusefultrials([m], min_dur, max_dur, nr,
                                                           durationlimit_upper=durationlimit_upper,
                                                           durationlimit_lower=durationlimit_lower,
                                                           touchlimit_upper=touchlimit_upper,
                                                           touchlimit_lower=touchlimit_lower,
                                                           directionpick=directionpick,
                                                           disregardsubjectlimit=disregardsubjectlimit)[0]
                if not usefultrials:
                    continue
                for i in range(1, 38):  # rungs
                    touch_durations = []
                    query = database.query_builder(outputlist, jointables=['subject', 'session', 'trial', 'touch'],
                                                   touchlimits=[min_dur, max_dur],
                                                   sessionpick=None,
                                                   subjectpick=None,
                                                   valuelimits1=None,
                                                   valuelimits2=None,
                                                   range1=None,
                                                   directionpick=directionpick,
                                                   vars_comp1=None,  # ['touch', 'side', 'touch', 'mouseside', '='],
                                                   vars_comp2=None,
                                                   rungpick=rungpick,
                                                   var_eq1=None,
                                                   var_eq2=['touch', 'rung', i, '='],
                                                   order1=['touch', 'touch_begin', 'ASC'],
                                                   trialselection=usefultrials)
                    data = database.query(query)
                    if data:
                        for d in range(len(data[0])):
                            if data[4][d] == 0:  # conversion for walk to start box
                                data[1][d] = (data[1][d] - 38) * -1
                                pass
                            touch_durations.append(data[3][d] - data[2][d])
                    # calculations:
                    if touch_durations:
                        df = pd.DataFrame(touch_durations, dtype='float')
                        nroftouches += len(touch_durations)
                        rung_means[j][i - 1] = df.mean()
                        rung_medians[j][i - 1] = df.median()
                        rung_Q1s[j][i - 1] = df.quantile(0.25)
                        rung_Q3s[j][i - 1] = df.quantile(0.75)
                        rung_Q1s_list[j][i - 1].append(rung_Q1s[j][i - 1])
                        rung_Q3s_list[j][i - 1].append(rung_Q3s[j][i - 1])

                        rung_IQR[j][i - 1] = rung_Q3s[j][i - 1] - rung_Q1s[j][i - 1]
                        rung_91quantile[j][i - 1] = df.quantile(0.91)
                        rung_9quantile[j][i - 1] = df.quantile(0.09)
                        rung_SDs[j][i - 1] = df.std()
                    elif i > 1:
                        rung_means[j][i - 1] = rung_means[j][i - 2]
                        rung_medians[j][i - 1] = rung_medians[j][i - 2]
                        rung_Q1s[j][i - 1] = rung_Q1s[j][i - 2]
                        rung_Q3s[j][i - 1] = rung_Q3s[j][i - 2]

                        rung_IQR[j][i - 1] = rung_Q3s[j][i - 2] - rung_Q1s[j][i - 2]
                        rung_91quantile[j][i - 1] = rung_91quantile[j][i - 2]
                        rung_9quantile[j][i - 1] = rung_9quantile[j][i - 2]
                        rung_SDs[j][i - 1] = rung_SDs[j][i - 2]
                    else:
                        rung_means[j][i - 1] = 150
                        rung_medians[j][i - 1] = 150
                        rung_Q1s[j][i - 1] = 250
                        rung_Q3s[j][i - 1] = 100

                        rung_IQR[j][i - 1] = rung_Q3s[j][i - 1] - rung_Q1s[j][i - 1]
                        rung_91quantile[j][i - 1] = 300
                        rung_9quantile[j][i - 1] = 50
                        rung_SDs[j][i - 1] = 75

                # z = np.polyfit(range(1, 38), rung_means[j], 5)
                # z = np.polyfit(range(1, 38), rung_medians[j], 5)
                # f = np.poly1d(np.concatenate(z).ravel())
                # x_val = np.linspace(1, 37, 100)
                # y_val = f(x_val)

                # plotting:
                data_dic["rung_means_%i" % j] = rung_means[j]
                data_dic["rung_medians%i" % j] = rung_medians[j]
                data_dic["rung_Q1s%i" % j] = rung_Q1s[j]
                data_dic["rung_Q3s%i" % j] = rung_Q3s[j]
                # data_dic["rung_IQR%i" % j] = rung_IQR[j]
                # data_dic["rung_91quantile%i" % j] = rung_91quantile[j]
                # data_dic["rung_9quantile%i" % j] = rung_9quantile[j]
                data_dic["rung_SDslow%i" % j] = list(map(operator.sub, rung_means[j], rung_SDs[j]))
                data_dic["rung_SDshigh%i" % j] = list(map(operator.add, rung_means[j], rung_SDs[j]))
                source = ColumnDataSource(data=data_dic)
                # print(source.data)
                colors = "blue", "green", "purple", "cyan", "violet", "pink", "darkgreen", "lightsalmon", "lightcyan",\
                         "turquoise", "limegreen", "orangered", "fuchsia", "firebrick", "lightsalmon", "slateblue",\
                         "mediumseagreen", "cadetblue", "brown", "midnightblue"
                # p.line(source=source, x='rungs', y='rung_means_%i' % j, line_color=colors[j], alpha=0.4,
                #        legend="mean ses. %i" % sessionnrs[j])
                # p.line(source=source, x='rungs', y='rung_medians%i' % j, line_color=colors[j + 1], alpha=0.4,
                #        legend="median ses. %i" % sessionnrs[j])
                # p.line(source=source, x='rungs', y='rung_Q1s%i' % j,
                # line_color=colors[j], alpha=0.4, line_dash="8 6", legend="Q1 ses. %i" % sessionnrs[j])
                # p.line(source=source, x='rungs', y='rung_Q3s%i' % j,
                # line_color=colors[j], alpha=0.4, line_dash="8 6", legend="Q3 session %i" % sessionnrs[j])
                # p.line(source=source, x='rungs', y='rung_SDslow%i' % j, line_color=colors[j + 2], alpha=0.4,
                #        line_dash="8 6", legend="SDs session %i" % sessionnrs[j])
                # p.line(source=source, x='rungs', y='rung_SDshigh%i' % j, line_color=colors[j + 2], alpha=0.4,
                #        line_dash="8 6", legend="SDs session %i" % sessionnrs[j])

                # p2.add_layout(
                #     Whisker(source=source, base='rungs', upper='rung_SDshigh%i' % j, lower='rung_SDslow%i' % j)
                # )
                # p2.circle(source=source, x='rungs', y='rung_means_%i' % j, line_color=colors[j], alpha=0.4,
                #           legend="mean ses. %i" % sessionnrs[j])
                # p2.circle(source=source, x='rungs', y='rung_medians%i' % j, line_color=colors[j + 1], alpha=0.4,
                #           legend="median ses. %i" % sessionnrs[j])

                # p.line(range(1, 38), rung_averages[j][:], line_color=colors[j], alpha=0.4, line_dash="8 6")
                # p.line(range(1, 38), rung_averages[j][:], line_color=colors[j], legend="average")
                # p.line(range(1, 38), rung_means[j][:], line_color=colors[j], legend="session %s" % sessionnrs[j],
                #        line_dash="4 4", alpha=0.5)
                # p.line(range(1, 38), rung_medians[j][:], line_color=colors[j + 4],
                #        legend="median session %s" % sessionnrs[j])
                # p.line(range(1, 38), rung_Q1s[j][:], line_color=colors[j + 3], legend="Q1")

                # TODO: the good shit
                # p.line(range(1, 38), list(map(operator.add, rung_means[j][:], rung_SDs[j][:])), line_color=colors[j],
                #        legend="session %s" % sessionnrs[j], alpha=0.05)
                # p.line(range(1, 38), list(map(operator.sub, rung_means[j][:], rung_SDs[j][:])), line_color=colors[j],
                #        legend="session %s" % sessionnrs[j], alpha=0.05)
                # p.varea(x=range(1, 38), y1=list(map(operator.add, rung_means[j][:], rung_SDs[j][:])),
                #         y2=list(map(operator.sub, rung_means[j][:], rung_SDs[j][:])), fill_color=colors[j],
                #         fill_alpha=0.0025, legend="session %s" % sessionnrs[j], )
                p.line(range(1, 38), rung_Q3s[j][:], line_color=colors[j],
                       legend="session %s" % sessionnrs[j], alpha=0.1)
                p.line(range(1, 38), rung_Q1s[j][:], line_color=colors[j],
                       legend="session %s" % sessionnrs[j], alpha=0.1)
                p.varea(x=range(1, 38), y1=rung_Q3s[j][:],
                        y2=rung_Q1s[j][:], fill_color=colors[j],
                        fill_alpha=0.002, legend="session %s" % sessionnrs[j], )

                # p.line(range(1, 38), rung_Q3s[j][:], line_color=colors[j + 3], legend="Q3")

                # p1.segment(range(1, 38), rung_9quantile[j][:], range(1, 38), rung_Q3s[j][:], line_color="black",
                #            legend="9th/91th quantile")
                # p1.segment(range(1, 38), rung_91quantile[j][:], range(1, 38), rung_Q1s[j][:], line_color="black")
                # p1.vbar(range(1, 38), 0.7, rung_medians[j][:], rung_Q3s[j][:], fill_color="#E08E79", line_color="black",
                #         legend="3rd quartile")
                # p1.vbar(range(1, 38), 0.7, rung_Q1s[j][:], rung_medians[j][:], fill_color="#3B8686", line_color="black",
                #         legend="1rd quartile")
                # p1.rect(range(1, 38), rung_91quantile[j][:], 0.2, 0.01, line_color="black")
                # p1.rect(range(1, 38), rung_9quantile[j][:], 0.2, 0.01, line_color="black")
                # p1.rect(range(1, 38), rung_means[j][:], 0.2, 0.8, line_color="black", fill_color="black", legend="mean")

                # p.circle(range(1, 38), rung_averages[j][:], line_color=colors[j], size=5, fill_color=colors[j],
                #              alpha=0.75)
                # p.line(x_val, y_val, line_color=colors[j], legend="fitted polynomials session %i" % sessionnrs[j],
                #        alpha=0.1)
                # p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="black")
        for nr in sessionnrs:
            j = nr - 1
            for i in range(1, 38):
                df_q1s = pd.DataFrame(rung_Q1s_list[j][i - 1], dtype='float')
                df_q3s = pd.DataFrame(rung_Q3s_list[j][i - 1], dtype='float')
                rung_Q1s_median[j][i - 1] = df_q1s.median()
                rung_Q3s_median[j][i - 1] = df_q3s.median()
            p.line(range(1, 38), rung_Q1s_median[j][:], line_color=colors[j],
                   legend="session %s" % sessionnrs[j], alpha=1)
            p.line(range(1, 38), rung_Q3s_median[j][:], line_color=colors[j],
                   legend="session %s" % sessionnrs[j], alpha=1)
            p.varea(x=range(1, 38), y1=rung_Q3s_median[j][:],
                    y2=rung_Q1s_median[j][:], fill_color='grey',
                    fill_alpha=0.2, legend="session %s" % sessionnrs[j], )

        labels = []
        for i in range(len(sessionnrs)):
            labels.append("session nr %i" % (i + 1))
        graph_checkboxgroup = CheckboxGroup(labels=labels,
                                            active=[0, 1, 2, 3])

        def update(attr, old, new):
            sessions_to_plot = [graph_checkboxgroup.labels[k] for k in graph_checkboxgroup.active]
            new_data_dic = {}
            print(sessions_to_plot)
            for k in range(len(sessions_to_plot)):
                if sessions_to_plot[i] == ("session nr %i" % (k + 1)):
                    new_data_dic = {"rungs": range(1, 38),
                                    "rung_means_%i" % k: data_dic["rung_means_%i" % k],
                                    "rung_medians%i" % k: data_dic["rung_medians%i" % k],
                                    "rung_Q1s%i" % k: data_dic["rung_Q1s%i" % k],
                                    "rung_Q3s%i" % k: data_dic["rung_Q3s%i" % k],
                                    "rung_SDslow%i" % k: data_dic["rung_SDslow%i" % k],
                                    "rung_SDshigh%i" % k: data_dic["rung_SDshigh%i" % k]}
                else:
                    new_data_dic = {"rungs": [0],
                                    "rung_means_%i" % k: [0],
                                    "rung_medians%i" % k: [0],
                                    "rung_Q1s%i" % k: [0],
                                    "rung_Q3s%i" % k: [0],
                                    "rung_SDslow%i" % k: [0],
                                    "rung_SDshigh%i" % k: [0]}
            new_source = ColumnDataSource(data=new_data_dic)

        # p1.title.text = "Touch duration distribution BOX PLOT | cutoffs: %i, %i | nr of touches: %s" \
        #                 % (min_dur, max_dur, nroftouches)
        # graph_checkboxgroup.on_change('active', update)
        legend = Legend()
        p.legend.click_policy = "hide"
        p.add_layout(legend)
        p.legend.label_text_font_size = "18px"
        # p1.legend.click_policy = "hide"
        # p2.legend.click_policy = "hide"
        # Put controls in a single element
        controls = WidgetBox(graph_checkboxgroup)
        # Create a row layout
        layout = row(controls, p)  # , p1, p2)
        # Make a tab with the layout
        tab = Panel(child=layout, title='www')
        tabs = Tabs(tabs=[tab])
        # curdoc().add_root(tabs)
        output_file("%s.html" % filename)
        show(layout)
        # show(checkbox)

    def plot_touchpattern(self, database, min_dur, max_dur, sample):
        # shows all touches and their duration of one trial, rung number vs time scale

        query = ("SELECT [side],[rung],[touch_begin], [touch_end] "
                 "FROM [touch] "
                 "WHERE trial_id = %i AND "
                 "(touch_end - touch_begin) > %i AND "
                 "(touch_end - touch_begin) < %i;" % (sample[-1], min_dur, max_dur))
        data = database.query(query)

        # plotting
        p = figure(title="walking pattern mouse | trial_id = %s | cutoffs: %i, %i | touches = %s "
                         % (sample[-1], min_dur, max_dur, len(data[0])),
                   x_axis_label='time (ms)',
                   y_axis_label='rung',
                   x_range=(0, 5000),
                   y_range=(0, 38),
                   plot_width=800)
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        for d in range(len(data[0])):
            if data[0][d] == 0:  # red side
                data_y = [data[1][d] - 0.1, data[1][d] - 0.1]
                data_x = [data[2][d], data[3][d]]
                if (data[1][d] % 2) == 0:
                    p.circle_cross(data_x[0], data_y[0], line_color="red", size=8)
                else:
                    p.circle(data_x[0], data_y[0], line_color="red", fill_color='red')
                p.line(data_x, data_y, line_color="red", legend="red side")
                if data[3][d] - data[2][d] > 250:
                    # possible double touch
                    p.circle(data[2][d] + 0.5 * (data[3][d] - data[2][d]), data_y[0], line_color="grey",
                             fill_color='grey')
            if data[0][d] == 1:  # green side
                data_y = [data[1][d], data[1][d]]
                data_x = [data[2][d], data[3][d]]
                if (data[1][d] % 2) == 1:
                    p.circle_cross(data_x[0], data_y[0], line_color="blue", size=8)
                else:
                    p.circle(data_x[0], data_y[0], line_color="blue", fill_color='blue')
                p.line(data_x, data_y, line_color="blue", legend="green side")
                if data[3][d] - data[2][d] > 250:
                    # possible double touch
                    p.circle(data[2][d] + 0.5 * (data[3][d] - data[2][d]), data_y[0], line_color="grey",
                             fill_color='grey')
        output_file("Trials\\trial %s.html" % sample[-1])
        save(p)
        show(p)

    def plot_frontsteppattern(self, database, min_dur, max_dur, sample):
        # This produces a graph like the plot_touchpattern pattern method, but here only increasing rung numbers
        # are considered to simulate only front paw, and lines are continued until a next rung on the same side is
        # activated, to simulate a full step. Note that the last steps into the box cannot be visualized as it is
        # unclear when they end.

        # variables needed for calculating and plotting:
        step_durations_0 = []  # for every side seperate records have to be kept
        step_durations_1 = []
        step_starts_0 = []
        step_starts_1 = []
        rung_seq_0 = []
        rung_seq_1 = []
        farthest_rung_side_0 = 0  # to keep track of where the mouse should go next with its front paw to discard
        # hind paw touches
        farthest_rung_side_1 = 0
        start_step_0 = 0  # start time of the step
        start_step_1 = 0

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]
        query = database.query_builder(outputlist,
                                       jointables=['session', 'trial', 'touch'],
                                       touchlimits=[min_dur, max_dur],
                                       sessionpick=None,
                                       valuelimits1=('touch', 'trial_id', 0, 244641),
                                       var_eq1=None,
                                       var_eq2=['trial', 'id', sample[-1], '='],
                                       vars_comp1=None,
                                       order1=None)
        data = database.query(query)
        if data[4][0] == 0:
            data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))
        for d in range(len(data[0])):  # loop calculating time between touches + touch time, thus forming a step time
            if data[0][d] == 0:
                if data[1][d] > farthest_rung_side_0:
                    farthest_rung_side_0 = data[1][d]
                    step_duration = data[2][d] - start_step_0
                    step_durations_0.append(step_duration)
                    rung_seq_0.append(data[1][d])
                    step_starts_0.append(data[2][d])
                    start_step_0 = data[2][d]
            if data[0][d] == 1:
                if data[1][d] > farthest_rung_side_1:
                    farthest_rung_side_1 = data[1][d]
                    step_duration = data[2][d] - start_step_1
                    step_durations_1.append(step_duration)
                    rung_seq_1.append(data[1][d])
                    step_starts_1.append(data[2][d])
                    start_step_1 = data[2][d]

        # plotting
        p = figure(title="Front step pattern | trial_id = %i | cutoffs: %i, %i " % (sample[-1], min_dur, max_dur),
                   x_axis_label='time (ms)',
                   y_axis_label='rung',
                   x_range=(0, 5000),
                   y_range=(0, 38),
                   plot_width=800)
        for i in range(0, len(rung_seq_0) - 1):
            if (rung_seq_0[i] % 2) == 0:
                p.circle_cross(step_starts_0[i], rung_seq_0[i], line_color="red", size=8)
            else:
                p.circle(step_starts_0[i], rung_seq_0[i], line_color="red")
            p.line([step_starts_0[i], step_starts_0[i] + step_durations_0[i + 1]], [rung_seq_0[i], rung_seq_0[i]],
                   # step durations array carries step duration at index i + 1 for every step indexed at i in
                   # step_starts and rung_sequence
                   line_color="red", legend="red side")
        for i in range(0, len(rung_seq_1) - 1):
            if (rung_seq_1[i] % 2) == 1:
                p.circle_cross(step_starts_1[i], rung_seq_1[i] - 0.1, line_color="blue", size=8)
            else:
                p.circle(step_starts_1[i], rung_seq_1[i] - 0.1, line_color="blue")
            p.line([step_starts_1[i], step_starts_1[i] + step_durations_1[i + 1]],
                   [rung_seq_1[i] - 0.1, rung_seq_1[i] - 0.1],
                   line_color="blue", legend="green side")
        show(p)

    def plot_frontsteplengthdistribution(self, database, min_dur, max_dur, subjectrange_db1,
                                         subjectrange_db1_2, subjectrange_db2=None, subjectrange_db2_2=None,
                                         sessionnr=None, durationlimit=None, touchlimit=None,
                                         control=None, mutant=None, directionpick=None, db2=None,
                                         trialselection=None):
        # This produces a graph with a distribution of the step lengths, which means the number of rungs that is
        # passed between consecutive touches on a certain side. Expected is a peak around 4.

        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"  # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"  # Balb/CBYJ females
        # customwhere = " session.subject_id in ('96','97','98','99','100','101')"  # C57Bl6 females
        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend'),
                      ('session', 'subject_id')]

        def processdata(data, customsubsescombo=None):
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))
            farthest_rung_side_0 = 0  # reset farthest rung to 0 every data query loop
            farthest_rung_side_1 = 0
            skipfirst_0 = True  # in order to skip the first calculation of step duration, which is the step out of
            # the box
            skipfirst_1 = True
            for d in range(len(data[0])):  # loop calculating length touches
                if customsubsescombo:
                    for combo in customsubsescombo:
                        if data[5][d] == combo[1]:
                            data[5][d] = combo[0]
                            break
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        previous_rung_side_0 = farthest_rung_side_0
                        farthest_rung_side_0 = data[1][d]
                        step_length = farthest_rung_side_0 - previous_rung_side_0
                        if skipfirst_0:
                            skipfirst_0 = False
                            continue
                        step_lengths[subjectrange.index(data[5][d])][1].append(step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_0)
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        previous_rung_side_1 = farthest_rung_side_1
                        farthest_rung_side_1 = data[1][d]
                        step_length = farthest_rung_side_1 - previous_rung_side_1
                        if skipfirst_1:
                            skipfirst_1 = False
                            continue
                        step_lengths[subjectrange.index(data[5][d])][1].append(step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_1)

        p = figure(
            title="Front step length | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
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
                if subjectrange_db2:
                    subjectrange = sorted(subjectrange_db1 + subjectrange_db2)
                else:
                    subjectrange = sorted(subjectrange_db1)
                movegraph = -0.1
                color = 'blue'
            if r == 1:
                # MUTANTS
                if subjectrange_db2_2:
                    subjectrange = sorted(subjectrange_db1_2 + subjectrange_db2_2)
                else:
                    subjectrange = sorted(subjectrange_db1_2)
                subjectrange_db1 = subjectrange_db1_2
                subjectrange_db2 = subjectrange_db2_2
                movegraph = 0.1
                color = 'red'
            step_lengths = [[subject, []] for subject in subjectrange]
            subjectsessioncombo = rewritesession(subjectrange_db1, sessionnr)
            if not trialselection:
                trialselection = database.filterusefultrials(subjectrange_db1, min_dur, max_dur, sessionnr,
                                                             durationlimit_upper=durationlimit,
                                                             touchlimit_upper=touchlimit,
                                                             directionpick=directionpick,
                                                             customsubsescombo=subjectsessioncombo, nofilter=True,
                                                             noremoval=True)[0]
            trialselection2 = None
            if db2:
                trialselection2 = db2.filterusefultrials(subjectrange_db2, min_dur, max_dur, sessionnr,
                                                         durationlimit_upper=durationlimit,
                                                         touchlimit_upper=touchlimit)[0]

            for t_id in trialselection:
                query = database.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                               touchlimits=[min_dur, max_dur],
                                               valuelimits1=None,
                                               range1=None,
                                               vars_comp1=None,
                                               vars_comp2=None,
                                               var_eq1=['trial', 'id', t_id, '='],
                                               order1=['touch', 'touch_begin', 'ASC'],
                                               customwhere=None,
                                               rungpick=None)
                data = database.query(query)
                if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                    continue
                processdata(data, customsubsescombo=subjectsessioncombo)
            if trialselection2:
                for t_id in trialselection2:
                    query = db2.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                              touchlimits=[min_dur, max_dur],
                                              valuelimits1=None,
                                              range1=None,
                                              vars_comp1=None,
                                              vars_comp2=None,
                                              var_eq1=['trial', 'id', t_id, '='],
                                              order1=['touch', 'touch_begin', 'ASC'],
                                              customwhere=None,
                                              rungpick=None)
                    data = db2.query(query)
                    if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                        continue
                    processdata(data)

            # plotting
            for s in step_lengths:
                s[1] = list(filter(lambda x2: x2 < 9, s[1]))

            i = 0
            while i < len(step_lengths):
                if len(step_lengths[i][1]) == 0:
                    del step_lengths[i]
                else:
                    i += 1
            stepsize_averages = [[[] for _ in range(8)] for _ in step_lengths]
            for i in range(1, 9):
                for j in range(len(step_lengths)):
                    if len(step_lengths[j][1]) > 0:
                        stepsize_averages[j][i - 1] = step_lengths[j][1].count(i) / len(step_lengths[j][1])
                    else:
                        stepsize_averages[j][i - 1] = 0
            concatenated = []
            for l in step_lengths:
                concatenated.extend(l[1])
            # hist, edges = np.histogram(concatenated, density=False, bins=50)
            # p.quad(bottom=0, top=hist / len(concatenated), left=edges[:-1], right=edges[1:], fill_color="blue",
            #        line_color="black")
            standard_deviations = [[] for _ in range(8)]
            means = [[] for _ in range(8)]
            steplen_array = [[] for _ in range(len(stepsize_averages))]
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
            for i in range(0, 8):
                for j in range(len(stepsize_averages)):
                    steplen_array[j] = stepsize_averages[j][i]
                standard_deviations[i] = np.std(steplen_array)
                means[i] = np.mean(steplen_array)
                p.line([i + 1 + movegraph, i + 1 + movegraph], [0, means[i]], alpha=1,
                       line_color=color, line_width=25)
                p.line([i + 1 + movegraph, i + 1 + movegraph], [means[i] - standard_deviations[i],
                                                                means[i] + standard_deviations[i]], alpha=1,
                       line_color='black', line_width=3)
            for i in stepsize_averages:
                p.circle([x + movegraph for x in range(1, 9)], i, alpha=1,
                         legend="mouse id %s" % step_lengths[stepsize_averages.index(i)][0],
                         line_color='black', fill_color=color, size=5)

        p.legend.click_policy = "hide"
        show(p)

    def plot_frontsteplengthdistribution_v2(self, database, min_dur, max_dur, subjectrange_db1,
                                            subjectrange_db1_2, subjectrange_db2=None, subjectrange_db2_2=None,
                                            sessionnr=None, durationlimit=None, touchlimit=None,
                                            durationlimit_lower=None, touchlimit_lower=None,
                                            control=None, mutant=None, directionpick=None, db2=None,
                                            trialselection_control=None, trialselection_mutant=None):
        # This produces a graph with a distribution of the step lengths, which means the number of rungs that is
        # passed between consecutive touches on a certain side. Expected is a peak around 4.

        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"  # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"  # Balb/CBYJ females
        # customwhere = " session.subject_id in ('96','97','98','99','100','101')"  # C57Bl6 females
        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend'),
                      ('trial', 'id')]

        def processdata(data, customsubsescombo=None):
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x: (x - 38) * -1, data[1]))
            farthest_rung_side_0 = 0  # reset farthest rung to 0 every data query loop
            farthest_rung_side_1 = 0
            skipfirst_0 = True  # in order to skip the first calculation of step duration, which is the step out of
            # the box
            skipfirst_1 = True
            for d in range(len(data[0])):  # loop calculating length touches
                if customsubsescombo:
                    for combo in customsubsescombo:
                        if data[5][d] == combo[1]:
                            data[5][d] = combo[0]
                            break
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        previous_rung_side_0 = farthest_rung_side_0
                        farthest_rung_side_0 = data[1][d]
                        step_length = farthest_rung_side_0 - previous_rung_side_0
                        if skipfirst_0:
                            skipfirst_0 = False
                            continue
                        step_lengths[trialselection.index(data[5][d])].append(step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_0)
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        previous_rung_side_1 = farthest_rung_side_1
                        farthest_rung_side_1 = data[1][d]
                        step_length = farthest_rung_side_1 - previous_rung_side_1
                        if skipfirst_1:
                            skipfirst_1 = False
                            continue
                        step_lengths[trialselection.index(data[5][d])].append(step_length)
                        if step_length < 0:
                            print(step_length, farthest_rung_side_1)

        p = figure(
            title="Front step length | cutoffs: %i, %i"  # " | nr of steps: %i | ",  # average step size: %4.3f"
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
                subjectrange = subjectrange_db1
                movegraph = -0.1
                color = 'blue'
                lgnd = 'control'
            if r == 1:
                # MUTANTS
                trialselection = trialselection_mutant
                subjectrange = subjectrange_db2
                movegraph = 0.1
                color = 'red'
                lgnd = 'mutant'
            if not trialselection:
                trialselection = database.filterusefultrials(subjectrange, min_dur, max_dur, sessionnr,
                                                             durationlimit_upper=durationlimit,
                                                             touchlimit_upper=touchlimit,
                                                             durationlimit_lower=durationlimit_lower,
                                                             touchlimit_lower=touchlimit_lower,
                                                             directionpick=directionpick,
                                                             customsubsescombo=None, nofilter=True,
                                                             noremoval=True)[0]
            step_lengths = [[] for _ in range(len(trialselection))]
            trialselection2 = None
            if db2:
                trialselection2 = db2.filterusefultrials(subjectrange_db2, min_dur, max_dur, sessionnr,
                                                         durationlimit_upper=durationlimit,
                                                         touchlimit_upper=touchlimit)[0]

            for t_id in trialselection:
                query = database.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                               touchlimits=[min_dur, max_dur],
                                               valuelimits1=None,
                                               range1=None,
                                               vars_comp1=None,
                                               vars_comp2=None,
                                               var_eq1=['trial', 'id', t_id, '='],
                                               order1=['touch', 'touch_begin', 'ASC'],
                                               customwhere=None,
                                               rungpick=None)
                data = database.query(query)
                if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                    continue
                processdata(data, customsubsescombo=None)
            if trialselection2:
                for t_id in trialselection2:
                    query = db2.query_builder(outputlist, jointables=['session', 'trial', 'touch'],
                                              touchlimits=[min_dur, max_dur],
                                              valuelimits1=None,
                                              range1=None,
                                              vars_comp1=None,
                                              vars_comp2=None,
                                              var_eq1=['trial', 'id', t_id, '='],
                                              order1=['touch', 'touch_begin', 'ASC'],
                                              customwhere=None,
                                              rungpick=None)
                    data = db2.query(query)
                    if not data:  # not all trial numbers are present, skip trial numbers that return empty queries
                        continue
                    processdata(data)

            # plotting
            for s in step_lengths:
                s = list(filter(lambda x2: x2 < 9, s))

            stepsize_averages = [[[] for _ in range(8)] for _ in range(len(trialselection))]
            for i in range(1, 9):
                for j in range(len(trialselection)):
                    if len(step_lengths[j]) > 0:
                        stepsize_averages[j][i - 1] = step_lengths[j].count(i) / len(step_lengths[j])
                    else:
                        stepsize_averages[j][i - 1] = 0
            concatenated = []
            for l in step_lengths:
                concatenated.extend(l)
            # hist, edges = np.histogram(concatenated, density=False, bins=50)
            # p.quad(bottom=0, top=hist / len(concatenated), left=edges[:-1], right=edges[1:], fill_color="blue",
            #        line_color="black")
            standard_deviations = [[] for _ in range(8)]
            means = [[] for _ in range(8)]
            steplen_array = [[] for _ in range(len(stepsize_averages))]
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
            for i in range(0, 8):
                for j in range(len(stepsize_averages)):
                    steplen_array[j] = stepsize_averages[j][i]
                standard_deviations[i] = np.std(steplen_array)
                means[i] = np.mean(steplen_array)
                p.line([i + 1 + movegraph, i + 1 + movegraph], [0, means[i]], alpha=1,
                       line_color=color, line_width=25, legend=lgnd)
                p.line([i + 1 + movegraph, i + 1 + movegraph], [means[i] - standard_deviations[i],
                                                                means[i] + standard_deviations[i]], alpha=1,
                       line_color='black', line_width=3)
            for i in stepsize_averages:
                p.circle([x + movegraph for x in range(1, 9)], i,
                         line_color='black', fill_color=color, line_alpha=0.1, fill_alpha=0.1, size=5)

        p.legend.click_policy = "hide"
        output_file("ERCC1steplengthdistribution_ses%s.html" % sessionnr)
        show(p)

    def plot_frontstepdurationdistribution(self, database, min_dur, max_dur, sample,
                                           durationlimit=None, touchlimit=None, durationlimit_lower=None,
                                           touchlimit_lower=None, trialselection=None, comment=None,
                                           sessionnr=None, mouse_ids=None, directionpick=None):
        # This produces a graph like plot_touchdurationdistribution, but instead of only sensor touch duration the
        # full step length is considered. A step is seen as the duration between two rung activations on the same
        # side, thus the time a mouse takes to put his paw on a rung, lift and move it to the next rung. Only front
        # paws are considered, this is done by only looking at increasing rung number.

        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"  # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"  # Balb/CBYJ females
        # customwhere = "session.subject_id in ('96','97','98','99','100','101')"  # C57Bl6 females
        start_step_0 = 0
        start_step_1 = 0
        step_durations_all = []

        # subjectsessioncombo = rewritesession(mouse_ids, sessionnr)
        if not trialselection:
            trialselection = database.filterusefultrials(mouse_ids, min_dur, max_dur, sessionnr,
                                                         durationlimit_upper=durationlimit, touchlimit_upper=touchlimit,
                                                         durationlimit_lower=durationlimit_lower,
                                                         touchlimit_lower=touchlimit_lower,
                                                         directionpick=directionpick, noremoval=False, nofilter=False,
                                                         customsubsescombo=None)[0]

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]

        for t_id in trialselection:
            query = database.query_builder(outputlist, jointables=['trial', 'touch'],
                                           touchlimits=[min_dur, max_dur],
                                           valuelimits1=None,
                                           range1=None,
                                           vars_comp1=None,
                                           vars_comp2=None,
                                           var_eq1=['trial', 'id', t_id, '='],
                                           order1=['touch', 'touch_begin', 'ASC'],
                                           directionpick=None,
                                           customwhere=None)
            data = database.query(query)
            if data[4][0] == 0:
                data[1] = tuple(map(lambda x1: (x1 - 38) * -1, data[1]))

            farthest_rung_side_0 = 0  # reset farthest rung to 0 every data query loop
            farthest_rung_side_1 = 0
            skipfirst_0 = True
            # in order to skip the first calculation of step duration, which is the step out of the box
            skipfirst_1 = True
            for d in range(
                    len(data[0])):  # loop calculating time between touches + touch time, thus forming a step time
                if data[0][d] == 0:  # distinction between the different sides
                    if data[1][d] > farthest_rung_side_0:
                        farthest_rung_side_0 = data[1][d]
                        step_duration = data[2][d] - start_step_0
                        start_step_0 = data[2][d]
                        if skipfirst_0:
                            skipfirst_0 = False
                            continue
                        step_durations_all.append(step_duration)
                        # calculated step lengths that exceed a certain
                        # amount of time must be discarded  #
                if data[0][d] == 1:
                    if data[1][d] > farthest_rung_side_1:
                        farthest_rung_side_1 = data[1][d]
                        step_duration = data[2][d] - start_step_1
                        start_step_1 = data[2][d]
                        if skipfirst_1:
                            skipfirst_1 = False
                            continue
                        step_durations_all.append(step_duration)

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
        output_file("ERCC1_%s_stepdurationdistribution_ses%s.html" % sessionnr, comment)
        show(p)

    def plot_touchdurationdistribution(self, database, min_dur, max_dur, sample, sessionnr=None, subject_ids=None,
                                       durationlimit=None, touchlimit=None, durationlimit_lower=None,
                                       touchlimit_lower=None, walkside=None,
                                       rungpick=None, directionpick=None, trialselection=None, comment=None):
        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"       # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"       # Balb/CBYJ females
        # customwhere = "session.subject_id in ('96','97','98','99','100','101')"  # C57Bl6 females
        if subject_ids:
            # switch trialselection in both querybuilders on if used
            trialselection = database.filterusefultrials(subject_ids, min_dur, max_dur, sessionnr,
                                                         durationlimit_upper=durationlimit, touchlimit_upper=touchlimit,
                                                         durationlimit_lower=durationlimit_lower,
                                                         touchlimit_lower=touchlimit_lower,
                                                         directionpick=directionpick)[0]

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]
        if trialselection:
            data_x = []
            data_x_redundant = []
            for t in trialselection:
                query = database.query_builder(outputlist,
                                               jointables=['session', 'trial', 'touch'],
                                               touchlimits=[min_dur, max_dur],
                                               sessionpick=None,
                                               valuelimits1=('touch', 'trial_id', 0, 250000),
                                               var_eq1=['touch', 'trial_id', t, '='],
                                               var_eq2=None,  # ['session', 'sessionnr', 8, '='],
                                               vars_comp1=None,
                                               order1=None,
                                               walkside=walkside,
                                               trialselection=None,
                                               rungpick=rungpick)

                data = database.query(query)
                for d in range(len(data[0])):
                    data_x.append(data[3][d] - data[2][d])
                # 2nd query for visualizing the part of the data unused for calculations, querying outside of
                # min_dur/max_dur, with a cutoff set at [max_dur + x]
                query = database.query_builder([('touch', 'side'), ('touch', 'rung'), ('touch', 'touch_begin'),
                                                ('touch', 'touch_end')],
                                               jointables=['session', 'trial', 'touch'],
                                               touchlimits=None,
                                               customwhere="(((touch.touch_end - touch.touch_begin) >= %s AND ("
                                                           "touch.touch_end - touch.touch_begin) <= 1000) OR  ( "
                                                           "touch.touch_end - touch.touch_begin) <= 34) " % (max_dur),
                                               sessionpick=None,
                                               valuelimits1=('touch', 'trial_id', 0, 250000),
                                               var_eq1=['touch', 'trial_id', t, '='],
                                               var_eq2=None,  # ['session', 'sessionnr', 8, '='],
                                               vars_comp1=None,
                                               order1=None,
                                               walkside=walkside,
                                               rungpick=rungpick,
                                               trialselection=None)
                data = database.query(query)
                if not data:
                    continue
                for d in range(len(data[0])):
                    data_x_redundant.append(data[3][d] - data[2][d])
        else:
            query = database.query_builder(outputlist,
                                           jointables=['session', 'trial', 'touch'],
                                           touchlimits=[min_dur, max_dur],
                                           sessionpick=None,
                                           valuelimits1=('touch', 'trial_id', 0, 250000),
                                           var_eq1=None,
                                           var_eq2=None,  # ['session', 'sessionnr', 8, '='],
                                           vars_comp1=None,
                                           order1=None,
                                           walkside=walkside,
                                           trialselection=None,
                                           rungpick=rungpick)

            data = database.query(query)
            data_x = []
            for d in range(len(data[0])):
                data_x.append(data[3][d] - data[2][d])
            # 2nd query for visualizing the part of the data unused for calculations, querying outside of
            # min_dur/max_dur, with a cutoff set at [max_dur + x]
            query = database.query_builder([('touch', 'side'), ('touch', 'rung'), ('touch', 'touch_begin'),
                                            ('touch', 'touch_end')],
                                           jointables=['session', 'trial', 'touch'],
                                           touchlimits=None,
                                           customwhere="(((touch.touch_end - touch.touch_begin) >= %s AND ("
                                                       "touch.touch_end - touch.touch_begin) <= 1000) OR  ( "
                                                       "touch.touch_end - touch.touch_begin) <= 34) " % (max_dur),
                                           sessionpick=None,
                                           valuelimits1=('touch', 'trial_id', 0, 250000),
                                           var_eq1=None,
                                           var_eq2=None,  # ['session', 'sessionnr', 8, '='],
                                           vars_comp1=None,
                                           order1=None,
                                           walkside=walkside,
                                           rungpick=rungpick,
                                           trialselection=None)
            data = database.query(query)
            data_x_redundant = []
            for d in range(len(data[0])):
                data_x_redundant.append(data[3][d] - data[2][d])
        data_all = data_x + data_x_redundant

        # histogram fitting
        bandwith = max_dur - min_dur
        hist_all, edges_all = np.histogram(data_all, density=True, bins=round(bandwith / 6))
        hist_redundant_lower, hist_redundant_upper, edges_redundant_lower, edges_redundant_upper = [], [], [], []
        hist_relevant, edges_relevant = [], []
        for i in range(len(hist_all)):
            if edges_all[i] < min_dur:
                hist_redundant_lower.append(hist_all[i])
                if len(edges_redundant_lower) == 0 or not (edges_redundant_lower[-1] == edges_all[i]):
                    edges_redundant_lower.append(edges_all[i])  # add left edge if not already in there
                edges_redundant_lower.append(edges_all[i + 1])  # add next right edge
            elif edges_all[i] > max_dur:
                hist_redundant_upper.append(hist_all[i])
                if len(edges_redundant_upper) == 0 or not (edges_redundant_upper[-1] == edges_all[i]):
                    edges_redundant_upper.append(edges_all[i])  # add left edge if not already in there
                edges_redundant_upper.append(edges_all[i + 1])  # add next right edge
            else:
                hist_relevant.append(hist_all[i])
                if len(edges_relevant) == 0 or not edges_relevant[-1] == edges_all[i]:
                    edges_relevant.append(edges_all[i])  # add left edge if not already in there
                edges_relevant.append(edges_all[i + 1])  # add next right edge

        # plotting
        norm_params = stats.norm.fit(data_x)
        mu = norm_params[0]
        sigma = norm_params[1]
        median = statistics.median(data_x)
        # maxwell_params = stats.maxwell.fit(data_x)
        # gamma_params = stats.gamma.fit(data_x)
        # expon_params = stats.expon.fit(data_x)
        lognorm_params = stats.lognorm.fit(data_x, loc=0)
        x = np.linspace(0, max_dur, 1000)
        # other distribtions that aren't really relevant
        # pdf_normal = stats.norm.pdf(x, *norm_params)
        # pdf_maxwell = stats.maxwell.pdf(x, *maxwell_params)
        # pdf_gamma = stats.gamma.pdf(x, *gamma_params)
        # pdf_expon = stats.expon.pdf(x, *expon_params)
        # arbitrarily poked the scale and loc parameters to nicely fit data
        #

        # pdf_lognorm = stats.lognorm.pdf(x, lognorm_params[0]*v, lognorm_params[1] + v1, lognorm_params[2] + v2)
        p = figure(title="touchdurationdistribution | nr of touches: %i " % len(data_x),
                   x_axis_label='duration', y_axis_label=' frequency ',
                   x_range=(0, 600), y_range=(0, 0.02), plot_width=1000, plot_height=1000)
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        legend = Legend(items=[
            LegendItem(label="Nr of touches: %s " % len(data_x), index=2),
        ])
        p.add_layout(legend)

        # p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="black")
        # p.quad(bottom=0, top=hist_redu, left=edges_redu[:-1], right=edges_redu[1:], fill_color="green",
        #        line_color="black")
        # p.quad(bottom=0, top=hist_all, left=edges_all[:-1], right=edges_all[1:], fill_color="grey",
        #        line_color="black")
        # p.quad(bottom=0, top=hist_redundant_lower, left=edges_redundant_lower[:-1], right=edges_redundant_lower[1:],
        #        fill_color="blue",
        #        line_color="black",
        #        alpha=0.1)
        # p.quad(bottom=0, top=hist_redundant_upper, left=edges_redundant_upper[:-1], right=edges_redundant_upper[1:],
        #        fill_color="blue",
        #        line_color="black",
        #        alpha=0.1)
        p.quad(bottom=0, top=hist_relevant, left=edges_relevant[:-1], right=edges_relevant[1:], fill_color="blue",
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
        output_file("ERCC1_%s_touchdurationdistribution_ses%s.html" % sessionnr, comment)
        show(p)

    def plot_trialdurationvstouches(self, database, min_dur, max_dur, subjectrange,
                                    sessionnr=None, durationlimit_upper=None, durationlimit_lower=None,
                                    touchlimit_upper=None, touchlimit_lower=None,
                                    control=None, mutant=None, directionpick=None):
        # customwhere = "session.subject_id in ('78','79','80','81','82','83')"  # Balb/CBYJ males
        # customwhere = "session.subject_id in ('84','85','86','87','88','89')"       # C57Bl6 males
        # customwhere = "session.subject_id in ('90','91','92','93','94','95')"       # Balb/CBYJ females
        # customwhere = "session.subject_id in ('96','97','98','99','100','101')"  # C57Bl6 females
        # mice = ('78', '79', '80', '81', '82', '83')
        # mice = ('84', '85', '86', '87', '88', '89')
        # mice = ('90', '91', '92', '93', '94', '95')
        # mice = ('96', '97', '98', '99', '100', '101')
        sessionnr = sessionnr
        state = 6
        nrtrials = 0
        colors = ["blue", "green", "purple", "cyan", "violet", "pink", "darkgreen", "lightsalmon"]
        p = figure(title="trial duration vs number of touches",
                   x_axis_label='duration (ms)',
                   y_axis_label='number of touches',
                   x_range=(0, 10000), y_range=(0, 100),
                   plot_height=900,
                   plot_width=900)
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        #  heatmap matrix generation
        lowerdurationbound = 0
        upperdurationbound = 10000
        lowertouchbound = 0
        uppertouchbound = 100

        horizontal = 30
        vertical = 30
        x = np.linspace(lowerdurationbound, upperdurationbound, 2 * horizontal + 1)
        y = np.linspace(lowertouchbound, uppertouchbound, 2 * vertical + 1)
        x = x[1::2]  # keep only the centers of the boxes for calculation
        y = y[1::2]
        matrixrowsize = len(x) * len(y)
        matrix1 = np.zeros((matrixrowsize, 3))

        for m1 in range(len(y)):
            for n in range(len(x)):
                matrix1[m1 * len(x) + n][0] = x[n]
                matrix1[m1 * len(x) + n][1] = y[m1]

        for m in range(len(subjectrange)):
            print("subject: %s" % subjectrange[m])
            customwhere = "session.subject_id in ('%s')" % subjectrange[m]
            if sessionnr:
                query = database.query_builder([('DISTINCT(state', 'trial_id)'), ('state', 'state_duration')],
                                               jointables=['session', 'trial', 'touch', 'state'],
                                               touchlimits=[min_dur, max_dur],
                                               sessionpick=None,
                                               valuelimits1=None,
                                               var_eq1=['state', 'state', state, '='],
                                               var_eq2=['session', 'sessionnr', sessionnr, '='],
                                               vars_comp1=None,
                                               order1=['state', 'trial_id', 'ASC'],
                                               customwhere=None,
                                               subjectpick=subjectrange[m],
                                               directionpick=directionpick)
            else:
                query = database.query_builder([('DISTINCT(state', 'trial_id)'), ('state', 'state_duration')],
                                               jointables=['session', 'trial', 'touch', 'state'],
                                               touchlimits=[min_dur, max_dur],
                                               sessionpick=None,
                                               valuelimits1=None,
                                               var_eq1=['state', 'state', state, '='],
                                               var_eq2=None,
                                               vars_comp1=None,
                                               order1=['state', 'trial_id', 'ASC'],
                                               customwhere=None,
                                               subjectpick=subjectrange[m],
                                               directionpick=directionpick)

            trialdurationsdata = database.query(query)
            if not trialdurationsdata:
                continue

            dt = {'trial_id': [],
                  'trialduration': [],
                  'nroftouches': []}
            trialid = -1
            temp = -1
            j = 0
            x_index = 0
            y_index = 0
            outputlist = [('touch', 'side'), ('touch', 'rung'), ('touch', 'touch_begin'), ('touch', 'touch_end')]
            for i in range(len(trialdurationsdata[0])):
                if trialid == trialdurationsdata[0][i]:  # checks whether it was the last state 6 of the trial
                    matrix1[x_index + y_index * len(x)][2] -= 1
                    del dt['trial_id'][-1]
                    if temp == -1:
                        temp = 0
                    temp += dt['trialduration'][-1]
                    del dt['trialduration'][-1]
                    del dt['nroftouches'][-1]
                    nrtrials -= 1
                    j -= 1
                else:
                    temp = -1
                trialid = trialdurationsdata[0][i]
                query = database.query_builder(outputlist,
                                               jointables=['touch'],
                                               touchlimits=[min_dur, max_dur],
                                               sessionpick=None,
                                               valuelimits1=None,
                                               var_eq1=['touch', 'touch_begin', temp, '>'],
                                               var_eq2=['touch', 'trial_id', trialid, '='],
                                               vars_comp1=None,
                                               order1=['touch', 'touch_begin', 'ASC'],
                                               customwhere=None)
                touchdata = database.query(query)
                if not touchdata:
                    continue
                nroftouches = len(touchdata[0])
                dt['trial_id'].append(trialid)
                dt['trialduration'].append(trialdurationsdata[1][i])
                dt['nroftouches'].append(nroftouches)
                nrtrials += 1
                j += 1

                #    heat map increment
                # x_index = 0
                # y_index = 0
                if upperdurationbound < trialdurationsdata[1][i] or uppertouchbound < nroftouches:
                    continue
                x_index = (np.abs(x - trialdurationsdata[1][i])).argmin()
                y_index = (np.abs(y - nroftouches)).argmin()
                # while upperdurationbound > trialdurationsdata[1][i] > x[x_index]:
                #     x_index += 1
                # while uppertouchbound > nroftouches > y[y_index]:
                #     y_index += 1
                matrix1[x_index + y_index * len(x)][2] += 1
                # if matrix1[x_index + y_index * len(x)][2] > 0:    # verification code
                #     print("x_index: " + str(x_index))
                #     print("y_index: " + str(y_index))
                #     print("value: " + str(matrix1[x_index + y_index * len(x)][2]))

            source = ColumnDataSource(data=dt)
            color = random_color()
            color = 'red'
            if control:
                if subjectrange[m] in control:
                    color = 'blue'
            if mutant:
                if subjectrange[m] in mutant:
                    color = 'red'
            p.circle(source=source, x='trialduration', y='nroftouches', line_color=color,
                     fill_color=color, alpha=0.5, legend="id = %s" % subjectrange[m])

        if control and mutant:
            usefultrials, _, controlmean, mutantmean = \
                database.filterusefultrials(subjectrange, min_dur=min_dur, max_dur=max_dur, sessionnr=sessionnr,
                                            durationlimit_upper=durationlimit_upper,
                                            durationlimit_lower=durationlimit_lower,
                                            touchlimit_upper=touchlimit_upper,
                                            touchlimit_lower=touchlimit_lower, control=control,
                                            mutant=mutant, directionpick=directionpick)
            source = ColumnDataSource(data=controlmean)
            p.circle_x(source=source, x='durmean', y='touchmean', alpha=1, legend="control mean",
                       line_color='blue', size=20)
            source = ColumnDataSource(data=mutantmean)
            p.circle_x(source=source, x='durmean', y='touchmean', alpha=1, legend="mutant mean",
                       line_color='red', size=20)
        elif control:
            usefultrials, _, controlmean = \
                database.filterusefultrials(subjectrange, min_dur=min_dur, max_dur=max_dur, sessionnr=sessionnr,
                                            durationlimit_upper=durationlimit_upper,
                                            durationlimit_lower=durationlimit_lower,
                                            touchlimit_upper=touchlimit_upper,
                                            touchlimit_lower=touchlimit_lower, control=control,
                                            mutant=mutant, directionpick=directionpick)
            source = ColumnDataSource(data=controlmean)
            p.circle_x(source=source, x='durmean', y='touchmean', alpha=1, legend="control mean",
                       line_color='blue', size=20)
        elif mutant:
            usefultrials, _, mutantmean = \
                database.filterusefultrials(subjectrange, min_dur=min_dur, max_dur=max_dur, sessionnr=sessionnr,
                                            durationlimit_upper=durationlimit_upper,
                                            durationlimit_lower=durationlimit_lower,
                                            touchlimit_upper=touchlimit_upper,
                                            touchlimit_lower=touchlimit_lower, control=control,
                                            mutant=mutant, directionpick=directionpick)
            source = ColumnDataSource(data=mutantmean)
            p.circle_x(source=source, x='durmean', y='touchmean', alpha=1, legend="mutant mean",
                       line_color='red', size=20)

        if durationlimit_upper and durationlimit_lower and touchlimit_upper and touchlimit_lower:
            p.line([durationlimit_upper, durationlimit_upper], [touchlimit_lower, touchlimit_upper],
                   line_color='black', line_dash='4 4', legend='limits')
            p.line([durationlimit_lower, durationlimit_lower], [touchlimit_lower, touchlimit_upper],
                   line_color='black', line_dash='4 4', legend='limits')
            p.line([durationlimit_lower, durationlimit_upper], [touchlimit_lower, touchlimit_lower],
                   line_color='black', line_dash='4 4', legend='limits')
            p.line([durationlimit_lower, durationlimit_upper], [touchlimit_upper, touchlimit_upper],
                   line_color='black', line_dash='4 4', legend='limits')

        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        p.title.text = ("trial duration vs number of touches | trials: %s" % nrtrials)

        # heatmap
        df = pd.DataFrame(matrix1, columns=['x', 'y', 'value'])
        # print(sum(df['value']))
        # print(matrix1[:, 2])
        # print(matrix1[:, 2] > 100)
        # print(df['value'].values)
        # print(df['value'].values > 100)
        p2 = self.heatmap("Heatmap touchnr vs trialduration", df=df, x_range=[lowerdurationbound, upperdurationbound],
                          y_range=[lowertouchbound, uppertouchbound],
                          horizontal=horizontal, vertical=vertical, colorlimit=nrtrials)
        layout = row(p, p2)
        show(layout)

    def plot_findlessthan30(self, database, min_dur, max_dur, subjectrange=None,
                            sessionnr=None, durationlimit=None, touchlimit=None,
                            control=None, mutant=None, directionpick=None):
        fewtouchestrials = []
        for i in range(0, 16000):
            query = database.query_builder([('COUNT(touch', 'id)'), ('touch', 'trial_id')],
                                           jointables=['touch'],
                                           touchlimits=[min_dur, 10000],
                                           sessionpick=None,
                                           valuelimits1=None,
                                           var_eq1=None,
                                           var_eq2=['touch', 'trial_id', i, '='],
                                           vars_comp1=None,
                                           order1=None,
                                           group1=['touch', 'trial_id'],
                                           customwhere=None,
                                           subjectpick=None,
                                           directionpick=None)
            touchdata = database.query(query)
            if not touchdata:
                continue
            if touchdata[0][0] <= 20:
                fewtouchestrials.append([touchdata[0][0], touchdata[1][0]])
                self.plot_touchpattern(database, 0, 10000, [touchdata[1][0]])

        pass

    def plot_doubletouchdurationdistribution(self, database, min_dur=None, max_dur=None, subjectrange=None,
                                             sessionnr=None, durationlimit=None, touchlimit=None,
                                             control=None, mutant=None, directionpick=None):

        outputlist = [('touch', 'side'),
                      ('touch', 'rung'),
                      ('touch', 'touch_begin'),
                      ('touch', 'touch_end'),
                      ('trial', 'directiontoend')]
        fronttouches = []
        followuptouches = []
        intervals = []

        x = np.linspace(0, 250000, 100000)
        for t in x:
            print("trial %s " % int(t))
            query = database.query_builder(outputlist,
                                           jointables=['trial', 'touch'],
                                           touchlimits=[min_dur, max_dur],
                                           sessionpick=None,
                                           valuelimits1=None,
                                           var_eq1=['touch', 'trial_id', int(t), '='],
                                           order1=['touch', 'rung, touch.side', 'ASC'])

            data = database.query(query)
            if not data:
                continue
            currentrung = 0
            currentside = 0
            currenttouch_begin = 0
            currenttouch_end = 0
            currentduration = currenttouch_end - currenttouch_begin

            for i in range(len(data[0])):
                followuptouch_begin = data[2][i]
                followuptouch_end = data[3][i]
                followuptouchduration = followuptouch_end - followuptouch_begin
                if data[1][i] > currentrung or not data[0][i] == currentside:
                    currentrung = data[1][i]
                    currentside = data[0][i]
                elif 500 > currentduration > 34 and 500 > followuptouchduration > 34 and (
                        followuptouch_begin - currenttouch_end) < 50:
                    # > 60ms to look at considerable doubles only
                    fronttouches.append(currentduration)
                    followuptouches.append(followuptouchduration)
                    intervals.append(followuptouch_begin - currenttouch_end)

                currenttouch_begin = followuptouch_begin
                currenttouch_end = followuptouch_end
                currentduration = followuptouchduration
        intervals = list(filter(lambda x: x >= 0, intervals))

        p = figure(title="doubletouchanalysis | nr of touches: %i " % (len(fronttouches) + len(followuptouches)),
                   x_axis_label='duration', y_axis_label=' frequency ',
                   x_range=(0, 600), y_range=(0, 0.02), plot_width=1000, plot_height=1000)
        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"
        p.xaxis.major_label_text_font_size = "25pt"
        p.yaxis.major_label_text_font_size = "25pt"

        legend = Legend(items=[])
        p.add_layout(legend)

        p2 = figure(
            title="followuptouchintervalanalysis | nr of intrvals: %i " % len(intervals),
            x_axis_label='duration', y_axis_label=' frequency ',
            x_range=(0, 60), y_range=(0, 0.15), plot_width=1000, plot_height=1000)
        p2.xaxis.axis_label_text_font_size = "20pt"
        p2.yaxis.axis_label_text_font_size = "20pt"
        p2.xaxis.major_label_text_font_size = "25pt"
        p2.yaxis.major_label_text_font_size = "25pt"
        p2.legend.label_text_font_size = "22px"
        legend = Legend(items=[])
        p2.add_layout(legend)

        hist_fronts, edges_fronts = np.histogram(fronttouches, density=True, bins=60)
        hist_followups, edges_followups = np.histogram(followuptouches, density=True, bins=60)
        hist_intervals, edges_intervals = np.histogram(intervals, density=True, bins=30)
        lognorm_params_fronts = stats.lognorm.fit(fronttouches, loc=0)
        x = np.linspace(0, 600, 1000)
        xx = np.linspace(0, 60, 1000)
        lognorm_params_followups = stats.lognorm.fit(followuptouches, loc=0)
        lognorm_params_intervals = stats.lognorm.fit(intervals, loc=0)
        pdf_lognorm_fronts = stats.lognorm.pdf(x, lognorm_params_fronts[0], lognorm_params_fronts[1],
                                               lognorm_params_fronts[2])
        pdf_lognorm_followups = stats.lognorm.pdf(x, lognorm_params_followups[0], lognorm_params_followups[1],
                                                  lognorm_params_followups[2])
        pdf_lognorm_intervals = stats.lognorm.pdf(xx, lognorm_params_intervals[0], lognorm_params_intervals[1],
                                                  lognorm_params_intervals[2])
        p.quad(bottom=0, top=hist_fronts, left=edges_fronts[:-1], right=edges_fronts[1:], fill_color="blue",
               line_color="blue", fill_alpha=0.1, legend='first touch', line_alpha=0.1)
        p.quad(bottom=0, top=hist_followups, left=edges_followups[:-1], right=edges_followups[1:], fill_color="yellow",
               line_color="orange", fill_alpha=0.3, legend='second touch', line_alpha=0.3)
        p.line(x, pdf_lognorm_fronts, line_color="blue", line_width=3, alpha=1, legend="PDF Lognorm 1st")
        p.line(x, pdf_lognorm_followups, line_color="orange", line_width=3, alpha=1, legend="PDF Lognorm 2nd")

        p.legend.click_policy = "hide"
        output_file("doubletouchanalysis.html")
        show(p)

        p2.quad(bottom=0, top=hist_intervals, left=edges_intervals[:-1], right=edges_intervals[1:], fill_color="green",
                line_color="green", fill_alpha=0.1, legend='intervals', line_alpha=0.3)
        p2.line(xx, pdf_lognorm_intervals, line_color="green", line_width=3, alpha=1, legend="PDF Lognorm intervals")
        output_file("followuptouchintervalanalysis.html")
        show(p2)
