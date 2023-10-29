import statistics

import adodbapi
import numpy as np
from numpy.core.defchararray import upper


class Database(object):
    def __init__(self, path, passw, logger):
        self.logger = logger
        self.connection = None

        conn_str = "Provider=Microsoft.SQLSERVER.CE.OLEDB.4.0;" \
                   "Data Source=%s;" \
                   "Persist Security Info=false;" \
                   "SSCE:Max Database Size=4000;" \
                   "SSCE:Database Password=%s;" % (path, passw)

        self.connection = adodbapi.connect(conn_str)
        self.db_structure = {'session': ['trial'],  # dictionary with tables as key, and tables that have foreign
                             # keys to the key as values
                             'trial': ['touch', 'cue', 'state'],
                             'touch': [],
                             'subject': ['session'],
                             'experiment': ['session'],
                             'state': []}

    def query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        res = cursor.fetchall()
        if not res:
            print(query)
            print("Query returned 0 rows...")
            return res
        res = [[res[row.index, n] for row in res] for n in range(len(res[0]))]

        return res

    def query2_0(self, query):
        res = None
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            rows = cursor.fetchall()  # Theoretically problematic with memory for huge SELECTs
            col_names = [col[0] for col in cursor.get_description()]
            res = [dict(zip(col_names, row)) for row in rows]
        except Exception as e:
            self.logger.log("Query failed: %s" % query)
            self.logger.log("Exception: %s" % str(e))

        cursor.close()
        return res

    def define_join_order(self, jointables):
        linklist = []
        partnerlist = []
        joinorder = []

        if len(jointables) > 1:
            for i in range(len(jointables)):  # every table gets a partner
                for j in range(len(self.db_structure[jointables[i]])):  # loop over all foreign keys
                    if self.db_structure[jointables[i]][j] in jointables:  # check for partner tables
                        partnerlist.append([jointables[i], self.db_structure[jointables[i]][j]])

            linklist.append(partnerlist[0])
            partnerlist.remove(partnerlist[0])

            timeout = 0
            breakout = False
            while len(partnerlist) > 0:
                timeout += 1
                for i in range(len(partnerlist)):
                    for j in range(len(linklist)):
                        if linklist[j][0] in partnerlist[i] or linklist[j][1] in partnerlist[i]:
                            linklist.append(partnerlist[i])
                            partnerlist.remove(partnerlist[i])
                            breakout = True
                            break
                    if breakout:
                        breakout = False
                        break
                    if timeout > 1000:
                        raise ValueError(
                            "No join structure possible, properly define all tables required for this query.")

            for k in range(len(linklist)):
                joinorder.append((linklist[k][0], "id", linklist[k][1], "%s_id" % linklist[k][0]))

            return joinorder
        else:
            joinorder = [jointables]
            return joinorder

    def query_builder(self, outputlist, jointables=None, touchlimits=None, sessionlimits=None, sessionpick=None,
                      valuelimits1=None, valuelimits2=None,
                      vars_comp1=None, vars_comp2=None, var_eq1=None, var_eq2=None, var_eq3=None,
                      range1=None, order1=None,
                      group1=None, customwhere=None, customwhere2=None,
                      rungpick=None, rungrange=None, rungside=None, subjectrange=None, subjectpick=None,
                      directionpick=None, trialselection=None, walkside=None):
        # - output list should be a dictionary in the form outputlist = {'variable1': 'origin_table',
        # 'variable2':'origin_table'} where origin_table is the table where the variable comes from
        # - joinorder is a list of tuple (tuple format [(table1, Key_1, table2, Key_2),(...),(...)],
        # where table1 is the present
        # table and table 2 is to be linked), which should be in hierarchal order of JOINING (with the base table as
        # the 1st one), eg. call them in the way the query should call them.
        # note: table1 can be the same as table2
        # - touchlimits requires a tuple/list with
        # the lower touch duration cutoff and the higher one, requires touch table
        # - valuelimits require a tuple containing the table, variable and the values you want to limit,
        # so does valuelimits2
        # - vars_comp requires a tuple like (table1, var1, table2, var2, operator) of the vars you want to compare,
        # plus the operator as a string(+/=/</>/>=/<=/<>).
        # - customwhere allows you to add a custom constraint, note: only the constraint, no "WHERE" or "AND/OR"
        # - var_eq requires a tuple like (table1, var1, value, operator)
        # - range1 requires a range(x) of values, given by a tuple with table name, variable and the range
        # - orderby requires a tuple with the variable and its belonging table, plus ASC or DESC
        # - rungpick requires the touch and trial table, and a 'high'/'low' as input
        # - rungrange requires touch table and a range of the desired rungs
        # - rungside requires touch table and the side you want by 0/1 or 'red'/'green'
        # - rungrange wants a tuple with subject ids, requires session table
        # - subjectpick wants an int with the subject id as input,
        # - directionpick wants 1/0 or 'end/start', requires trial table
        # - trialselection wants an list/tuple of integers, requires trial table
        # requires a table with subject_id (session, useriv_values)

        select = "SELECT "
        for i in range(len(outputlist) - 1):
            select = select + " %s.%s, " % (outputlist[i][0], outputlist[i][1])  # add all values of output
        select = select + " %s.%s " % (outputlist[len(outputlist) - 1][0],
                                       outputlist[len(outputlist) - 1][1])  # add last value, without comma

        joinorder = self.define_join_order(jointables)
        from_clause = " FROM %s " % joinorder[0][0]
        joins = ""
        if len(jointables) > 1:
            for i in range(len(joinorder)):
                joins = joins + " JOIN %s ON %s.%s=%s.%s " % (joinorder[i][2], joinorder[i][0], joinorder[i][1],
                                                              joinorder[i][2], joinorder[i][3])

        where = " WHERE "
        if touchlimits:
            where += " (touch.touch_end - touch.touch_begin) >= %i AND " % touchlimits[0]
            where += " (touch.touch_end - touch.touch_begin) <= %i AND " % touchlimits[1]
        if sessionlimits:
            where += " session.sessionnr >= %i AND " % sessionlimits[0]
            where += " session.sessionnr <= %i AND " % sessionlimits[1]
        if sessionpick:
            where += " session.sessionnr = %i AND " % sessionpick
        if valuelimits1:
            where += " %s.%s >= %i AND " % (valuelimits1[0], valuelimits1[1], valuelimits1[2])
            where += " %s.%s <= %i AND " % (valuelimits1[0], valuelimits1[1], valuelimits1[3])
        if valuelimits2:
            where += " %s.%s >= %i AND " % (valuelimits2[0], valuelimits2[1], valuelimits2[2])
            where += " %s.%s <= %i AND " % (valuelimits2[0], valuelimits2[1], valuelimits2[3])
        if vars_comp1:
            where += " %s.%s %s %s.%s AND " % (
                vars_comp1[0], vars_comp1[1], vars_comp1[4], vars_comp1[2], vars_comp1[3])
        if vars_comp2:
            where += " %s.%s %s %s.%s AND " % (
                vars_comp2[0], vars_comp2[1], vars_comp2[4], vars_comp2[2], vars_comp2[3])
        if var_eq1:
            where += " %s.%s %s %s AND " % (var_eq1[0], var_eq1[1], var_eq1[3], var_eq1[2])
        if var_eq2:
            where += " %s.%s %s %s AND " % (var_eq2[0], var_eq2[1], var_eq2[3], var_eq2[2])
        if var_eq3:
            where += " %s.%s %s %s AND " % (var_eq3[0], var_eq3[1], var_eq3[3], var_eq3[2])
        if range1:
            where += " ( "
            for i in range(range1[2], range1[3]):
                where += " %s.%s = %i OR " % (range1[0], range1[1], i)
            where += " 1 = 0 ) AND "
        if customwhere:
            where += customwhere + " AND "
        if customwhere2:
            where += customwhere2 + " AND "
        if rungpick:
            if rungpick == "high":
                where += "(NOT(touch.side = 0) OR (touch.rung % 2 = 1)) AND " \
                         "(NOT(touch.side = 1) OR (touch.rung % 2 = 0)) AND "
            if rungpick == "low":
                where += "(NOT(touch.side = 0) OR (touch.rung % 2 = 0)) AND " \
                         "(NOT(touch.side = 1) OR (touch.rung % 2 = 1)) AND "
        if rungrange:
            where += " touch.rung >= %s AND " % rungrange[0]
            where += " touch.rung <= %s AND " % rungrange[1]
        if rungside:
            if upper(rungside) == "RED":
                where += " touch.side = 0 AND "
            if upper(rungside) == "GREEN":
                where += " touch.side = 1 AND "
        if subjectrange:
            where += " session.subject_id in ("
            for i in range(len(subjectrange) - 1):
                where += "'%s'," % subjectrange[i]
            where += "'%s') AND " % subjectrange[-1]
        if subjectpick:
            where += " session.subject_id = %s AND " % subjectpick
        if directionpick:
            if directionpick == 1 or directionpick == 'end':
                where += " trial.directiontoend = 1 AND "
            if directionpick == -1 or directionpick == 'start':
                where += " trial.directiontoend = 0 AND "
        if trialselection:
            where += " trial.id in ("
            for i in range(len(trialselection) - 1):
                where += "'%s'," % trialselection[i]
            where += "'%s') AND " % trialselection[-1]
            if walkside:
                if upper(walkside) == "LEFT":
                    where += " ((touch.side = 0 and trial.directiontoend = 1) OR " \
                             "(touch.side = 1 and trial.directiontoend = 0)) AND "
                if upper(walkside) == "RIGHT":
                    where += " ((touch.side = 1 and trial.directiontoend = 1) OR " \
                             "(touch.side = 0 and trial.directiontoend = 0)) AND "

        where += " 1 = 1 "  # ever true statement to end where clause to escape ANDs

        orderby = ""
        if order1:
            orderby = " ORDER BY %s.%s %s " % (order1[0], order1[1], order1[2])
        group = ""
        if group1:
            group = " GROUP BY %s.%s" % (group1[0], group1[1])

        query = select + from_clause + joins + where + orderby + group

        return query

    def filterusefultrials(self, subject_nrs, min_dur=0, max_dur=100000, sessionnr=None,
                           durationlimit_upper=None, durationlimit_lower=None,
                           touchlimit_upper=None, touchlimit_lower=None, control=None, mutant=None,
                           directionpick=None, noremoval=None, nofilter=None, customsubsescombo=None,
                           disregardsubjectlimit=30):

        global controlmean, mutantmean
        print("Finding useful trials nrs for subject_ids %s, session = %s, min. cutoff = %s ms, max. cutoff = %s ms \n "
              "durationlimit_upper/lower = %s | %s, touchlimits upper/lower = %s | %s" %
              (subject_nrs, sessionnr, min_dur, max_dur, durationlimit_upper, durationlimit_lower,
               touchlimit_upper, touchlimit_lower))
        usefultrials = []
        trialsubject = []
        controltrials = []
        mutanttrials = []
        mousewarningcount = {}
        # 'mouse_id': [0 for _ in range(len(subject_nrs))],
        #                  'count': [0 for _ in range(len(subject_nrs))]}
        dt = {'trial_id': [],
              'trialduration': [],
              'nroftouches': [],
              'mouse_id': [],
              'useful': []}

        for m in range(len(subject_nrs)):
            subject = subject_nrs[m]
            if customsubsescombo:
                subject = customsubsescombo[m][1]
                sessionnr = customsubsescombo[m][2]
            if sessionnr:
                query = self.query_builder([('DISTINCT(state', 'trial_id)'), ('state', 'state_duration')],
                                           jointables=['session', 'trial', 'touch', 'state'],
                                           touchlimits=[min_dur, max_dur],
                                           sessionpick=None,
                                           valuelimits1=None,
                                           var_eq1=['state', 'state', 6, '='],
                                           var_eq2=['session', 'sessionnr', sessionnr, '='],
                                           vars_comp1=None,
                                           order1=['state', 'trial_id', 'ASC'],
                                           customwhere=None,
                                           subjectpick=subject,
                                           directionpick=directionpick, )
            else:
                query = self.query_builder([('DISTINCT(state', 'trial_id)'), ('state', 'state_duration')],
                                           jointables=['session', 'trial', 'touch', 'state'],
                                           touchlimits=[min_dur, max_dur],
                                           sessionpick=None,
                                           valuelimits1=None,
                                           var_eq1=['state', 'state', 6, '='],
                                           var_eq2=None,
                                           vars_comp1=None,
                                           order1=['state', 'trial_id', 'ASC'],
                                           customwhere=None,
                                           subjectpick=subject,
                                           directionpick=directionpick, )
            trialdurationsdata = self.query(query)
            if not trialdurationsdata:
                continue

            trialid = -1
            temp = -1
            j = 0
            outputlist = [('touch', 'side'), ('touch', 'rung'), ('touch', 'touch_begin'), ('touch', 'touch_end')]
            for i in range(len(trialdurationsdata[0])):
                if trialid == trialdurationsdata[0][i]:  # checks whether it was the last state 6 of the trial
                    del dt['trial_id'][-1]
                    if temp == -1:
                        temp = 0
                    temp += dt['trialduration'][-1]
                    del dt['trialduration'][-1]
                    del dt['nroftouches'][-1]
                    del dt['mouse_id'][-1]
                    del dt['useful'][-1]
                    j -= 1
                else:
                    temp = -1
                trialid = trialdurationsdata[0][i]
                query = self.query_builder(outputlist,
                                           jointables=['touch'],
                                           touchlimits=[min_dur, max_dur],
                                           sessionpick=None,
                                           valuelimits1=None,
                                           var_eq1=['touch', 'touch_begin', temp, '>'],
                                           var_eq2=['touch', 'trial_id', trialid, '='],
                                           vars_comp1=None,
                                           order1=['touch', 'touch_begin', 'ASC'],
                                           customwhere=None)
                touchdata = self.query(query)
                if not touchdata:
                    continue
                #     TODO: remove above, as this should not happen but happened in a certain case
                nroftouches = len(touchdata[0])
                dt['trial_id'].append(trialid)
                dt['trialduration'].append(trialdurationsdata[1][i])
                dt['nroftouches'].append(nroftouches)
                dt['mouse_id'].append(subject_nrs[m])
                dt['useful'].append(False)
                j += 1

        if not dt['trial_id']:
            # check if any trials were found
            print("No trials found for subjects %s" % subject_nrs)
            return usefultrials, dt  # which are empty

        for t_id, dur, nr, m_id, uf in zip(dt['trial_id'], dt['trialduration'], dt['nroftouches'],
                                           dt['mouse_id'], dt['useful']):
            if durationlimit_lower > dur or dur > durationlimit_upper or touchlimit_lower > nr or nr > touchlimit_upper:
                #     # usefultrials.append([t_id])
                #     dt['useful'][dt['trial_id'].index(t_id)] = True
                # else:
                if not "%s" % dt['mouse_id'][dt['trial_id'].index(t_id)] in mousewarningcount.keys():
                    mousewarningcount[("%s" % dt['mouse_id'][dt['trial_id'].index(t_id)])] = 1
                else:
                    mousewarningcount[("%s" % dt['mouse_id'][dt['trial_id'].index(t_id)])] += 1
        removemice = []
        trialsremoved = 0
        for key, val in mousewarningcount.items():
            print(key, val)
            if val > disregardsubjectlimit:
                print("Warning: a lot of disuseful trials (%s) for mouse id: %s." % (val, key))
                if noremoval:
                    print("Removing this mouse from experiment data? NO (Noremoval enabled)")
                elif not sessionnr:
                    print("Removing this mouse from experiment data? NO "
                          "(No session selectec so removelimit does not apply)")
                else:
                    print("Removing this mouse from experiment data? YES")
                    removemice.append(key)
        for t_id, dur, nr, m_id, uf in zip(dt['trial_id'], dt['trialduration'], dt['nroftouches'],
                                           dt['mouse_id'], dt['useful']):
            if (durationlimit_lower <= dur <= durationlimit_upper and
                touchlimit_lower <= nr <= touchlimit_upper and str(m_id) not in removemice) or nofilter:
                usefultrials.append(t_id)
                if control:
                    if m_id in control:
                        controltrials.append(t_id)
                if mutant:
                    if m_id in mutant:
                        mutanttrials.append(t_id)
                dt['useful'][dt['trial_id'].index(t_id)] = True
            elif str(m_id) in removemice:
                trialsremoved += 1
        if control and len(controltrials) > 0:
            sumdur = 0
            sumtouch = 0
            for t_id in controltrials:
                sumdur += dt['trialduration'][dt['trial_id'].index(t_id)]
                sumtouch += dt['nroftouches'][dt['trial_id'].index(t_id)]
            controldurmean = sumdur / len(controltrials)  # TODO: general TODO to build in warning if array is empty
            # and thus probably never filled by a fault in settings at run, eg. no control ids given but anyway given
            # as input.
            controltouchesmean = sumtouch / len(controltrials)
            controlmean = {'durmean': [controldurmean], 'touchmean': [controltouchesmean]}
        if mutant and len(mutanttrials) > 0:
            sumdur = 0
            sumtouch = 0
            for t_id in mutanttrials:
                sumdur += dt['trialduration'][dt['trial_id'].index(t_id)]
                sumtouch += dt['nroftouches'][dt['trial_id'].index(t_id)]
            mutantdurmean = sumdur / len(mutanttrials)
            mutanttouchesmean = sumtouch / len(mutanttrials)
            mutantmean = {'durmean': [mutantdurmean], 'touchmean': [mutanttouchesmean]}

        if not usefultrials:
            # check if any trials left after removing
            print("No trials valid (either removed from selection and/or nonexistent subjects) for subjects %s"
                  % subject_nrs)
            return usefultrials, dt  # which are empty

        print("%.1f%% (%s) of considered trials useful." %  # , %s mice (%s) unconsidered. " % (
              ((len(usefultrials) / (len(dt['trial_id']) - trialsremoved) * 100), len(usefultrials)))
        if nofilter:
            print("Nofilter enabled")
        # , len(removemice),
        #     ', '.join(map(str, removemice))))
        if mutant:
            print("%s mutant trials in selection." % len(mutanttrials))
        if control:
            print("%s control trials in selection." % len(controltrials))
        #     TODO: make a cluster average for control and mutant independent

        # # print(dt)
        # print(usefultrials)
        # print(len(usefultrials))
        if len(mutanttrials) > 0 and len(controltrials) > 0:
            return usefultrials, dt, controlmean, mutantmean  # returns list of integers
        if len(mutanttrials) > 0:
            return usefultrials, dt, mutantmean
        if len(controltrials) > 0:
            return usefultrials, dt, controlmean

        return usefultrials, dt
