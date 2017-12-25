import numpy as np
import math
import random
import pylab
import time
import matplotlib.pyplot as plt
import es
import sim_ann_scale

def parse(in_arr, idx_compare, idx_out, limit, is_greater = False):
    # Extracts the relevant lists from the dataset such that a limit is satisfied
    ret_vals = []
    comp_arr = [i[idx_compare] for i in in_arr]
    idx_end = 0
    if not limit  == -1:
        if is_greater:
            idx_end = np.argwhere(np.array(comp_arr) > limit)
        else:
            idx_end = np.argwhere(np.array(comp_arr) < limit)
    else: 
        idx_end = [-2]
    end_id = -1
    if len(idx_end) == 0:
        return None
    else:
        end_id = np.squeeze(idx_end[0])
    for i in idx_out:
        out_arr = [j[i] for j in in_arr]
        ret_vals.append(out_arr[:end_id+1])
    return ret_vals

DIM = 5
SHOULD_PLOT = True


es.DIM = DIM
sim_ann_scale.DIM = DIM

mean_time_sa = []
mean_time_es = []
mean_evals_sa = []
mean_evals_es = []
mean_f_sa = []
mean_f_es = []
i = 0
print('Starting...')
while i < 4:
    print(' Processing:', i, end='\r')
    f_es, x_es, stats_es = es.evaluate(False, True)
    f_sa, x_sa, stats_sa = sim_ann_scale.evaluate(False, True)
    # sa_p = parse(stats_sa, 3, [1, 3, 2], 9999, True)
    # time_end = sa_p[-1][-1]
    # es_p = parse(stats_es, 2, [1, 3, 2], time_end, True) # Runtime-adjusted on 5D case
   
    es_p = parse(stats_es, 3, [1, 3, 2], 9999, True) # Evaluate on 5D case
    sa_p = parse(stats_sa, 3, [1, 3, 2], 9999, True)
    # es_p = parse(stats_es, 1, [1, 3, 2], -959)
    # sa_p = parse(stats_sa, 1, [1, 3, 2], -959) For 2D Case
    if es_p and sa_p:
        mean_f_sa.append(sa_p[0][-1])
        mean_f_es.append(es_p[0][-1])
        mean_time_sa.append(sa_p[2][-1])
        mean_time_es.append(es_p[2][-1])
        mean_evals_sa.append(sa_p[1][-1])
        mean_evals_es.append(es_p[1][-1])
        if SHOULD_PLOT:
            pylab.figure()
            pylab.plot(es_p[1], es_p[0] , '-o',color='r')
            pylab.plot(sa_p[1], sa_p[0], '-o',color='b')
            pylab.title('SA vs. ES on 5D Eggholder')
            pylab.xlabel('# Objective Function Evaluations')
            pylab.ylabel('Minimum Objective Function f(x)')
            pylab.legend(['ES', 'SA'])
            pylab.show()
        i += 1

# print('SA Mean Time:', np.mean(mean_time_sa), 'Evals', np.mean(mean_evals_sa))
# print('ES Mean Time:', np.mean(mean_time_es), 'Evals', np.mean(mean_evals_es))
# print('SA Mean F', np.mean(mean_f_sa), 'SA Std F', np.std(mean_f_sa))
# print('ES Mean F', np.mean(mean_f_es), 'ES Std F', np.std(mean_f_es))
# c_chk = np.hstack((mean_f_sa, mean_f_es))
# bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
# pylab.figure()
# pylab.hist(mean_f_sa, bins, alpha=0.5)
# pylab.hist(mean_f_es, bins, alpha=0.5)
# pylab.title('Histogram of per-run minimum f(x) of SA vs. Time-adjusted ES')
# pylab.xlabel('f(x)')
# pylab.ylabel('Frequency')
# pylab.legend(['SA', 'ES'])
# pylab.show()
# c_chk = np.hstack((mean_time_sa, mean_time_es))
# bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
# pylab.figure()
# pylab.hist(mean_time_sa, bins, alpha=0.5)
# pylab.hist(mean_time_es, bins, alpha=0.5)
# pylab.title('Histogram of per-run runtime of SA vs. ES')
# pylab.xlabel('Time (s)')
# pylab.ylabel('Frequency')
# pylab.legend(['SA', 'ES'])
# pylab.show()