import numpy as np
import math
import random
import pylab
import time
import matplotlib.pyplot as plt
import es
import sim_ann_scale

def parse(in_arr, idx_compare, idx_out, limit):
    ret_vals = []
    comp_arr = [i[idx_compare] for i in in_arr]
    idx_end = np.argwhere(np.array(comp_arr) < limit)
    end_id = -1
    if len(idx_end) == 0:
        return None
    else:
        end_id = np.squeeze(idx_end[0])
    for i in idx_out:
        out_arr = [j[i] for j in in_arr]
        ret_vals.append(out_arr[:end_id+1])
    return ret_vals

DIM = 2
SHOULD_PLOT = False


es.DIM = DIM
sim_ann_scale.DIM = DIM

mean_time_sa = []
mean_time_es = []
mean_evals_sa = []
mean_evals_es = []
i = 0
while i < 100:
    print(i)
    f_es, x_es, stats_es = es.evaluate(False, True)
    f_sa, x_sa, stats_sa = sim_ann_scale.evaluate(False, True)
    es_p = parse(stats_es, 1, [1, 3, 2], -959)
    sa_p = parse(stats_sa, 1, [1, 3, 2], -959)
    if es_p and sa_p:
        mean_time_sa.append(sa_p[2][-1])
        mean_time_es.append(es_p[2][-1])
        mean_evals_sa.append(sa_p[1][-1])
        mean_evals_es.append(es_p[1][-1])
        if SHOULD_PLOT:
            pylab.figure()
            pylab.plot(es_p[1], es_p[0] , '-o',color='r')
            pylab.plot(sa_p[1], sa_p[0], '-o',color='b')
            pylab.title('Comparison of SA and ES: Objective Function Evaluations')
            pylab.xlabel('# Objective Function Evaluations')
            pylab.ylabel('Minimum Objective Function f(x)')
            pylab.legend(['ES', 'SA'])
            pylab.show()
        i += 1

print('SA Mean Time:', np.mean(mean_time_sa), 'Evals', np.mean(mean_evals_sa))
print('ES Mean Time:', np.mean(mean_time_es), 'Evals', np.mean(mean_evals_es))
