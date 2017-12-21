import numpy as np
import math
import random
import heapq
from functools import partial
import pylab

DIM = 2
LIM = 512
SEGMENTS_PER_DIM = 40
STM_LEN = 10
MTM_LEN = 10
MTM_LIM = 20
LTM_LIM = 5
SHOTGUN_LIM = 100
ROUND_LIM = 2

DELTA_INIT = 2
OBJ_LIM = 10000

class klargest(object):

    WIDTH_DIF = 30

    def __init__(self, k):
        self.k = k
        self.q = []
    def add(self, val_in):
        it_add = (np.around(val_in[0], ROUND_LIM), np.around(val_in[1], ROUND_LIM).tolist())
        if not it_add[1] in [i[1] for i in self.q]:
            self.q.append(it_add)
        if len(self.q) > self.k:
            self.q.sort(key = lambda x:x[0])
            del self.q[-1]
    def getq(self):
        return self.q
    def clear(self):
        self.q = []

def bound_x(x, e_s = 0):
    #Checks current bound on x
    lim = LIM
    x = np.array(x)
    x[x > LIM] = LIM
    x[x < -1*LIM] = LIM
    return x

def f(x):
    #Evaluates Eggholder function for an arbitrarily long 'x'
    c_sum = 0
    for i in range(1, len(x)):
        c_sum += -1*(x[i] + 47)*np.sin(np.sqrt(np.abs(x[i] + 0.5*x[i-1] + 47))) \
                -x[i-1]*np.sin(np.sqrt(np.abs(x[i-1] - x[i] - 47)))
    return c_sum

def neighbourhood(x_in, d, cur_evs, q):
    # Set up parameters for step (3)
    test_min = np.inf
    test_n_x = None
    # Run step (3) - add/subtract delta separately and record min
    lst = []
    for i in range(len(x_in)):
        t_x = np.copy(x_in)
        t_x[i] = t_x[i] + d[i]
        t_x = bound_x(t_x)
        if check_in_q(t_x, q): 
            continue
        lst.append(t_x)
        f_t_x = f(t_x)
        cur_evs += 1 # Increment number of f evaluations
        if f_t_x < test_min:
            test_min = f_t_x
            test_n_x = np.copy(t_x)

    for i in range(len(x_in)):
        t_x = np.copy(x_in)
        t_x[i] = t_x[i] - d[i]
        t_x = bound_x(t_x)
        if check_in_q(t_x, q): 
            continue
        lst.append(t_x)
        f_t_x = f(t_x)
        cur_evs += 1 # Increment number of f evaluations
        if f_t_x < test_min:
            test_min = f_t_x
            test_n_x = np.copy(t_x)

    if test_n_x is None: 
        # Resupply current element - to wait until q is emptied!
        test_n_x = np.copy(x_in)
        test_min = f(test_n_x)
    return test_min, test_n_x, cur_evs

def add_to_q(val, q, q_lim):
    q = q[:]
    if len(q) > q_lim - 1:
        del q[0]
    q.append(val.tolist())
    return q

def check_in_q(val, q):
    if np.around(val, ROUND_LIM).tolist() in q:
        return True
    else:
        return False

def grid_add(x, grid, width):
    x_c = tuple([int(x[i]/width) for i in range(len(x))])
    grid[x_c] += 1
    return grid

def grid_sample(grid, width, grid_sum):
    while (1 < 2):
        x = np.random.uniform(-1*LIM, LIM, DIM)
        x = np.around(x, ROUND_LIM)
        x = bound_x(x)
        x_c = tuple([int(x[i]/width) for i in range(len(x))])
        mean = grid_sum/SEGMENTS_PER_DIM**2
        if grid[x_c] <= mean:
            return np.array(x)  
        
def evaluate(should_plot = False):
    hist = []
    #Initialisations
    x_init = None
    f_x_init = np.inf
    mtm_q = klargest(MTM_LEN)
    grid = np.zeros(tuple([SEGMENTS_PER_DIM*2 for i in range(DIM)]))
    req_width = int(np.ceil(512*2/SEGMENTS_PER_DIM))

    for i in range(SHOTGUN_LIM):
        x_test = np.random.uniform(-1*LIM, LIM, DIM)
        x_test = np.around(x_test, ROUND_LIM)
        f_x_test = f(x_test)
        if f_x_test < f_x_init:
            f_x_init = f_x_test
            x_init = x_test
        grid = grid_add(x_test, grid, req_width)
    d_init = DELTA_INIT*np.ones((DIM))

    x = x_init
    f_x = f(x_init)
    d = d_init
    x_star = x_init
    f_star = f(x_star)
    q = []
    env = SHOTGUN_LIM

    no_improv = 0
    no_improv_next = 0

    iters = 0
    while(env < OBJ_LIM):
        f_x_n, x_n, env = neighbourhood(np.copy(x), np.copy(d), env, q)
        if f_x_n < f_x:
            x_try = x_n + (x_n - x)
            x_try = bound_x(x_try)
            f_x_n_d = f(x_try)
            if f_x_n_d < f_x_n:
                q = add_to_q(x_n, q, STM_LEN)
                f_x_n = f_x_n_d
                x_n = x_try
        f_x = f_x_n
        x = x_n
        mtm_q.add((f_x, x))
        q = add_to_q(x, q, STM_LEN)

        if f_x < f_star:
            f_star = f_x
            x_star = np.copy(x)
        else:
            no_improv += 1
            if no_improv_next > 0:
                no_improv_next += 1

        if no_improv > MTM_LIM:
            new_jump = [it[1] for it in mtm_q.getq()]
            new_jump = np.mean(new_jump, axis = 0) #Search intensification
            f_x = f(new_jump)
            x = new_jump
            no_improv_next += 1

        if no_improv_next > LTM_LIM:
            no_improv_next = 0
            no_improv = 0
            x = grid_sample(grid, req_width, iters + SHOTGUN_LIM)
            f_x = f(x)
            mtm_q.clear()

        grid = grid_add(x, grid, req_width)
        iters += 1
        hist.append(x.tolist())
    if should_plot:
        print(f_star, x_star)
        hist = np.array(hist)
        pylab.plot(hist[:, 0], hist[:, 1], '-o')
        pylab.show()
    return f_star, x_star


def round_to_multiple(x, bucket = 10):
    for i in range(len(x)):
        x[i] = int(bucket * round(float(x[i])/bucket))
    return tuple(x.tolist())

if __name__ == "__main__":
    f_hist = []
    x_hist = []
    histogram = {}
    HIST_WIDTH = 0.35
    TOT_EVALS = 50
    METHOD = 'Tabu Search'
    while (len(f_hist) < TOT_EVALS):
        f_cur, x_cur = evaluate(False)
        f_hist.append(f_cur)
        x_hist.append(x_cur)
        x_b = round_to_multiple(x_cur)
        if x_b in histogram:
            histogram[x_b].append(f_cur)
        else:
            histogram[x_b] = [f_cur]
        print('Current run: ', len(f_hist), 'Value: ', f_cur)
    histogram_list = list(histogram.items())
    histogram_list.sort(key = lambda t: t[1], reverse= False)
    histo_x = [str(x[0]) + '\n f(x):' + str(np.around(np.mean(x[1]), 1)) for x in histogram_list]
    histo_y = [len(y[1])/TOT_EVALS for y in histogram_list]
    x_hist = np.array(x_hist)
    pylab.figure()
    pylab.bar(np.arange(len(histo_x)), histo_y, HIST_WIDTH, color = 'r')
    pylab.title(METHOD + ' Minima Regions')
    pylab.xticks(np.arange(len(histo_x)), histo_x)
    pylab.xlabel('Minima Regions (with mean f(x))')
    pylab.ylabel('Proportion of runs within region')
    pylab.tight_layout()
    pylab.show()
    pylab.figure()
    pylab.plot(x_hist[:, 0], x_hist[:, 1], 'o')
    pylab.title(METHOD + ' Evaluations')
    pylab.xlabel('$x_{1}$')
    pylab.ylabel('$x_{2}$')
    pylab.show()