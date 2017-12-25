import numpy as np
import math
import random
import pylab
import time
import matplotlib.pyplot as plt

DIM = 5
LIM = 512
OBJ_LIM = 10000
SIGMA_INIT_MAG = 50 # Initial magnitude of standard deviation
TAU = 1/np.sqrt(2*np.sqrt(DIM)) # As proposed by Schwefel (1995)
TAU_PRIME = 1/np.sqrt(2*DIM) # As proposed by Schwefel (1995)
BETA = 0.0873 # As proposed by Schwefel (1995)
L_CNT = 200 # The number of offspring to generate at each recombination
MU_CNT = 20 # The number of parents to select from cohort
BURN_IN = 150 # The burn in period to randomly sample points in search space
MU_PLUS_L = True # Whether to use (mu, l) or (mu + l)
OMEGA = 0.5 # Omega value specified for intermediate recombination
EPSILON = 0.001 # Absolute difference end critierion parameter
EPSILON_RELATIVE = 0.001 # Relative difference end criterion parameter
HIST_WIDTH = 0.35 # Histogram width
TOT_EVALS = 200 # Number of runs for results
METHOD = 'Evolutionary Strategies' # Name of current method
GEN_WALK_ID = [0, 5, 10, -1] # Desired Generations to be printed
M_L = 5 # Capped number of regions for histogram
DELTA = 2.5 # For plotting - width of each f(x) calculation
SHOW = False # Show plot
IS_GLOBAL = True
IS_CONTROL_INT = False
IS_STRATEGY_INT = True

def f(x):
    #Evaluates Eggholder function for an arbitrarily long 'x'
    c_sum = 0
    for i in range(1, len(x)):
        c_sum += -1*(x[i] + 47)*np.sin(np.sqrt(np.abs(x[i] + 0.5*x[i-1] + 47))) \
                -x[i-1]*np.sin(np.sqrt(np.abs(x[i-1] - x[i] - 47)))
    return c_sum

def bound_x(x, e_s = 0):
    #Checks current bound on x
    lim = LIM
    x = np.array(x)
    x[x > LIM] = LIM
    x[x < -1*LIM] = LIM
    return x

def update_x(x, c_matrix):
    sample = np.random.multivariate_normal(np.zeros(DIM), c_matrix)
    return x + sample

def update_strategy(sigma, rotation):
    chi_nought = np.random.normal(0, 1)
    for i in range(DIM):
        sigma[i] = sigma[i]*np.exp(TAU_PRIME*chi_nought + TAU*np.random.normal(0, 1))
        for j in range(DIM):
            rotation[i, j] = rotation[i, j] + BETA*np.random.normal(0, 1)
    return sigma, rotation

def get_cov(a_matrix, sigma):
    c_matrix = np.zeros(np.shape(a_matrix))
    for i in range(DIM):
        for j in range(DIM):
            if i == j:
                c_matrix[i, j] = sigma[i]**2
            else:
                c_matrix[i, j] = 0.5*(sigma[i]**2 - sigma[j]**2)*np.tan(2*a_matrix[i,j])
    return c_matrix

def get_combination(controls, sigmas, rotations):
    global IS_GLOBAL, IS_CONTROL_INT, IS_STRATEGY_INT
    is_global = IS_GLOBAL
    if len(controls) < 1:
        raise ValueError('Not enough values to recombine!')
    control_shape = np.shape(controls)
    c_out = None
    sigma = None
    rotation = None
    c_out = np.zeros(control_shape[-1])

    idx_one = np.random.randint(0, len(controls))
    idx_two = np.random.randint(0, len(controls))
    
    # Process Control Variables
    for i in range(control_shape[1]):
        if is_global:
            c_out[i] = np.random.randint(0, len(controls))
        else:
            toss = np.random.random()
            if toss < 0.5:
                c_out[i] = idx_two
            else: 
                c_out[i] = idx_one
    
    # Process Strategy Parameters
    if is_global:
        sigma = recombine(sigmas, is_global = True, is_intermediate = IS_STRATEGY_INT)
        rotation = recombine(rotations, is_global = True, is_intermediate = IS_STRATEGY_INT)
    else:
        sigma = recombine_ids(sigmas, np.unique(c_out), is_global = False, is_intermediate = IS_STRATEGY_INT)
        rotation = recombine_ids(rotations, c_out, is_global = False, is_intermediate = IS_STRATEGY_INT)

    if IS_CONTROL_INT and is_global:
        c_out = recombine(controls, True, True)
    else:  
        c_out = recombine_ids(controls, c_out, is_global = is_global, is_intermediate = IS_CONTROL_INT)
    return np.squeeze(c_out), np.squeeze(sigma), rotation
    
def recombine_ids(vals, ids, is_global = True, is_intermediate = False):
    if len(np.shape(vals[0])) == 1:
        vals = vals[:, np.newaxis, :]
    
    SHAPE = np.shape(vals[0])
    ids = np.int64(ids)
    if is_global and not is_intermediate and not len(ids) == SHAPE[-1]:
        # Expecting (id_a, id_b, id_c, id_e, id_a, id_f, .....)
        raise ValueError('Incorrect number of random ids for Global, Discrete recombination')
    elif not is_global and not is_intermediate: 
        # Expecting (id_a, id_b, id_a, id_a, id_b, id_a, .....)
        if not len(ids) == SHAPE[-1]:
            raise ValueError('Incorrect number of random ids for Local, Discrete recombination')
        if not len(np.unique(ids)) <= 2:
            raise ValueError('Incorrect number of different ids - expecting <= 2, got', len(np.unique(ids)))
    elif is_global and is_intermediate and not len(ids) == SHAPE[-1]:
        # Expecting (id_a, id_b, id_c, id_e, id_a, id_f, .....)
        raise ValueError('Incorrect number of random ids for Global, Intermediate recombination')
    elif not is_global and is_intermediate and not len(np.unique(ids)) <= 2:
        # Expecting (id_a, id_b)
        raise ValueError('Incorrect number of unique random ids - expecting <= 2, got', len(np.unique(ids)))

    val_out = None
    if not is_intermediate and is_global:
        temp = np.zeros(SHAPE)
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                idx = ids[i]
                temp[dim_idx, i] = vals[idx, dim_idx, i]
        val_out = temp
    elif not is_intermediate and not is_global:
        temp = np.zeros(SHAPE)
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                idx = ids[i]
                temp[dim_idx, i] = vals[idx, dim_idx, i]
        val_out = temp
    elif is_intermediate and not is_global:
        temp = np.zeros(SHAPE)
        idx_one = ids[0]
        idx_two = None
        if len(np.unique(ids)) < 2:
            idx_two = ids[0]
        else:
            idx_two = ids[1]
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                temp[dim_idx, i] = OMEGA*vals[idx_one, dim_idx, i] + (1 - OMEGA)*vals[idx_two, dim_idx, i]
        val_out = temp
    elif is_intermediate and is_global:
        raise NotImplementedError('Not Implemented!')
    return np.array(val_out)       

def recombine(vals, is_global = True, is_intermediate = False):
    if len(vals) < 1:
        raise ValueError('Not enough values to recombine!')
    if len(np.shape(vals[0])) == 1:
        vals = vals[:, np.newaxis, :]
    SHAPE = np.shape(vals[0])
    ret_val = None
    if not is_intermediate and is_global:
        temp = np.zeros(SHAPE)
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                idx = np.random.randint(0, len(vals))
                temp[dim_idx, i] = vals[idx, dim_idx, i]
        ret_val = temp
    elif not is_intermediate and not is_global:
        temp = np.zeros(SHAPE)
        idx_one = np.random.randint(0, len(vals))
        idx_two = np.random.randint(0, len(vals))
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                chk = np.random.randint(0, 2)
                if chk == 0: 
                    temp[dim_idx, i] = vals[idx_one, dim_idx, i]
                else: 
                    temp[dim_idx, i] = vals[idx_two, dim_idx, i]
        ret_val = temp
    elif is_intermediate and not is_global:
        temp = np.zeros(SHAPE)
        idx_one = np.random.randint(0, len(vals))
        idx_two = np.random.randint(0, len(vals))
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                temp[dim_idx, i] = OMEGA*vals[idx_one, dim_idx, i] + (1 - OMEGA)*vals[idx_two, dim_idx, i]
        ret_val = temp
    elif is_intermediate and is_global:
        temp = np.zeros(SHAPE)
        for dim_idx in range(SHAPE[0]):
            for i in range(SHAPE[-1]):
                idx_one = np.random.randint(0, len(vals))
                idx_two = np.random.randint(0, len(vals))
                temp[dim_idx, i] = OMEGA*vals[idx_one, dim_idx, i] + (1 - OMEGA)*vals[idx_two, dim_idx, i]
        ret_val = temp
    return np.array(ret_val)       

def evaluate(should_plot = False, ret_stats = False):
    global MU_CNT, L_CNT
    eval_time = time.time()
    stat_hist = []
     #Initialisations
    x_init = np.random.uniform(-1*LIM, LIM, DIM)

    x_star = None
    f_star = np.inf

    sigma = SIGMA_INIT_MAG*np.ones(DIM)
    rotation = np.eye(DIM)*(SIGMA_INIT_MAG**2) # 0 angle if initialised with constant sigma

    parents = []
    for i in range(BURN_IN):
        rnd = np.random.uniform(-1*LIM, LIM, DIM)
        parents.append((rnd, (sigma, rotation), f(rnd)))
    
    env = BURN_IN
    hist = []
    de = np.inf
    re = np.inf
    off_hist = []
    f_hist = []
    while (env < OBJ_LIM):
        # SELECTION: Accept only top MU items
        parents = sorted(parents, key=lambda x: x[-1])[:MU_CNT]
        if parents[0][-1] < f_star:
            f_star = parents[0][-1]
            x_star = parents[1][0]
        de = np.abs(parents[-1][-1] - parents[0][-1])
        re = EPSILON_RELATIVE/MU_CNT * np.abs(np.sum([i[-1] for i in parents]))
        if de < EPSILON or de < re:
            break
        cur = [i[0].tolist() for i in parents]
        f_cur = [i[-1] for i in parents]
        f_hist.append((np.mean(f_cur), f_cur[0]))
        stat_hist.append((np.mean(f_cur), f_star, time.time() - eval_time, env))
        hist = hist + cur
        off_hist.append(cur)
        control_arr = np.array([i[0] for i in parents])
        sigma_arr = np.array([i[1][0] for i in parents])
        rot_arr = np.array([i[1][1] for i in parents])
        # RECOMBINATION: Generate new offspring
        offspring = []
        while (len(offspring) < L_CNT):
            control, sigma, rotation = get_combination(control_arr, sigma_arr, rot_arr)
            sigma, rotation = update_strategy(sigma, rotation)
            control = bound_x(update_x(control, get_cov(rotation, sigma)))
            offspring.append((control, (sigma, rotation), f(control)))
        env += L_CNT
        # MU+L out recombination
        if MU_PLUS_L:
            offspring = parents + offspring
        parents = offspring[:]
    if not (de < EPSILON or de < re):
        pass
        # print('Terminated due to max # objective function evals')
    if should_plot:
        # Plotting code
        print('NOTE: ', f_star, x_star, 'Eval:', env)
        if DIM == 2 and False:
            x_one_mesh = np.arange(-513, 513, DELTA)
            x_two_mesh = np.arange(-513, 513, DELTA)
            X_1, X_2 = np.meshgrid(x_one_mesh, x_two_mesh)
            Z = np.zeros(np.shape(X_1))
            for i in range(len(x_one_mesh)):
                for j in range(len(x_two_mesh)):
                    Z[i, j] = f([x_two_mesh[j], x_one_mesh[i]])
            for idx in GEN_WALK_ID:
                x_coords = np.array(off_hist[idx])
                if idx < 0: 
                    cur_id = len(off_hist) + idx
                else:
                    cur_id = idx
                pylab.figure()
                pylab.contour(X_1, X_2, Z)
                pylab.plot(x_coords[:, 0], x_coords[:, 1], 'o',color='r', zorder = 1)
                pylab.title('Selection in Generation: ' + str(cur_id))
                pylab.xlabel('$x_{1}$')
                pylab.ylabel('$x_{2}$')
                pylab.show()
            hist = np.array(hist)
            pylab.figure()
            pylab.contour(X_1, X_2, Z)
            pylab.plot(hist[:, 0], hist[:, 1], 'o',color='r', zorder = 1)
            pylab.title('Visited Points (All Generations)')
            pylab.xlabel('$x_{1}$')
            pylab.ylabel('$x_{2}$')
            pylab.show()
        f_hist = np.array(f_hist)
        pylab.figure()
        pylab.plot(np.array(f_hist[:, 0]), '-')
        pylab.plot(np.array(f_hist[:, 1]), '-.')
        pylab.title('Variation of Average + Minimum Objective Function f(x) over generations')
        pylab.xlabel('Generation')
        pylab.ylabel('Objective Function value f(x)')
        pylab.legend(['Average', 'Minimum'])
        pylab.show()
    if ret_stats:
        return f_star, x_star, stat_hist
    else:
        return f_star, x_star

def round_to_multiple(x, bucket = 10):
    for i in range(len(x)):
        x[i] = int(bucket * round(float(x[i])/bucket))
    return tuple(x.tolist())

def run(should_plot = False):
    global TOT_EVALS, SHOW
    f_hist = []
    x_hist = []
    histogram = {}
    print("")
    while (len(f_hist) < TOT_EVALS):
        f_cur, x_cur = evaluate(SHOW)
        if SHOW: TOT_EVALS = 1
        f_hist.append(f_cur)
        x_hist.append(x_cur.copy())
        x_b = round_to_multiple(x_cur)
        if x_b in histogram:
            histogram[x_b].append(f_cur)
        else:
            histogram[x_b] = [f_cur]
        print(' Current run: ' + str(len(f_hist)) + ' Value: ' + str(f_cur), end='\r')
    print("\n*********************")
    print("Lowest Minimum Found at: " + str(x_hist[np.argmin(f_hist)]) + " Value: " + str(np.min(f_hist)))
    print("Average Minimum Value: " + str(np.mean(f_hist)) + " Standard Deviation: " + str(np.std(f_hist)))
    print("*********************")
    if not SHOW or not should_plot:
        x_one_mesh = np.arange(-513, 513, DELTA)
        x_two_mesh = np.arange(-513, 513, DELTA)
        X_1, X_2 = np.meshgrid(x_one_mesh, x_two_mesh)
        Z = np.zeros(np.shape(X_1))
        for i in range(len(x_one_mesh)):
            for j in range(len(x_two_mesh)):
                Z[i, j] = f([x_two_mesh[j], x_one_mesh[i]])
        histogram_list = list(histogram.items())
        histogram_list = list(zip([x[0] for x in histogram_list], [x[1] for x in histogram_list], [len(x[1]) for x in histogram_list]))
        histogram_list.sort(key = lambda t: t[2], reverse= True)
        histo_x = [str(x[0]) + '\n f(x):' + str(np.around(np.mean(x[1]), 1)) for x in histogram_list]
        histo_x_cord = np.array([x[0] for x in histogram_list])[:M_L]
        histo_y = [len(y[1])/TOT_EVALS for y in histogram_list]
        histo_x = histo_x[:M_L]
        histo_y = histo_y[:M_L]
        x_hist = np.array(x_hist)
        f_hist = np.array(f_hist)
        if (DIM == 2):
            pylab.figure()
            pylab.bar(np.arange(len(histo_x)), histo_y, HIST_WIDTH, color = 'r')
            pylab.title(METHOD + ' Minima Regions')
            pylab.xticks(np.arange(len(histo_x)), histo_x)
            pylab.xlabel('Minima Regions (with mean f(x))')
            pylab.ylabel('Proportion of runs within region')
            pylab.tight_layout()
            pylab.show()
            pylab.figure()
            pylab.contour(X_1, X_2, Z, cmap=pylab.cm.bone)
            marker_size = [10*2**(10*i) for i in histo_y]
            pylab.scatter(histo_x_cord[:, 0], histo_x_cord[:, 1], c='r', s = marker_size, edgecolor='black', zorder = 2)
            # pylab.plot(x_hist[:, 0], x_hist[:, 1], 'o')
            pylab.title(METHOD + ' Evaluations')
            pylab.xlabel('$x_{1}$')
            pylab.ylabel('$x_{2}$')
            pylab.show()
    combined = zip(f_hist.tolist(), x_hist.tolist())
    f_hist = [f for f, _ in sorted(combined)][:10]
    x_hist = [x for _, x in sorted(combined)][:10]
    return f_hist, x_hist

if __name__ == "__main__":
    #################
    # IS_GLOBAL = True
    # for i in range(4):
    #     IS_CONTROL_INT = False
    #     f_hist, x_hist = run()
    #     IS_CONTROL_INT = True
    #     f_hist_a, x_hist_a = run()
    #     c_chk = np.hstack((f_hist_a, f_hist))
    #     bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
    #     pylab.figure()
    #     pylab.hist(f_hist, bins, alpha=0.5)
    #     pylab.hist(f_hist_a, bins, alpha=0.5)
    #     pylab.title('Minimum f(x) of Discrete vs. Intermediate Control Recombination')
    #     pylab.xlabel('f(x)')
    #     pylab.ylabel('Frequency')
    #     pylab.legend(['Discrete', 'Intermediate'])
    #     pylab.show()
    #################
    # IS_GLOBAL = False
    # f_hist, x_hist = run()
    # IS_GLOBAL = True
    # f_hist_a, x_hist_a = run()
    # c_chk = np.hstack((f_hist_a, f_hist))
    # bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
    # pylab.figure()
    # pylab.hist(f_hist, bins, alpha=0.5)
    # pylab.hist(f_hist_a, bins, alpha=0.5)
    # pylab.title('Histogram of per-run minimum f(x) of Local vs. Global recombination')
    # pylab.xlabel('f(x)')
    # pylab.ylabel('Frequency')
    # pylab.legend(['Local', 'Global'])
    # pylab.show()
    # f_hist, x_hist = run()
    # IS_GLOBAL = True
    # f_hist_a, x_hist_a = run()
    # c_chk = np.hstack((f_hist_a, f_hist))
    # bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
    # pylab.figure()
    # pylab.hist(f_hist, bins, alpha=0.5)
    # pylab.hist(f_hist_a, bins, alpha=0.5)
    # pylab.title('Histogram of per-run minimum f(x) of Local vs. Global recombination')
    # pylab.xlabel('f(x)')
    # pylab.ylabel('Frequency')
    # pylab.legend(['Local', 'Global'])
    # pylab.show()
    # f_hist, x_hist = run()
    # IS_GLOBAL = True
    # f_hist_a, x_hist_a = run()
    # c_chk = np.hstack((f_hist_a, f_hist))
    # bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
    # pylab.figure()
    # pylab.hist(f_hist, bins, alpha=0.5)
    # pylab.hist(f_hist_a, bins, alpha=0.5)
    # pylab.title('Histogram of per-run minimum f(x) of Local vs. Global recombination')
    # pylab.xlabel('f(x)')
    # pylab.ylabel('Frequency')
    # pylab.legend(['Local', 'Global'])
    # pylab.show()
    #################
    # avg, std_dev = [], []
    # times = []
    # m = np.linspace(1, 15, 15, dtype=np.int16) #Go to 2sigma
    # for i in m:
    #     L_CNT = i*MU_CNT
    #     start_time = time.time()
    #     f_hist, _ = run()
    #     times.append(np.around(time.time() - start_time, 0))
    #     avg.append(np.mean(f_hist))
    #     std_dev.append(np.std(f_hist))
    # fig, ax1 = plt.subplots()
    # ax1.errorbar(m, np.array(avg) , yerr = np.array(std_dev), c = 'r', fmt = "-o", label = 'Avg. Min f(x)')
    # ax1.set_xlabel('Ratio of $\lambda$:$\mu$')
    # # Make the y-axis label, ticks and tick labels match the line color.
    # ax1.set_ylabel('Average Minimum f(x)', color='r')
    # ax1.tick_params('y', colors='r')
    # ax2 = ax1.twinx()
    # ax2.plot(m, np.array(times), '-.o', c = 'b', label = 'Time Taken')
    # ax2.set_ylabel('Time (s)', color='b')
    # ax2.tick_params('y', colors='b')
    # ax1.set_title('Variation of average minimum objective f(x) with $\lambda:\mu$ ratio')
    # ax1.legend(loc = 3)
    # ax2.legend(loc = 4)
    # fig.tight_layout()
    # plt.show()
    #################
    # TOT_EVALS = 25
    # avg, std_dev = [], []
    # times = []
    # m = np.linspace(10, 100, 10, dtype=np.int16) #Go to 2sigma
    # for i in m:
    #     MU_CNT = i
    #     L_CNT = 7*MU_CNT
    #     start_time = time.time()
    #     f_hist, _ = run()
    #     times.append(np.around(time.time() - start_time, 0))
    #     avg.append(np.mean(f_hist))
    #     std_dev.append(np.std(f_hist))
    # fig, ax1 = plt.subplots()
    # ax1.errorbar(m, np.array(avg) , yerr = np.array(std_dev), c = 'r', fmt = "-o", label = 'Avg. Min f(x)')
    # ax1.set_xlabel('Number of parents per selection')
    # # Make the y-axis label, ticks and tick labels match the line color.
    # ax1.set_ylabel('Average Minimum f(x)', color = 'r')
    # ax1.tick_params('y', colors='r')
    # ax2 = ax1.twinx()
    # ax2.plot(m, np.array(times), '-.o', c = 'b', label = 'Time Taken')
    # ax2.set_ylabel('Time (s)', color='b')
    # ax2.tick_params('y', colors='b')
    # ax1.set_title('Variation of average minimum objective f(x) with # parents')
    # ax1.legend(loc = 2)
    # ax2.legend(loc = 4)
    # fig.tight_layout()
    # plt.show()
    #################
    # MU_PLUS_L = False
    # f_hist, x_hist = run()
    # MU_PLUS_L = True
    # f_hist_a, x_hist_a = run()
    # c_chk = np.hstack((f_hist_a, f_hist))
    # bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
    # pylab.figure()
    # pylab.hist(f_hist, bins, alpha=0.5)
    # pylab.hist(f_hist_a, bins, alpha=0.5)
    # pylab.title('Histogram of per-run minimum f(x) of $(\mu, \lambda)$ vs. $(\mu + \lambda)$ strategies')
    # pylab.xlabel('f(x)')
    # pylab.ylabel('Frequency')
    # pylab.legend(['$(\mu, \lambda)$', '$(\mu + \lambda)$'])
    # pylab.show()
    #################
    # avg, std_dev = [], []
    # s = np.linspace(1, 512, 10) #Go to 2sigma
    # for i in s:
    #     SIGMA_INIT_MAG = i
    #     f_hist, _ = run()
    #     avg.append(np.mean(f_hist))
    #     std_dev.append(np.std(f_hist))
    # pylab.figure()
    # pylab.errorbar(s, np.array(avg) , yerr = np.array(std_dev), c = 'r', fmt = "o")
    # pylab.title('Average Minimum f(x) with varying magnitude of initial sigma elements')
    # pylab.xlabel('Magnitude of initial sigma elements')
    # pylab.ylabel('Average Minimum f(x)')
    # pylab.show()
    #################
    # TOT_EVALS = 75
    # IS_GLOBAL = True
    # for i in range(4):
    #     IS_STRATEGY_INT = False
    #     f_hist, x_hist = run()
    #     IS_STRATEGY_INT = True
    #     f_hist_a, x_hist_a = run()
    #     c_chk = np.hstack((f_hist_a, f_hist))
    #     bins = np.linspace(np.min(c_chk), np.max(c_chk), 15)
    #     pylab.figure()
    #     pylab.hist(f_hist, bins, alpha=0.5)
    #     pylab.hist(f_hist_a, bins, alpha=0.5)
    #     pylab.title('Minimum f(x) of Discrete vs. Intermediate Strategy Recombination')
    #     pylab.xlabel('f(x)')
    #     pylab.ylabel('Frequency')
    #     pylab.legend(['Discrete', 'Intermediate'])
    #     pylab.show()
    ################
    # TOT_EVALS = 30
    # avg, std_dev = [], []
    # w = np.linspace(0, 0.5, 11) #Go to 2sigma
    # for i in w:
    #     OMEGA = i
    #     f_hist, _ = run()
    #     avg.append(np.mean(f_hist))
    #     std_dev.append(np.std(f_hist))
    # pylab.figure()
    # pylab.errorbar(w, np.array(avg) , yerr = np.array(std_dev), c = 'r', fmt = "o")
    # pylab.title('Average Minimum f(x) with varying $\omega$')
    # pylab.xlabel('$\omega$')
    # pylab.ylabel('Average Minimum f(x)')
    # pylab.show()