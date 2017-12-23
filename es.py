import numpy as np
import math
import random
import pylab

DIM = 5
LIM = 512
OBJ_LIM = 10000
SIGMA_INIT_MAG = 100 # Initial magnitude of standard deviation
TAU = 1/np.sqrt(2*np.sqrt(DIM)) # As proposed by Schwefel (1995)
TAU_PRIME = 1/np.sqrt(2*DIM) # As proposed by Schwefel (1995)
BETA = 0.0873 # As proposed by Schwefel (1995)
L_CNT = 140 # The number of offspring to generate at each recombination
MU_CNT = 20 # The number of parents to select from cohort
BURN_IN = 150 # The burn in period to randomly sample points in search space
MU_PLUS_L = True # Whether to use (mu, l) or (mu + l)
OMEGA = 0.5 # Omega value specified for intermediate recombination
EPSILON = 0.001 # Absolute difference end critierion parameter
EPSILON_RELATIVE = 0.001 # Relative difference end criterion parameter
HIST_WIDTH = 0.35 # Histogram width
TOT_EVALS = 200 # Number of runs for results
METHOD = 'Evolutionary Strategies' # Name of current method

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

def get_combination(controls, sigmas, rotations, is_global = False):
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
        sigma = recombine(sigmas, is_global = True, is_intermediate = True)
        rotation = recombine(rotations, is_global = True, is_intermediate = True)
    else:
        sigma = recombine_ids(sigmas, np.unique(c_out), is_global = False, is_intermediate = True)
        rotation = recombine_ids(rotations, c_out, is_global = False, is_intermediate = True)

    c_out = recombine_ids(controls, c_out, is_global = is_global, is_intermediate = False)
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

def evaluate(should_plot = False):
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
        hist = hist + [i[0].tolist() for i in parents]
        control_arr = np.array([i[0] for i in parents])
        sigma_arr = np.array([i[1][0] for i in parents])
        rot_arr = np.array([i[1][1] for i in parents])
        # RECOMBINATION: Generate new offspring
        offspring = []
        while (len(offspring) < L_CNT):
            control, sigma, rotation = get_combination(control_arr, sigma_arr, rot_arr, is_global = True)
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
        print('Terminated due to max # objective function evals')
    if should_plot:
        print(f_star, x_star)
        hist = np.array(hist)
        pylab.figure()
        pylab.plot(hist[:, 0], hist[:, 1], 'o')
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
    TOT_EVALS = 10
    METHOD = 'Evolutionary Strategies'
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
    histo_x = [str(x[0]) + '\n E[f(x)]:' + str(np.around(np.mean(x[1]), 1)) for x in histogram_list]
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