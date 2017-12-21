import numpy as np
import math
import random
import pylab

DIM = 2
LIM = 512
PROB_INIT = 0.8
OBJ_LIM = 10000
BURN_IN = 100
D_INIT_MAG = 100
L_K = 200
ETA_MIN_SCALE = 0.5


def f(x):
    #Evaluates Eggholder function for an arbitrarily long 'x'
    c_sum = 0
    for i in range(1, len(x)):
        c_sum += -1*(x[i] + 47)*np.sin(np.sqrt(np.abs(x[i] + 0.5*x[i-1] + 47))) \
                -x[i-1]*np.sin(np.sqrt(np.abs(x[i-1] - x[i] - 47)))
    return c_sum

def gen_x_parks(x, D):
    # Uses the method to generate next solution as suggested by Parks [1990]
    u = np.random.uniform(-1, 1, DIM)
    x_new = x + np.matmul(D, u)
    # Ensure generated solution does not leave bounds of problem
    # Cap at +/- 512 for every dimension
    x_new[x_new > LIM] = LIM
    x_new[x_new < -1*LIM] = -1*LIM
    return x_new, u

def gen_D_update(D, u, alpha = 0.1, weighting = 2.1):
    # Update the diagonal matrix D as suggested by Parks
    R = np.eye(DIM)*(np.abs(np.matmul(D, u)))
    D_new = (1-alpha)*D + alpha*weighting*R
    D_new = np.clip(D_new, -1*np.inf, np.inf) #Clip values *as required*
    return D_new

def check_accept(df, T, D, u):
    if df < 0:
        return True
    else:
        R = np.eye(DIM)*(np.abs(np.matmul(D, u)))
        d_hat = np.sqrt(np.sum(np.square(R)))
        p = np.exp(-1 * df/(T*d_hat))
        if random.random() < p:
            return True
        else:
            return False

def evaluate(should_plot = False):
    #Initialisations
    x_init = np.random.uniform(-1*LIM, LIM, DIM)

    x_star = None
    f_star = np.inf
    env = BURN_IN + 1
    D_init = D_INIT_MAG*np.eye(DIM)

    # Search locally via Burn In to find starting Temperature
    diff_buffer = []
    burn_f = f(x_init)
    burn_x = x_init
    burn_d = D_init
    while (len(diff_buffer) < BURN_IN):
        new_burn_x, u = gen_x_parks(burn_x, burn_d)
        new_burn_f = f(new_burn_x)
        diff_buffer.append(new_burn_f - burn_f)
        burn_f = new_burn_f
        burn_x = new_burn_x
        if burn_f < f_star:
            f_star = burn_f
            x_star = burn_x
            burn_d = gen_D_update(burn_d, u)
    b_f = np.array(diff_buffer)
    T_avg = -1*(np.mean(b_f[b_f > 0]))/np.log(PROB_INIT)
    T_std = np.std(b_f)

    # Reset to starting value
    x = x_star
    f_x = f(x)
    D = D_init
    T = T_std

    buffer = []
    l_cur = 0
    eta_cur = 0
    hist = []
    while (env < OBJ_LIM):
        if eta_cur > ETA_MIN_SCALE*L_K or l_cur > L_K:
            eta_cur = 0
            l_cur = 0
            if len(buffer) > 1:
                T = max(0.5, np.exp(-0.7*T/np.std(buffer)))*T #As in notes
            buffer = []
        x_dash, u = gen_x_parks(x, D)
        env += 1
        f_x_dash = f(x_dash)
        l_cur += 1
        if check_accept(f_x_dash - f_x, T, D, u):
            hist.append(x_dash)
            eta_cur += 1
            buffer.append(f_x_dash)
            x = x_dash
            f_x = f_x_dash
            if f_x < f_star:
                f_star = f_x
                x_star = x
                D = gen_D_update(D, u)
        else:
            pass
    if should_plot:
        print(f_star, x_star)
        hist = np.array(hist)
        pylab.plot(hist[:, 0], hist[:, 1], 'o')
        pylab.xlabel('$x_{1}$')
        pylab.ylabel('$x_{2}$')
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
    METHOD = 'Simulated Annealing'
    M_L = 5
    SHOW = True
    while (len(f_hist) < TOT_EVALS):
        f_cur, x_cur = evaluate(SHOW)
        if SHOW: TOT_EVALS = 1
        f_hist.append(f_cur)
        x_hist.append(x_cur)
        x_b = round_to_multiple(x_cur)
        if x_b in histogram:
            histogram[x_b].append(f_cur)
        else:
            histogram[x_b] = [f_cur]
        print('Current run: ', len(f_hist), 'Value: ', f_cur)
    histogram_list = list(histogram.items())
    histogram_list = list(zip([x[0] for x in histogram_list], [x[1] for x in histogram_list], [len(x[1]) for x in histogram_list]))
    histogram_list.sort(key = lambda t: t[2], reverse= True)
    histo_x = [str(x[0]) + '\n f(x):' + str(np.around(np.mean(x[1]), 1)) for x in histogram_list]
    histo_y = [len(y[1])/TOT_EVALS for y in histogram_list]
    histo_x = histo_x[:M_L]
    histo_y = histo_y[:M_L]
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