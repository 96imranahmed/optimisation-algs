import numpy as np
import math
import random
import pylab
import pickle

DIM = 5
LIM = 1
SCALE = 512
PROB_INIT = 0.8 # Initial acceptance probability
OBJ_LIM = 10000 # Cap on # Evaluations
BURN_IN = 100 # Burn-in to find good starting point
D_INIT_MAG = 2 # Initial D magnitude - 1 = identity
L_K = 200 # Length of Markov Chain
ETA_MIN_SCALE = 0.6 # Proportional (to L_k) length of Markov Chain acceptances
ALPHA = 0.95 # Alpha for exponential cooling
ADAPTIVE = True # Use adaptive cooling
SHOW = False # Show plot
MIN_ACCEPTANCE = 0.001 # Min solution acceptance ratio
RESTART_THRESH = 1000 # Restart if no solutions found
TEMP_WALK_ID = [0, 3, 6, -1] # Desired Walks to be printed
HIST_WIDTH = 0.35 # Histogram width
TOT_EVALS = 200 # Number of runs for results
METHOD = 'Simulated Annealing' # Name of current method
M_L = 5 # Capped number of regions for histogram
DELTA = 2.5 # For plotting - width of each f(x) calculation

def f_ns(x):
    #Evaluates Eggholder function for an arbitrarily long 'x'
    c_sum = 0
    for i in range(1, len(x)):
        c_sum += -1*(x[i] + 47)*np.sin(np.sqrt(np.abs(x[i] + 0.5*x[i-1] + 47))) \
                -x[i-1]*np.sin(np.sqrt(np.abs(x[i-1] - x[i] - 47)))
    return c_sum

def f(x):
    # Evaluates Eggholder function for an arbitrarily long 'x'
    c_sum = 0
    x = SCALE*x # Scale-up to evaluate on (-512, 512) instead of (-1, 1)
    for i in range(1, len(x)):
        c_sum += -1*(x[i] + 47)*np.sin(np.sqrt(np.abs(x[i] + 0.5*x[i-1] + 47))) \
                -x[i-1]*np.sin(np.sqrt(np.abs(x[i-1] - x[i] - 47)))
    return c_sum

def gen_x_parks(x, D):
    # Uses the method to generate next solution as suggested by Parks [1990]
    u = np.random.uniform(-1, 1, DIM)
    x_new = x + np.matmul(D, u)
    # Ensure generated solution does not leave bounds of problem
    # Cap at +/- 512 for every dimension (scaled at +/- 1)
    x_new[x_new > LIM] = LIM
    x_new[x_new < -1*LIM] = -1*LIM
    return x_new, u

def gen_D_update(D, u, alpha = 0.1, weighting = 2.1):
    # Update the diagonal matrix D as suggested by Parks
    R = np.eye(DIM)*(np.abs(np.matmul(D, u)))
    D_new = (1-alpha)*D + alpha*weighting*R
    D_new = np.clip(D_new, -1, 1) #Clip values to +/- 1
    return D_new

def check_accept(df, T, D, u):
    if df < 0:
        return True
    else:
        R = np.eye(DIM)*(np.abs(np.matmul(D, u)))
        d_hat = np.sqrt(np.sum(np.square(R)))
        p = np.exp(-1 * df/(T*d_hat))
        guess = random.random()
        return guess < p

def evaluate(should_plot = False):
    #Initialisations
    x_init = np.random.uniform(-1, 1, DIM)

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
    T = T_avg

    buffer = []
    l_cur = 0
    eta_cur = 0
    hist = []
    did_find_sol = False
    acc = 0
    return_to_base = 0
    temp_hist = []
    cur_temp_walk = []
    f_hist = []
    while (env < OBJ_LIM):
        if eta_cur > ETA_MIN_SCALE*L_K or l_cur > L_K:
            eta_cur = 0
            l_cur = 0
            if len(cur_temp_walk) > 0:
                temp_hist.append((T, cur_temp_walk))
            cur_temp_walk = []
            if len(buffer) > 1:
                if not ADAPTIVE:
                    # Exponential Cooling Scheme (proposed by Kirkpatrick)
                    T = ALPHA * T
                else:
                    # Adaptive Cooling Scheme
                    a = max(0.5, np.exp(-0.7*T/np.std(buffer)))
                    T = a*T #As in notes
            buffer = []
            if not did_find_sol and acc/env < MIN_ACCEPTANCE:
                # Halt search when these constraints are satisfied
                # print('Ended at ', env, 'iterations')
                break
            did_find_sol = False
        x_dash, u = gen_x_parks(x, D)
        env += 1
        f_x_dash = f(x_dash)
        l_cur += 1
        df = f_x_dash - f_x
        if check_accept(df, T, D, u):
            acc += 1
            hist.append(x_dash)
            cur_temp_walk.append(x_dash)
            eta_cur += 1
            buffer.append(f_x_dash)
            x = x_dash
            f_x = f_x_dash
            f_hist.append(f_x)
            if f_x < f_star:
                f_star = f_x
                x_star = x
                D = gen_D_update(D, u)
                did_find_sol = True
            else:
                return_to_base += 1
            if return_to_base > RESTART_THRESH:
                x = x_star
                eta_cur = 0
                l_cur = 0
                did_find_sol = False
                return_to_base = 0
                print('Restarting search from current best soln')
        else:
            pass
    if env >= OBJ_LIM:
        # print('Terminated search due to maximum allowable # objective functions being exceeded')
        pass
    x_star = SCALE * x_star
    if should_plot:
        # Plotting code
        print('NOTE: ', f_star, x_star)
        if DIM == 2:
            x_one_mesh = np.arange(-513, 513, DELTA)
            x_two_mesh = np.arange(-513, 513, DELTA)
            X_1, X_2 = np.meshgrid(x_one_mesh, x_two_mesh)
            Z = np.zeros(np.shape(X_1))
            for i in range(len(x_one_mesh)):
                for j in range(len(x_two_mesh)):
                    Z[i, j] = f_ns([x_two_mesh[j], x_one_mesh[i]])
            for idx in TEMP_WALK_ID:
                T, temp_hist_coords = temp_hist[idx]
                temp_hist_coords = SCALE * np.array(temp_hist_coords)
                pylab.figure()
                pylab.contour(X_1, X_2, Z, cmap=pylab.cm.bone)
                pylab.plot(temp_hist_coords[:, 0], temp_hist_coords[:, 1], '-o',color='r', zorder = 1)
                pylab.title('Temperature Walk for T = ' + str(np.around(T, 2)))
                pylab.xlabel('$x_{1}$')
                pylab.ylabel('$x_{2}$')
                pylab.show()
            hist = SCALE * np.array(hist)
            pylab.figure()
            pylab.contour(X_1, X_2, Z)
            pylab.plot(hist[:, 0], hist[:, 1], 'o',color='r', zorder = 1)
            pylab.title('Visited Points (All Temperatures)')
            pylab.xlabel('$x_{1}$')
            pylab.ylabel('$x_{2}$')
            pylab.show()
        pylab.figure()
        pylab.plot(np.array(f_hist), '-')
        pylab.title('Objective Function f(x) with # Accepted Iterations')
        pylab.xlabel('# Accepted Iterations')
        pylab.ylabel('Objective Function value f(x)')
        pylab.show()
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
                Z[i, j] = f_ns([x_two_mesh[j], x_one_mesh[i]])
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
    f_hist = [f for f, _ in sorted(combined)][:50]
    x_hist = [x for _, x in sorted(combined)][:50]
    return f_hist, x_hist

if __name__ == "__main__":
    TOT_EVALS = 200
    # _, _ = run()
    # avg, std_dev = [], []
    # l_k_test = np.linspace(10, 1100, 20)
    # for i in l_k_test:
    #     L_K = i
    #     f_hist, _ = run()
    #     avg.append(np.mean(f_hist))
    #     std_dev.append(np.std(f_hist))
    # pylab.figure() 
    # pylab.errorbar(l_k_test, np.array(avg) , yerr = np.array(std_dev), c = 'r', fmt = "o")
    # pylab.title('Average Minimum f(x) with varying Markov Chain Length $L_k$')
    # pylab.xlabel('Markov Chain Length $L_k$ (# f(x) Evaluations)')
    # pylab.ylabel('Average Minimum f(x)')
    # pylab.show()
    ##################
    # avg, std_dev = [], []
    # d_grad = np.linspace(0, 2, 10)
    # for i in d_grad:
    #     D_INIT_MAG = i
    #     f_hist, _ = run()
    #     avg.append(np.mean(f_hist))
    #     std_dev.append(np.std(f_hist))
    # pylab.figure()
    # pylab.errorbar(d_grad, np.array(avg) , yerr = np.array(std_dev), c = 'r', fmt = "o")
    # pylab.title('Average Minimum f(x) with varying magnitude of $D$ matrix')
    # pylab.xlabel('Magnitude of diagonal element in $D$ matrix')
    # pylab.ylabel('Average Minimum f(x)')
    # pylab.show()
    ###################
    # f_hist, _ = run()
    # pickle.dump(file = open('./SA_scale_f_hist.pickle', 'wb'), obj = f_hist)
    ###################
    ADAPTIVE = False
    avg, std_dev = [], []
    a_grad = np.linspace(0, 1, 20)
    for i in a_grad:
        ALPHA = i
        f_hist, _ = run()
        avg.append(np.mean(f_hist))
        std_dev.append(np.std(f_hist))
    pylab.figure()
    pylab.errorbar(a_grad, np.array(avg) ,yerr = np.array(std_dev), c = 'r', fmt = "o")
    pylab.title('Average Minimum f(x) with ECS and varying alpha')
    pylab.xlabel('Alpha')
    pylab.ylabel('Average f(x)')
    pylab.show()
    ###################
    # ADAPTIVE = False
    # f_hist, x_hist = run()
    # ADAPTIVE = True
    # f_hist_a, x_hist_a = run()
    # c_chk = np.hstack((f_hist_a, f_hist))
    # bins = np.linspace(np.min(c_chk), np.max(c_chk), 30)
    # pylab.figure()
    # pylab.hist(f_hist, bins, alpha=0.5)
    # pylab.hist(f_hist_a, bins, alpha=0.5)
    # pylab.title('Histogram of per-run minimum f(x) with different cooling schemes')
    # pylab.xlabel('f(x)')
    # pylab.ylabel('Frequency')
    # pylab.legend(['Exponential', 'Adaptive'])
    # pylab.show()