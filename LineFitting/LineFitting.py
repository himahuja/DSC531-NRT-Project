#%%
from utilities import open_file, save_file
import apt_general_2
import mh_sampler
import analyticLineFittingToy
import numpy as np
import matplotlib.pyplot as plt
import corner
import delfi.distribution as dd
import time
import os.path 
from collections import defaultdict #for creating n-d dictionaries
import emcee 
#%%
def baseParamInit():
    
    N = 1000
    alpha_true = .5
    beta_true = -2
    err = 0.4 # err to be fixed
    labels = ['slope', 'intercept']
    true_params = [alpha_true, beta_true]
    
    seed_p = 2
    
    prior_apt =  dd.Uniform(lower = np.asarray([-10, -10]), upper = np.asarray([10, 10]), seed = seed_p)
    analyticModel = None
    apt_model = None
    s = None
    
    hyps = {'prior': prior_apt,
        'm': apt_model,
        's': s,
        'n_train': 5000,
        'n_rounds': 3,
        'n_hiddens': [50, 50],
        'pilot_samples': 2000,
        'val_frac': 0.05,
        'minibatch': 500,
        'epochs': 100,
        'density': 'maf',
        'n_mades': 5,
        'seed_inf': 1}
    
    n_steps = 15000
    
    fignames = ''
    folderName = ''
    
    param_dict = {'N': N,
                  'true_params': true_params,
                  'hyps': hyps,
                  'analyticModel': analyticModel, 
                  'fignames': fignames,
                  'folderName': folderName,
                  'n_steps': n_steps,
                  'err': err}
    
    return param_dict
#%%
def selfConsistencyOverDataSizes():
    param_dict = baseParamInit()
    seeds = [1, 10, 100, 202, 102, 1022, 1002, 102293, 10202]
    folderNameBase = 'selfConsistencyOverDataSizes'
    # param_dict['folderName'] = folderNameBase
    
    Ns = [200, 1000]
    
    for n in Ns:
        # param_dict['hyps']['n_train'] = n_train[t]
        # param_dict['hyps']['n_rounds'] = n_rounds
        # param_dict['n_steps'] = n_steps[t]
        param_dict['N'] = n
        baseFigName = str(param_dict['N'])
        # param_dict['fignames'] = str(param_dict['N']) + '_' + str(n_train[t]) + '_' + str(n_rounds)
        
        param_dict['folderName'] = os.path.join(folderNameBase, baseFigName)
        
        if not os.path.exists( param_dict['folderName'] ):
            os.mkdir(os.path.join(os.getcwd(), param_dict['folderName']))
    
        for seed in seeds:
            param_dict['hyps']['seed_inf'] = seed
            param_dict['fignames'] =  baseFigName + '_' + 'seed' + '_' + str(seed)
            apt_duration, mh_duration = main(param_dict)
        
#%%
def selfConsistencyOverParamDraws():
    param_dict = baseParamInit()
    seeds = [1, 10, 100, 202, 102, 1022, 1002, 102293, 10202]
    folderNameBase = 'selfConsistencyOverParamDraws'
    # param_dict['folderName'] = folderNameBase
    timings_avg_file_name = os.path.join(folderNameBase, 'timings_avg.pk')
    
    n_train = [500]
    n_rounds = 2
    n_steps = np.asarray(n_train) * n_rounds
    
    timings_avg = open_file(timings_avg_file_name)
    
    if timings_avg is -1:
        timings_avg = {}
    
    for t in range(len(n_train)):
        param_dict['hyps']['n_train'] = n_train[t]
        param_dict['hyps']['n_rounds'] = n_rounds
        param_dict['n_steps'] = n_steps[t]
        baseFigName = str(param_dict['N']) + '_' + str(n_train[t]) + '_' + str(n_rounds)
        # param_dict['fignames'] = str(param_dict['N']) + '_' + str(n_train[t]) + '_' + str(n_rounds)
        
        temp_avg = {}
        temp_avg['apt'] = []
        temp_avg['mh'] = []
        param_dict['folderName'] = os.path.join(folderNameBase, baseFigName)
        
        if not os.path.exists( param_dict['folderName'] ):
            os.mkdir(os.path.join(os.getcwd(), param_dict['folderName']))
    
        for seed in seeds:
            param_dict['hyps']['seed_inf'] = seed
            param_dict['fignames'] =  baseFigName + '_' + 'seed' + '_' + str(seed)
            apt_duration, mh_duration = main(param_dict)
            temp_avg['apt'].append(apt_duration)
            temp_avg['mh'].append(mh_duration)
            
        temp_avg['apt'] = np.average(np.asarray(temp_avg['apt']))
        temp_avg['mh'] = np.average(np.asarray(temp_avg['mh']))
        timings_avg[param_dict['fignames']] = [temp_avg['apt'], temp_avg['mh']]
        
    save_file(timings_avg_file_name, timings_avg)

#%%
def compareSelfConsistency():
    param_dict = baseParamInit()
    
    seeds = [1, 10, 100, 202, 102, 1022, 1002, 102293, 10202]
    
    param_dict['folderName'] = 'selfConsistency'
    for seed in seeds:
        param_dict['hyps']['seed_inf'] = seed
        param_dict['fignames'] = 'seed' + str(seed)
        main(param_dict)
#%%
def compareDifferentDataSizes():
    param_dict = baseParamInit()
    Ns = [200, 1000]
    param_dict['folderName'] = 'dataSizes'
    for n in Ns:
        param_dict['N'] = n
        param_dict['fignames'] = str(n)
        main(param_dict)
# %%
def compareDifferentParamDraws():
    param_dict = baseParamInit()
    param_dict['folderName'] = 'paramDraws'
    timings_file_name = os.path.join(param_dict['folderName'], 'timings.pk')
    n_train = [2000, 3000, 5000, 10000]
    n_rounds = 2
    n_steps = np.asarray(n_train) * n_rounds
    
    timings = open_file(timings_file_name)
    
    if timings is -1:
        timings = {}
    
    for t in range(len(n_train)):
        param_dict['hyps']['n_train'] = n_train[t]
        param_dict['hyps']['n_rounds'] = n_rounds
        param_dict['n_steps'] = n_stepse[t]
        param_dict['fignames'] = str(param_dict['N']) + '_' + str(n_train[t]) + '_' + str(n_rounds)
        apt_duration, mh_duration = main(param_dict) 
        timings[param_dict['fignames']] = [apt_duration, mh_duration]
           
    save_file(timings_file_name, timings)
#%%
def main(param_dict):
    
    file_paths = {
        'analytical': 'analytical.pk' ,
        'mh_samples': os.path.join(param_dict['folderName'], 'mh_samples' + '_' + param_dict['fignames'] + '.pk'),
        'posterior_apt': os.path.join(param_dict['folderName'], 'posterior_apt' + '_' + param_dict['fignames'] + '.pk')
    }
    
    what_to_do = {
        'analytical': False,
        'mh_samples': True,
        'apt': True
    }
    
    N = param_dict['N']
    err = param_dict['err'] # err to be fixed
    labels = ['slope', 'intercept']
    true_params = param_dict['true_params']
    hyps = param_dict['hyps']
    n_steps = param_dict['n_steps']
    
    fignames = os.path.join(param_dict['folderName'], param_dict['fignames'])
    
    seed_p = 2
    
    # prior_apt =  dd.Uniform(lower = np.asarray([-10, -10]), upper = np.asarray([10, 10]), seed = seed_p)
    
    np.random.seed(1234)
    
    x0 = np.random.uniform(0, 10, N)
    
    prior_analytical = {'mu_alpha': 10, 'var_alpha': 2, 'mu_beta': 2.5, 'var_beta': 10}
    
    if param_dict['analyticModel'] is None:
        analyticModel = analyticLineFittingToy.analyticalLinearNoise(x0, err, true_params, prior_analytical)
    else:
        analyticModel = param_dict['analyticModel']
        
    obs0 = {'y': analyticModel.y.reshape(-1)} 
    
    if hyps['m'] is None:
        apt_model = apt_general_2.linearNoiseModel(analyticModel.linearNoiseSimulator, labels, true_params, dim_param = 2)
        hyps['m'] = apt_model
        
    if hyps['s'] is None:
        s = apt_general_2.linearNoiseStats(x0)
        hyps['s'] = s
    
    
    number_of_samples = 400
    # bounds for the posterior label
    a_i, a_e, b_i, b_e = 0, number_of_samples, 0, number_of_samples
    
    alpha = np.linspace(-.1+true_params[0], .1+true_params[0], number_of_samples)
    beta = np.linspace(-.2+true_params[1], .2+true_params[1], number_of_samples)

    # true_params = [alpha_true, beta_true]
    # labels = ['slope', 'intercept']
    
    # this is for corner.histd to define the range of the x and y axis
    ranger = [[alpha[0], alpha[-1]], 
              [beta[0],  beta[-1]]]
    
    # prior for the true analytical line fitting problems
    prior_analytical = {'mu_alpha': 10, 'var_alpha': 2, 'mu_beta': 2.5, 'var_beta': 10}

    analytical = open_file(file_paths['analytical'])
    
    if analytical is -1 or what_to_do['analytical']:
        
        posterior = np.empty([np.size(beta), np.size(alpha)])
        for ib, b in enumerate(beta):
            for ia, a in enumerate(alpha):
                    t = [a, b]
                    posterior[ib][ia] = analyticModel.posteriorLinearNoise(t)
                
        posterior_1 = posterior - posterior.max()
        posterior_1[posterior_1 < -1000] = -1000
        
        analytical = {'posterior_1': posterior_1, 'alpha': alpha, 'beta': beta}
        
        save_file(file_paths['analytical'], analytical)
    else:
        posterior_1 = analytical['posterior_1']
        alpha = analytical['alpha']
        beta  = analytical['beta']
        
    mh_samples = open_file(file_paths['mh_samples'])

    if mh_samples is -1 or what_to_do['mh_samples']:
        
        np.random.seed(hyps['seed_inf'])
        nwalkers = 32
        p0 = np.random.randn(nwalkers, len(true_params))
        mh_start = time.time()
        # chain, _ ,acc_frac = mh_sampler.run_metropolis_hastings(p0, analyticModel, n_steps=n_steps, proposal_sigmas=[.5,.5])
        
        sampler = emcee.EnsembleSampler(nwalkers, len(true_params), analyticModel)
        sampler.run_mcmc(p0, n_steps, progress = False)
        mh_end = time.time()
        mh_samples = sampler.get_chain(discard=200, thin=15, flat=True)
        
        # print("Acceptance fraction: {:.1%}".format(acc_frac))
        mh_duration = mh_end - mh_start
        # mh_samples = chain[2000::8]
        print(file_paths['mh_samples'])
        save_file(file_paths['mh_samples'], mh_samples)
        
    apt_posterior = open_file(file_paths['posterior_apt'])
    
    if apt_posterior is -1 or what_to_do['apt']:
        apt_start = time.time()
        apt_posterior = apt_general_2.runAPT2LinearNoise(obs0, hyps, labels, true_params, fignames, plot = True)
        apt_end = time.time()
        apt_duration = apt_end - apt_start
        save_file(file_paths['posterior_apt'], apt_posterior)
    
    ## Plotting the Metropolis on top of the analytical posterior
    fig, ax = plt.subplots(2, 1, figsize=(5,5))
    
    posterior_2 = posterior_1[np.ix_(np.arange(b_i, b_e), np.arange(a_i, a_e) )]
    
    ax[0].contourf(alpha[a_i:a_e], beta[b_i:b_e], posterior_2, 
                   cmap='Blues', levels=100, vmin=posterior_2.max()-128, vmax=posterior_2.max())    
    ax[0].plot(true_params[0], true_params[1], marker = 'o', zorder=999, color='r')
    
    corner.hist2d(mh_samples[:, 0], mh_samples[:, 1], ax = ax[0],zorder=5, fill_contours=False, range = ranger)
    
    
    ax[1].contourf(alpha[a_i:a_e], beta[b_i:b_e], posterior_2, 
                   cmap='Blues', levels=100, vmin=posterior_2.max()-128, vmax=posterior_2.max())
    
    ax[1].plot(true_params[0], true_params[1], marker = 'o', zorder=999, color='r')

    apt_samples = apt_posterior[0].gen(1000)
    apt_samples_r = np.reshape(apt_samples, apt_samples.size, order='F').reshape(apt_samples.shape[1], apt_samples.shape[0])
    corner.hist2d(apt_samples_r[0], apt_samples_r[1], ax = ax[1], zorder=5, fill_contours=False, 
              range = ranger)
    plt.savefig(fignames + '_res.png')
    plt.close('all')
    return apt_duration, mh_duration
# %%
if __name__ == '__main__':
    # param_dict = baseParamInit()
    # main(param_dict)
    # selfConsistencyOverDataSizes()
    # compareDifferentDataSizes()
    selfConsistencyOverParamDraws()
    # compareDifferentParamDraws()
    # compareSelfConsistency()