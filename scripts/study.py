import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
from dataclasses import dataclass
from dataclasses import replace
import overcomplete_learning.run_study as ol_study

@dataclass
class StudyConfig:
    #data setup
    nreps                                   : int   = 2500      #sample size
    nobs                                    : int   = 4         #dim of X
    nsrc                                    : int   = 4         #dim of S
    S_scale                                 : float = 1         #scaling S
    A_scale                                 : float = 1         #scaling A
    
    #error options  
    err_sd                                  : float = 0.24      #scaling of error.
    sn_ratio                                : float = 100       #<1 low quality, >1 high quality
    correlation_type                        : str   = 'iid'     #Keep at i.i.d. error for now
    
    seed                                    : int   = 1          #error seed for reproducibility
    
    #EM options
    EM_iters                                : int   = 2500     
    convergence_threshold                   : float = 1e-5      
    outer_iters                             : int   = 10        #only used for LS-B-VEM+
    outer_tol                               : float = 1e-4      #only used for LS-B-VEM+
    mu                                      : float = 5         #only used for OverICA
    
    n_inits                                 : int   = 10        #initializations of random parameters
    n_data_sets                             : int   = 30        #number of random datasets
    
    method                                  : str   = 'standard' #estimation method. See overcomplete_learning/run_study.py for EM_METHOD_REGISTRY and 
    normalize_A_col                         : bool  = False         #normalizing A of unknown columns
    #whiten_data                             : bool = False      #prewhiten the data -> not fully implemnte
    #scale_source                            : bool = False     #not fully implemented
    
    #misc-setup
    known_src_step                          : int   = 1             #deciding how fine grained the known source steps should be
    n_workers                               : int   = None          #n. parallellized worker. if set to None, then it is set automatically to n core - 
    
    #Storing Data
    results_dir                             : str   = 'results_oracle/results'
    log_dir                                 : str   = 'results_oracle/logs'

if __name__ == '__main__':
    base_cfg = StudyConfig()
    os.makedirs(base_cfg.log_dir,     exist_ok=True)
    os.makedirs(base_cfg.results_dir, exist_ok=True)

    
    
    #additional looping for multiple studies
    nsrc_range =list(range(base_cfg.nsrc, 13,1))# + list(range(21,30,2))
    nreps_range = [base_cfg.nreps]
    methods = [base_cfg.method]   #['OLS', 'OLS_iid', 'fast_ICA', 'OverICA', 'random', 'standard_iid',  'standard', 'standard_wrong']
    snr_range = [base_cfg.sn_ratio] 
    for current_snr in snr_range:
        for current_nreps in nreps_range:
            for current_nsrc in nsrc_range:       #for method in ['OLS', 'OLS_iid', 'fast_ICA', 'sdp', 'random', 'standard_iid',  'standard', 'standard_wrong']:'OLS_debias', 
                for current_method in methods:            
                    
                    cfg = replace(
                        base_cfg, 
                        nreps = current_nreps,
                        nsrc=current_nsrc, 
                        method=current_method, 
                        sn_ratio = current_snr
                    )
                    #!!!----either fix sd or SNR!!!!
                    updated_err_sd = 0.45 #np.sqrt((current_nsrc * 2) / (base_cfg.nobs * cfg.sn_ratio))
                    #!!!-----
                    
                    cfg = replace(
                        cfg, 
                        err_sd=updated_err_sd,
                    )

                    
                    # 4. Build a distinct name descriptive of the updated configuration parameters
                    name = (f'em_study_nobs{cfg.nobs}_nsrc{cfg.nsrc}_n{cfg.nreps}_snratio{cfg.sn_ratio}'
                                f'_errsd{np.round(cfg.err_sd, 2)}_seed{cfg.seed}_n_datasets{cfg.n_data_sets}'
                                f'_method{cfg.method}_err{cfg.correlation_type}_normalizeA{cfg.normalize_A_col}_whiten_data{cfg.whiten_data}')
                    
                    if cfg.method == 'OverICA':
                        name = f'{name}' f'_mu{cfg.mu}'
                    
                    df = ol_study.run_study(
                        cfg                   = cfg,
                        known_src_range       = range(0, cfg.nsrc + 1, cfg.known_src_step),
                        n_workers             = cfg.n_workers,
                        csv_name              = f'{name}.csv',
                        results_dir           = cfg.results_dir,
                        log_file              = f'{cfg.log_dir}/{name}.log'
                    )
        
