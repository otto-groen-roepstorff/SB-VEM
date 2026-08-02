from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import overcomplete_learning.metrics    as ol_metrics
import overcomplete_learning.em         as ol_em
import overcomplete_learning.data       as ol_data
import overcomplete_learning.worker_logic as ol_worker
import overcomplete_learning.fastICA as ol_fast
import overcomplete_learning.semiblind_em as ol_semiblind

from overcomplete_learning.data         import EMData

import pandas                           as pd
import functools
import os
import logging
import logging.handlers
import multiprocessing
import numpy as np

GLOBAL_CFG = None
EM_METHOD_REGISTRY = {
    'random'       : lambda cfg: ol_em.run_EM_random,
    'OverICA'      : lambda cfg: ol_em.run_EM_OverICA,
    'fast_ICA'     : lambda cfg: ol_em.run_EM_FastICA,

    'OLS'          : lambda cfg: ol_em.run_EM_OLS,
    'standard'     : lambda cfg: ol_em.run_EM_extended,
    'semiblind'    : lambda cfg: functools.partial(
        ol_semiblind.run_EM_semiblind,
        lam=getattr(cfg, 'sb_lambda', 0.01),
        xi_thr=getattr(cfg, 'sb_xi_thr', 0.4),
        eta_scale=getattr(cfg, 'sb_eta_scale', 0.9),
        n_inner=getattr(cfg, 'sb_n_inner', 50),
        iid_noise=getattr(cfg, 'sb_iid_noise', True),
        order_correct=getattr(cfg, 'sb_order_correct', True),
    ),

    'OLS_iid'      : lambda cfg: ol_em.run_EM_OLS_iid(),
    'standard_iid' : lambda cfg: ol_em.run_EM_extended_iid(),

    'standard_wrong': lambda cfg: ol_em.run_EM_extended_wrong(),

    'OLS_debias': lambda cfg: ol_em.run_EM_OLS_debias(
        outer_iters=cfg.outer_iters,
        outer_tol=cfg.outer_tol,
    ),
    'OLS_debias_iid': lambda cfg: ol_em.run_EM_OLS_debias_iid(
        outer_iters=cfg.outer_iters,
        outer_tol=cfg.outer_tol,
    ),
    'OLS_oracle':lambda cfg: ol_em.run_EM_OLS_oracle
    
}


#---- LOGGING-----

# ── Module-level logger — used in main process ────────────────────────────────
logger = logging.getLogger(__name__)
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ── Logging helpers ───────────────────────────────────────────────────────────
def setup_logging(log_file: str = 'em_study.log',
                  level: int   = logging.INFO) -> None:
    """Configure root logger — call once in the main process."""
    logging.basicConfig(
        level   = level,
        format  = '%(asctime)s [%(levelname)s] %(processName)s | %(message)s',
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ]
    )


# ── EM wrapper ────────────────────────────────────────────────────────────────
def choose_alg(method, cfg):
    assert method in EM_METHOD_REGISTRY, f'please provide one of the following key word {EM_METHOD_REGISTRY.keys()}'
    return EM_METHOD_REGISTRY[method](cfg)
    
    

def run_em(data: EMData, cfg, log, n_known_src, rng) -> dict:
    """
    Runs different learning algorithms. Returns final estimates and convergence info.
    """
   
        
    sknown = ol_data.make_sknown(S = data.S, n_known_src=n_known_src)
   
   # --- PRE-PROCESSING LAYER ---
    if cfg.whiten_data:
        # Whiten observations: X_input shape becomes (nreps, cfg.nsrc)
        X_input, K, X_mean = ol_data.whiten_canonical(X=data.X, n_components=cfg.nsrc)
        
        # Extract the ground-truth physical scale from known sources
        global_real_scale = np.nanmean(np.abs(sknown))
        if np.isnan(global_real_scale) or global_real_scale == 0:
            global_real_scale = 1.0  # Fallback if no sources are known
            
        # Scale known sources down to match the unit-variance sandbox
        sknown_input = sknown / global_real_scale
    else:
        X_input = data.X
        sknown_input = sknown
        global_real_scale = 1.0
   
   
    f_em = choose_alg(cfg.method, cfg=cfg)
    
    out    = f_em(
        X               = X_input,
        S_known         = sknown_input,
        rng            =  rng,
        EM_iter         = cfg.EM_iters,
        err_tolerance   = cfg.convergence_threshold,
        whiten_data     = cfg.whiten_data,
        normalize_A_col = cfg.normalize_A_col,
        scale_source    = cfg.scale_source,
        mu              = cfg.mu,
        err_sd          = cfg.err_sd,
        Sigma           = cfg.err_sd**2,
        A_true          = data.A
    )
    
    # --- POST-PROCESSING / UN-WHITENING LAYER ---
    if cfg.whiten_data:
        # K shape: (nobs, nsrc) -> K_inv shape: (nsrc, nobs)
        K_inv = np.linalg.pinv(K)
        
        # 1. Un-whiten and Re-scale the Mixing Matrix A
        # out['A'] out of EM is shape (nsrc, nsrc)
        # K_inv.T shape is (nobs, nsrc)
        # Math: A_real = (K_inv.T @ A_whitened) / scale
        if 'A' in out:
            out['A'] = (K_inv.T @ out['A']) / global_real_scale
            
        # 2. Re-scale the Source Waveforms back to real-world amplitude
        # out['S'] shape is (nreps, nsrc)
        # Math: S_real = S_whitened * scale
        for s_key in ['S', 'post_mean']:
            if s_key in out:
                out[s_key] = out[s_key] * global_real_scale
                
        # 3. Store the centering mean vector for full X reconstruction
        out['X_mean'] = X_mean
    
    i = out['convergence_iteration']
    if i == cfg.EM_iters - 1:
        log.warning(f'Max iterations reached without convergence')
    else:
        log.debug(f'Converged at iteration {i} ')
    
    return out

# ── Worker function ───────────────────────────────────────────────────────────
def run_single_job(args):
    """
    Worker function for one (n_known_src, init, init_data) combination.
    Must be defined at module level for multiprocessing pickling.
    """
    import time, traceback
    cfg = ol_worker.GLOBAL_CFG
    n_known_src, init, init_data = args
    # Route this worker's logs through the queue
    log = logging.getLogger(__name__)
    log.info(f'START | n_known_src={n_known_src} | init={init:02d} | data={init_data}')

    t0 = time.perf_counter()
    try:
         # Generate a unique dataset per (init, init_data) combination
        data_seed = cfg.seed + init_data*10_000
        data_rng = np.random.default_rng(seed=data_seed)
        data = ol_data.generate_data(nreps= cfg.nreps, 
                             nobs = cfg.nobs, 
                             nsrc = cfg.nsrc, 
                             err_sd=cfg.err_sd, 
                             rng = data_rng,
                             correlation_type=cfg.correlation_type, S_scale=cfg.S_scale, A_scale=cfg.A_scale)
           
        sknown = ol_data.make_sknown(S = data.S, n_known_src=n_known_src)
        
        init_seed = cfg.seed + init*1_000
        init_rng = np.random.default_rng(init_seed)
        #run the naive EM procedure
        results = run_em(
            data=data, 
            cfg = cfg, 
            log=log, 
            n_known_src=n_known_src,
            rng = init_rng
        )
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        errors = ol_metrics.evaluate(data=data, results=results, sknown=sknown)
        n_unknown_src = data.A.shape[1] - n_known_src
        matrix_coherence = ol_data.coherence(data.A[:,n_known_src:]) if n_unknown_src>1 else 0
        log.info(
            f'DONE  | n_known_src={n_known_src} | init={init:02d} | '
            f"converged={results['converged']} | "
            f"iter={results['convergence_iteration']} | "
            f"A_err={errors['A_err']:.4f} | "#S_err={errors['S_err']:.4f} | "
            f"time={time.perf_counter() - t0:.1f}s"
        )

        out = {
            'n_known_src'           : n_known_src,
            'init'                  : init,
            'init_seed'             : init_seed,
            'init_data'             : init_data,
            'data_seed'             : data_seed,
            'converged'             : results['converged'],
            'convergence_iteration' : results['convergence_iteration'],
            'normalize_A'           : cfg.normalize_A_col,
            'sn_ratio'              : cfg.sn_ratio,
            'Au_coherence'          : matrix_coherence,
            'elapsed'               : elapsed,
            **errors
        }

        return out

    except Exception as e:
        log.error(
            f'FAILED | n_known_src={n_known_src} | init={init:02d} | '
            f'time={time.perf_counter() - t0:.1f}s | '
            f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        )
        raise

def run_study(cfg, known_src_range, n_workers = None, 
              results_dir: str ='results', csv_name: str ='em_study.csv',
              log_file: str ='em_study.log') -> pd.DataFrame:
    """
    Run the full study over known_src_range x n_inits x n_data_sets in parallel.
    All worker logs are routed through a queue to the main process.
    """
    
    setup_logging(log_file=log_file)
    # ── Logging queue — workers send records here, main process writes them ───
    # Shared logging queue — workers write here, main process flushes to file
    ctx = multiprocessing.get_context("spawn")
    log_queue = ctx.Queue()
    listener  = logging.handlers.QueueListener(
        log_queue, *logging.getLogger().handlers,
        respect_handler_level=True
    )
    listener.start()

    # ── Build job list — pass queue to each worker ────────────────────────────
    all_args = [
        (n_known_src, init, init_data)
        for n_known_src in known_src_range
        for init        in range(cfg.n_inits)
        for init_data   in range(cfg.n_data_sets)
    ]
    
    max_comp =  os.cpu_count()-1
    n_workers = n_workers or max(1, max_comp)
    logger.info(f'Total jobs: {len(all_args)} | Workers: {n_workers}')

    # ── Run in parallel ───────────────────────────────────────────────────────
    records = []
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=ctx,
                             initializer=ol_worker.init_worker,
                             initargs=(cfg, log_queue), 
                             ) as executor:
        futures = {executor.submit(run_single_job, args): args
                   for args in all_args}
        for future in tqdm(as_completed(futures), total=len(all_args)):
            try:
                records.append(future.result())
            except Exception as e:
                args = futures[future]
                logger.error(
                    f'Job failed | n_known={args[1]} init={args[2]} | '
                    f'{type(e).__name__}: {e}'
                )

    
    listener.stop()   # flush and close the listener cleanly

    # ── Build and save dataframe ──────────────────────────────────────────────
    df = pd.DataFrame(records)
    df['nsrc'] = cfg.nsrc #data.A.shape[1]
    df['nobs'] = cfg.nobs #data.A.shape[0]
    df['sd_err'] = cfg.err_sd
    df['nreps'] = cfg.nreps
    df['method'] = cfg.method
    df['err_correlation_type'] = cfg.correlation_type
    df['whiten_data'] = cfg.whiten_data 
    df['mu']    = cfg.mu

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, csv_name)
    df.to_csv(path, index=False)
    logger.info(f'Saved {len(df)} rows to {path}')
    return df