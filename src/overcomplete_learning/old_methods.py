import numpy as np
import matplotlib.pyplot as plt

    
def plot_mse_by_nsrc(df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    for nsrc_val, group in df.groupby('nsrc'):
        group_sorted = group.sort_values('nonsparsity')
        ax.plot(group_sorted['nonsparsity'], group_sorted['mse'],
                marker='o', linewidth=2, label=f'nsrc={nsrc_val}')

    ax.set_xlabel('Nonsparsity factor', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Recovery MSE by sparsity level and number of sources', fontsize=13)
    ax.legend(title='nsrc', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f'Figure saved to {save_path}')

    plt.show()
    
def plot_mse_recovery(df, save_path=None, log_scale=False):
    nobs_vals = sorted(df['nobs'].unique())
    nsrc_vals = sorted(df['nsrc'].unique())

    colours    = cm.viridis(np.linspace(0.1, 0.9, len(nsrc_vals)))
    colour_map = dict(zip(nsrc_vals, colours))

    linestyle_map = {True: '-', False: '--'}
    label_map     = {True: 'n non-zero src known', False: 'n non-zero nsrc unknown'}

    # Back to one panel per nobs
    fig, axes = plt.subplots(
        nrows=len(nobs_vals), ncols=1,
        figsize=(8, 4 * len(nobs_vals)),
        sharex=True, sharey=True)

    if len(nobs_vals) == 1:
        axes = [axes]

    for ax, nobs_val in zip(axes, nobs_vals):
        subset = df[df['nobs'] == nobs_val]
        k0     = int(subset['k0'].unique()[0])

        for nsrc_val in nsrc_vals:
            for known, ls in linestyle_map.items():
                group = (subset[
                            (subset['nsrc'] == nsrc_val) &
                            (subset['known_n_src'] == known)]
                         .sort_values('n_nonzeroes'))

                if group.empty:
                    continue

                colour = colour_map[nsrc_val]
                label  = f'nsrc={nsrc_val} ({label_map[known]})'

                # MSE line
                ax.plot(group['n_nonzeroes'], group['mse'],
                        marker='o', linewidth=2, markersize=6,
                        color=colour, linestyle=ls, label=label)

                # Warning shading
                warn_mask = group['w'] > 0
                if warn_mask.any():
                    ax.fill_between(
                        group['n_nonzeroes'],
                        group['mse'],
                        alpha=0.15,
                        color=colour,
                        where=warn_mask.values,
                        label=f'nsrc={nsrc_val} (OMP warnings)')

        # Recovery bound
        ax.axvline(x=k0, color='red', linestyle=':', linewidth=1.5,
                   label=f'Recovery bound (k0={k0})')

        ax.set_title(f'nobs={nobs_val} — guaranteed recovery at k0={k0}',
                     fontsize=12)
        ax.set_ylabel('MSE', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        if log_scale:
            ax.set_yscale('log')

    axes[-1].set_xlabel('Number of non-zero elements', fontsize=11)
    fig.suptitle('Sparse recovery MSE vs sparsity level', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        base = save_path.replace('.png', '')
        fig.savefig(f'{base}_linear.png', dpi=150, bbox_inches='tight')
        print(f'Saved: {base}_linear.png')
        #for ax in axes:
        #    ax.set_yscale('log')
        #fig.savefig(f'{base}_log.png', dpi=150, bbox_inches='tight')
        #print(f'Saved: {base}_log.png')

    plt.show()
    
    
def make_plot_2(df1, x_col, title_suffix, filename, df2=None, df1_label='Default', df2_label='Comparison'):
    """
    x_col:      'n_known_src' or 'n_known_columns'
    df2:        optional second dataframe to compare against
    df1_label:  legend label for df1
    df2_label:  legend label for df2
    """
    # ── Select iterations ─────────────────────────────────────────────────────
    all_iters   = sorted(df1['iteration'].unique())
    first_iter  = all_iters[0]
    median_iter = all_iters[len(all_iters) // 2]
    max_iter    = all_iters[-1]
    plot_iters  = [first_iter, median_iter, max_iter]
    print(plot_iters)
    iter_labels = ['First iteration', 'Median iteration', 'Last iteration']

    other_col = 'n_known_src' if x_col == 'n_known_columns' else 'n_known_columns'

    # ── Colors ────────────────────────────────────────────────────────────────
    color_matrix_1 = 'steelblue'
    color_latent_1 = 'coral'
    color_matrix_2 = 'navy'
    color_latent_2 = 'firebrick'
    color_omp_1    = 'darkorange'
    color_omp_2    = 'saddlebrown'

    def get_grouped(df, it):
        sub = df[(df['iteration'] == it) & (df[other_col] == 0)]
        cols = ['err_matrix', 'err_latent']
        if 'OMP_error' in df.columns:
            cols.append('OMP_error')
        return sub.groupby(x_col)[cols].mean().reset_index()

    # ── Pre-compute global y-limits ───────────────────────────────────────────
    all_latent_vals, all_matrix_vals = [], []
    for it in plot_iters:
        g1 = get_grouped(df1, it)
        all_latent_vals.extend(g1['err_latent'].values)
        all_matrix_vals.extend(g1['err_matrix'].values)
        # Include OMP in the latent axis limits since they share the same scale
        if 'OMP_error' in df1.columns:
            all_latent_vals.extend(g1['OMP_error'].dropna().values)
        if df2 is not None:
            g2 = get_grouped(df2, it)
            all_latent_vals.extend(g2['err_latent'].values)
            all_matrix_vals.extend(g2['err_matrix'].values)
            if 'OMP_error' in df2.columns:
                all_latent_vals.extend(g2['OMP_error'].dropna().values)

    latent_max = 8#max(all_latent_vals) * 1.05
    matrix_max = max(all_matrix_vals) * 1.05

    # ── Check for optional columns ────────────────────────────────────────────
    has_coherence = 'current_coherence' in df1.columns
    has_omp       = 'OMP_error' in df1.columns
    if df2 is not None:
        has_coherence = has_coherence and 'current_coherence' in df2.columns
        has_omp       = has_omp       and 'OMP_error' in df2.columns

    coherence_offset = 60

    fig, axes = plt.subplots(1, len(plot_iters), figsize=(22, 5))
    fig.suptitle(f'EM convergence by {title_suffix}', fontsize=14, fontweight='bold')

    for ax, it, label in zip(axes, plot_iters, iter_labels):
        g1 = get_grouped(df1, it)

        # ── Left y-axis: err_matrix ───────────────────────────────────────────
        ax.plot(g1[x_col], g1['err_matrix'],
                color=color_matrix_1, marker='o',
                label=f'err_matrix ({df1_label})')
        ax.set_ylim(0, matrix_max)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel('err_matrix (Frobenius)', color=color_matrix_1, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color_matrix_1)
        ax.grid(True, alpha=0.3)

        # ── Right y-axis: err_latent + OMP (shared scale) ─────────────────────
        ax2 = ax.twinx()
        ax2.plot(g1[x_col], g1['err_latent'],
                 color=color_latent_1, marker='s', linestyle='--',
                 label=f'err_latent ({df1_label})')
        ax2.set_ylim(0, latent_max)
        ax2.set_ylabel('err_latent / OMP error (MSE)', color='black', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='black')

        # OMP plotted on ax2 directly — same scale as err_latent
        if has_omp:
            ax2.plot(g1[x_col], g1['OMP_error'],
                     color=color_omp_1, marker='D', linestyle='-.',
                     label=f'OMP error ({df1_label})')

        # ── Second dataframe ──────────────────────────────────────────────────
        if df2 is not None:
            g2 = get_grouped(df2, it)
            ax.plot(g2[x_col], g2['err_matrix'],
                    color=color_matrix_2, marker='o', linestyle=':',
                    label=f'err_matrix ({df2_label})')
            ax2.plot(g2[x_col], g2['err_latent'],
                     color=color_latent_2, marker='s', linestyle=':',
                     label=f'err_latent ({df2_label})')
            if has_omp:
                ax2.plot(g2[x_col], g2['OMP_error'],
                         color=color_omp_2, marker='D', linestyle=':',
                         label=f'OMP error ({df2_label})')

        # ── Coherence axis ────────────────────────────────────────────────────
        lines3, labels3 = [], []
        if has_coherence:
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', coherence_offset))
            coh1 = df1[(df1['iteration'] == it) & (df1[other_col] == 0)] \
                       .groupby(x_col)['current_coherence'].mean().reset_index()
            ax3.plot(coh1[x_col], coh1['current_coherence'],
                     color='seagreen', marker='^', linestyle='-.',
                     label=f'Coherence ({df1_label})')
            ax3.set_ylim(0, 1)
            if df2 is not None:
                coh2 = df2[(df2['iteration'] == it) & (df2[other_col] == 0)] \
                           .groupby(x_col)['current_coherence'].mean().reset_index()
                ax3.plot(coh2[x_col], coh2['current_coherence'],
                         color='darkgreen', marker='^', linestyle=':',
                         label=f'Coherence ({df2_label})')
            ax3.set_ylabel('Coherence', color='seagreen', fontsize=10)
            ax3.tick_params(axis='y', labelcolor='seagreen')
            lines3, labels3 = ax3.get_legend_handles_labels()

        ax.set_title(f'{label} (iter={it})', fontsize=11)

        # ── Combined legend ───────────────────────────────────────────────────
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        #ax.legend(lines1 + lines2 + lines3,
        #          labels1 + labels2 + labels3,
        #          fontsize=7, loc='best')
    # Option 2: place legend outside the plot (most reliable)
        ax.legend(lines1 + lines2 + lines3,
                labels1 + labels2 + labels3,
                fontsize=7,
                loc='upper left',
                bbox_to_anchor=(1.35, 1),  # push right of the coherence axis
                borderaxespad=0)

    plt.tight_layout()
    plt.savefig(f'../results/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to results/{filename}")
    
def make_plot_legacy(df, x_col, title_suffix, filename):
    """
    x_col: 'n_known_src' or 'n_known_columns'
    """
    # ── Select iterations ─────────────────────────────────────────────────────
    all_iters  = sorted(df['iteration'].unique())
    #initial_iter =all_iters[0] 
    
    first_iter  = all_iters[0]
    median_iter = all_iters[len(all_iters) // 2]
    max_iter    = all_iters[-1]
    plot_iters  = [
        #initial_iter, 
        first_iter, median_iter, max_iter]
    iter_labels = [#'Initialization', 
                   'First iteration', 'Median iteration', 'Last iteration']

    fig, axes = plt.subplots(1, len(iter_labels), figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle(f'EM convergence by {title_suffix}', fontsize=14, fontweight='bold')

    for ax, it, label in zip(axes, plot_iters, iter_labels):
        # Filter to rows where the OTHER known variable is 0
        # (so we only vary the x_col of interest)
        other_col = 'n_known_src' if x_col == 'n_known_columns' else 'n_known_columns'
        sub = df[(df['iteration'] == it) & (df[other_col] == 0)]

        # Average over seeds
        grouped = sub.groupby(x_col)[['err_matrix', 'err_latent']].mean().reset_index()

        # ── Left y-axis: err_matrix ───────────────────────────────────────────
        color_matrix = 'steelblue'
        color_latent = 'coral'

        ax.plot(grouped[x_col], grouped['err_matrix'],
                color=color_matrix, marker='o', label='err_matrix (Frobenius)')
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel('err_matrix (Frobenius)', color=color_matrix, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color_matrix)

        # ── Right y-axis: err_latent ──────────────────────────────────────────
        ax2 = ax.twinx()
        ax2.plot(grouped[x_col], grouped['err_latent'],
                 color=color_latent, marker='s', linestyle='--', label='err_latent (MSE)')
        ax2.set_ylabel('err_latent (MSE)', color=color_latent, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=color_latent)

        ax.set_title(f'{label} (iter={it})', fontsize=11)
        ax.grid(True, alpha=0.3)

        # ── Combined legend ───────────────────────────────────────────────────
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

    plt.tight_layout()
    #plt.savefig(f'results/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to results/{filename}")

def plot_em_convergence(df, title_suffix='', ylim_s=None, ylim_s_np=None,
                        ylim_f=None, ylim_x=None, ylim_like = None, convergence_threshold=0.95):
    """
    Plots EM convergence curves from a records dataframe.
    One row of 4 subplots per unique value of n_known_src.
    Shows mean across initialisations with ±1 std shaded band.
    Vertical lines mark iterations where any initialisation dropped out.
    X-axis is clipped at the iteration where convergence_threshold fraction
    of initialisations have converged.
    All columns share the same y-axis scale across rows.

    Parameters
    ----------
    df                   : DataFrame with columns [init, iteration, n_known_src,
                                                   err_s, err_s_non_perm, err_f,
                                                   err_x, err_0_s, err_0_A, err_0_x]
    title_suffix         : optional string appended to suptitle
    ylim_s               : (min, max) for permuted source error    — auto if None
    ylim_s_np            : (min, max) for non-permuted source error — auto if None
    ylim_f               : (min, max) for matrix error             — auto if None
    ylim_x               : (min, max) for reconstruction error     — auto if None
    convergence_threshold: fraction of initialisations that must have converged
                           before clipping the x-axis (default 0.95)
    """
    known_src_values = sorted(df['n_known_src'].unique())
    n_rows           = len(known_src_values)

    # ── Compute global y-limits columnwise if not provided ────────────────────
    def global_ylim(col, extra_col=None, prop = 99):
        vals = df[col].dropna().values
        if extra_col is not None:
            vals = np.concatenate([vals, df[extra_col].dropna().values])

        #lower limit
        res_max = np.percentile(vals, prop) * 1.05
        res_min = np.percentile(vals, 100-prop)*1.05
        #res = np.percentile(vals, prop) * 1.05
        return (min(res_min,0), max(res_max, 0) )

    ylim_s    = ylim_s    or global_ylim('err_s',          extra_col='err_0_s')
    ylim_s_np = ylim_s_np or global_ylim('err_s_non_perm', extra_col='err_0_s')
    ylim_f    = ylim_f    or global_ylim('err_f',          extra_col='err_0_A')
    ylim_x    = ylim_x    or global_ylim('err_x',          extra_col='err_0_x')
    ylim_like    = ylim_like    or global_ylim('obs_likelihood', prop = 100)

    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    def get_convergence_xlim(sub, threshold):
        """
        Returns the iteration at which `threshold` fraction of initialisations
        have converged, defined as the iteration of their last recorded entry.
        """
        n_inits_total = sub['init'].nunique()
        n_must_converge = int(np.ceil(threshold * n_inits_total))

        # Last recorded iteration per initialisation = convergence iteration
        last_iter_per_init = sub.groupby('init')['iteration'].max().sort_values()

        if len(last_iter_per_init) < n_must_converge:
            return sub['iteration'].max()

        # The iteration at which the n_must_converge-th init finishes
        cutoff_iter = last_iter_per_init.iloc[n_must_converge - 1]
        return cutoff_iter

    def plot_with_errbar(ax, x, mean_vals, std_vals, label, color):
        ax.plot(x, mean_vals, label=label, color=color)
        ax.fill_between(
            x,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.2, color=color
        )

    def add_dropout_lines(ax, counts):
        """
        Draw a short vertical line at iterations where the number of
        active initialisations drops compared to the previous iteration.
        """
        dropout_iters = counts.index[counts.diff() > 0]
        ymin, ymax    = ax.get_ylim()
        line_height   = (ymax - ymin) * 0.08

        for it in dropout_iters:
            n_dropped = int(-counts.diff()[it])
            ax.vlines(
                x=it,
                ymin=ymin,
                ymax=ymin + line_height,
                color='red',
                linewidth=1.0,
                alpha=0.7,
                label='_nolegend_'
            )
            if n_dropped > 1:
                ax.text(
                    it, ymin + line_height * 1.1,
                    str(n_dropped),
                    color='red', fontsize=6,
                    ha='center', va='bottom'
                )

    def pad_to_k_fast(df, group_cols, iter_col, k):
        # Get actual group combinations
        groups = df[group_cols].drop_duplicates()

        # Create full index manually
        full_tuples = [
            (*row, it)
            for row in groups.to_numpy()
            for it in range(k)
        ]

        full_index = pd.MultiIndex.from_tuples(
            full_tuples,
            names=[*group_cols, iter_col]
        )

        # Reindex + forward fill
        df = df.set_index(group_cols + [iter_col])
        df = df.reindex(full_index)
        df = df.groupby(level=group_cols).ffill()
        return df.reset_index() 
    max_k = df['iteration'].max()
    

    df = pad_to_k_fast(df = df,group_cols=['init', 'n_known_src'], iter_col='iteration', k = max_k) #padding df to make sure the mse does not increase by removing the best runs
    for row, n_known in enumerate(known_src_values):

        sub     = df[df['n_known_src'] == n_known]

        # ── Compute x cutoff for this n_known_src ─────────────────────────────
        x_cutoff = get_convergence_xlim(sub, convergence_threshold)
        sub      = sub[sub['iteration'] <= x_cutoff]   # clip to cutoff

        grouped  = sub.groupby('iteration')
        mean     = grouped[['err_s', 'err_0_s', 'err_f', 'err_0_A',
                             'err_x', 'err_0_x', 'err_s_non_perm', 'obs_likelihood']].mean()
        std      = grouped[['err_s', 'err_0_s', 'err_f', 'err_0_A',
                             'err_x', 'err_0_x', 'err_s_non_perm','obs_likelihood']].std()
        
        counts   = grouped['converged'].sum()
        x        = mean.index
        
        conv_percentage = grouped['converged'].mean()
        
        iter_stop = np.where(conv_percentage<convergence_threshold)[0][-1]
        x_conv_threshold = iter_stop+1
        x_cutoff= x_conv_threshold #setting the x_limit to be where convergence_threshold percentage of iterations converged.
        xlim     = (0, x_cutoff)

        pct_label = f'{int(convergence_threshold * 100)}% converged at iter {x_cutoff}'

        # ── MSE error S (permuted) ────────────────────────────────────────────
        plot_with_errbar(axes[row, 0], x, mean['err_s'],   std['err_s'],
                         'MSE error S',         'steelblue')
        plot_with_errbar(axes[row, 0], x, mean['err_0_s'], std['err_0_s'],
                         'Baseline (always 0)', 'grey')
        axes[row, 0].axvline(x_cutoff, color='black', linestyle=':', linewidth=1.0,
                             label=pct_label)
        axes[row, 0].set_title(f'Source error (permuted) | n_known_src={n_known}')
        axes[row, 0].set_xlabel('Iteration')
        axes[row, 0].set_ylim(*ylim_s)
        axes[row, 0].set_xlim(*xlim)
        axes[row, 0].legend(loc='best', fontsize=7)
        axes[row, 0].grid(True, alpha=0.3)
        add_dropout_lines(axes[row, 0], counts)


        # ── Frobenius error ───────────────────────────────────────────────────
        
        matrix_err = df['err_f_name'].unique()[0]

        plot_with_errbar(axes[row, 1], x, mean['err_f'],   std['err_f'],
                         f'{matrix_err} error',     'coral')
        plot_with_errbar(axes[row, 1], x, mean['err_0_A'], std['err_0_A'],
                         'Baseline (always 0)', 'grey')
        axes[row, 1].axvline(x_cutoff, color='black', linestyle=':', linewidth=1.0,
                             label=pct_label)
        axes[row, 1].set_title(f'Matrix error | n_known_src={n_known}')
        axes[row, 1].set_xlabel('Iteration')
        axes[row, 1].set_ylim(*ylim_f)
        axes[row, 1].set_xlim(*xlim)
        axes[row, 1].legend(loc='best', fontsize=7)
        axes[row, 1].grid(True, alpha=0.3)
        add_dropout_lines(axes[row, 1], counts)

        # ── MSE error X ───────────────────────────────────────────────────────
        plot_with_errbar(axes[row, 2], x, mean['err_x'],   std['err_x'],
                         'MSE error X',         'seagreen')
        plot_with_errbar(axes[row, 2], x, mean['err_0_x'], std['err_0_x'],
                         'Baseline (always 0)', 'grey')
        axes[row, 2].axvline(x_cutoff, color='black', linestyle=':', linewidth=1.0,
                             label=pct_label)
        axes[row, 2].set_title(f'Reconstruction error | n_known_src={n_known}')
        axes[row, 2].set_xlabel('Iteration')
        axes[row, 2].set_ylim(*ylim_x)
        axes[row, 2].set_xlim(*xlim)
        axes[row, 2].legend(loc='best', fontsize=7)
        axes[row, 2].grid(True, alpha=0.3)
        add_dropout_lines(axes[row, 2], counts)

        #---- Likelihood X --------
        plot_with_errbar(axes[row, 3], x, mean['obs_likelihood'],   std['obs_likelihood'],
                         'Likelihood X',         'seagreen')
        axes[row, 3].axvline(x_cutoff, color='black', linestyle=':', linewidth=1.0,
                             label=pct_label)
        axes[row, 3].set_title(f'Estimated Log-Likelihood | n_known_src={n_known}')
        axes[row, 3].set_xlabel('Iteration')
        axes[row, 3].set_ylim(*ylim_like)
        axes[row, 3].set_xlim(*xlim)
        axes[row, 3].legend(loc='best', fontsize=7)
        axes[row, 3].grid(True, alpha=0.3)
        add_dropout_lines(axes[row, 3], counts)
        
        
        # # ── MSE error S (non-permuted) ────────────────────────────────────────
        # plot_with_errbar(axes[row, 3], x, mean['err_s_non_perm'], std['err_s_non_perm'],
        #                  'MSE error S non permuted', 'steelblue')
        # plot_with_errbar(axes[row, 3], x, mean['err_0_s'],        std['err_0_s'],
        #                  'Baseline (always 0)',       'grey')
        # axes[row, 3].axvline(x_cutoff, color='black', linestyle=':', linewidth=1.0,
        #                      label=pct_label)
        # axes[row, 3].set_title(f'Source error (non-permuted) | n_known_src={n_known}')
        # axes[row, 3].set_xlabel('Iteration')
        # axes[row, 3].set_ylim(*ylim_s_np)
        # axes[row, 3].set_xlim(*xlim)
        # axes[row, 3].legend(loc='best', fontsize=7)
        # axes[row, 3].grid(True, alpha=0.3)
        # add_dropout_lines(axes[row, 3], counts)

    title = f'EM convergence {title_suffix}'.strip()
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_EM_results(df):
    """
    Plot EM errors vs number of known sources.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        ['n_known_src', 'init', 'A_err', 'S_err', 'X_err', 'nsrc']
    """

    # -------------------------------------------------
    # Step 1: Copy to avoid modifying original
    # -------------------------------------------------
    df = df.copy()

    # -------------------------------------------------
    # Step 2: Fix S_err when all sources are known
    # -------------------------------------------------
    mask_full_known = df["n_known_src"] == df["nsrc"]
    df.loc[mask_full_known, "S_err"] = 0.0

    # -------------------------------------------------
    # Step 3: Group by n_known_src
    # -------------------------------------------------
    grouped = df.groupby("n_known_src")

    # Compute mean and std
    stats_mean = grouped[["A_err", "S_err", "X_err"]].mean()
    stats_std = grouped[["A_err", "S_err", "X_err"]].std()

    # x-axis
    x = stats_mean.index.values

    # -------------------------------------------------
    # Step 4: Plot
    # -------------------------------------------------
    plt.figure(figsize=(8, 5))

    error_types = ["A_err", "S_err", "X_err"]

    for err in error_types:
        y = stats_mean[err].values
        yerr = stats_std[err].values

        plt.errorbar(
            x, y, yerr=yerr,
            marker='o',
            capsize=4,
            label=err
        )

    # -------------------------------------------------
    # Step 5: Formatting
    # -------------------------------------------------
    plt.xlabel("Number of known sources")
    plt.ylabel("Error")
    plt.title("EM Performance vs Known Sources")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def plot_errors(df, ax=None, label_prefix=""):
    if ax is None:
        ax = plt.gca()

    for col in df.columns:
        ax.plot(df.index, df[col], label=f"{label_prefix} {col}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")



def plot_compare(df1, df2, labels=("Naive", "EM"), abline = None):
    fig, ax = plt.subplots()

    # Use consistent color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, col in enumerate(df1.columns):
        color = colors[i % len(colors)]

        # First dataframe → solid line
        ax.plot(df1.index, df1[col],
                linestyle='-',
                color=color,
                label=f"{labels[0]} {col}")

        # Second dataframe → dashed line
        ax.plot(df2.index, df2[col],
                linestyle='--',
                color=color,
                label=f"{labels[1]} {col}")

    if not abline is None:
        ax.axhline(y = abline,ls = '--', c ='black')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title("Naive vs EM")
    ax.legend()

    plt.show()


def plot_EM_results_grid(df, err_type, only_best_init=True, best_init_metric='X_err', 
                         max_val=None, min_val=None, plot_individual_runs=False):
    """
    Plot EM results in a grid over (nobs, nsrc).

    Each subplot:
    - x-axis: n_unknown (nsrc - n_known_src)
    - y-axis: mean error
    - error bars: std
    - optional faint lines: individual initializations/runs
    """

    df = df.copy()

    # -------------------------------------------------
    # Unique experiment settings
    # -------------------------------------------------
    groups = df.groupby(["nobs", "nsrc", "err_correlation_type", "nreps"])
    keys = list(groups.groups.keys())

    n_plots = len(keys)

    # -------------------------------------------------
    # Create grid
    # -------------------------------------------------
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    if n_cols == 1 and n_rows == 1:
        size_multiplier = 2
    else:
        size_multiplier = 1
        
    if lower_is_better(metric=err_type):
        legend_location = 'upper left'
        legend_anchor = (0, 0.9)
    else:
        legend_location = "upper right"
        legend_anchor = (1, 0.9)
        
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(5 * n_cols * size_multiplier, 4 * n_rows * size_multiplier),
                             squeeze=False)
    axes = axes.reshape(-1)

    # 1. Define a unified master style registry
    color_map     = {k: v["color"] for k, v in METHOD_STYLES.items()}
    linestyle_map = {k: v["linestyle"] for k, v in METHOD_STYLES.items()}
    marker_map    = {k: v["marker"] for k, v in METHOD_STYLES.items()}
    
    # -------------------------------------------------
    # Loop over each (nobs, nsrc)
    # -------------------------------------------------
    for ax, (nobs, nsrc, err_correlation_type, nreps) in zip(axes, keys):

        sub_df = groups.get_group((nobs, nsrc, err_correlation_type, nreps))

        # --- Aggregate over n_known_src and method ---
        # ---------------------------------------------
        # Step 1: pick best configuration per init
        group_cols = ["init_data", "method", "n_known_src"]
        valid_methods = sub_df.groupby("method")[best_init_metric].apply(lambda x: x.notna().all())
        valid_methods = valid_methods[valid_methods].index
        
        if only_best_init:
            eligible = sub_df[sub_df["method"].isin(valid_methods)]
            
            best_per_init = eligible.loc[
                eligible.groupby(group_cols)[best_init_metric].idxmin()
            ]
            
            # Step 2: aggregate across initializations
            grouped = best_per_init.groupby(["n_known_src", "method"])
            methods = best_per_init['method'].unique()
            source_df = best_per_init  # Reference for individual lines
        else: 
            grouped = sub_df.groupby(["n_known_src", "method"])
            methods = sub_df["method"].unique()
            source_df = sub_df  # Reference for individual lines

        stats_mean = grouped[err_type].mean()
        stats_std = grouped[err_type].std()

        # -------------------------------------------------
        # Loop over methods separately 
        # -------------------------------------------------
        for method in methods:
            
            # --- Optional Faint Individual Lines ---
            if plot_individual_runs:
                method_ind_data = source_df[source_df["method"] == method]
                
                for init_val, init_df in method_ind_data.groupby("init_data"):
                    # Sort to prevent zig-zagging plotting anomalies
                    init_df = init_df.sort_values("n_known_src")
                    x_ind = nsrc - init_df["n_known_src"].values
                    y_ind = init_df[err_type].values
                    
                    ax.plot(
                        x_ind,
                        y_ind,
                        color=color_map[method],
                        linestyle=linestyle_map[method],
                        alpha=0.1,
                        linewidth=1.5,
                        zorder=1  # Keep in background
                    )

            # --- Mean and Std Components ---
            method_mean = stats_mean.xs(method, level="method")
            method_std = stats_std.xs(method, level="method")

            x = nsrc - method_mean.index.values  # convert to n_unknown

            y     = method_mean.values
            y_std = method_std.values
            
            # Plot mean line
            ax.plot(
                x,
                y,
                marker=marker_map[method],
                color=color_map[method],
                linestyle=linestyle_map[method],
                label=f"{metric_label(err_type)} ({method_label(method)})",
                linewidth=2,
                zorder=3  # Ensure mean is on top
            )
            
            ## Shaded region (mean ± std)
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color_map[method],
                alpha=0.1,
                zorder=2  # Layer std fill between faint lines and mean line
            )

        # -------------------------------------------------
        # Axis Limits and Formatting
        # -------------------------------------------------
        upper_bound = max_val if max_val else metric_bounds(err_type)[1] * 1.05
        lower_bound = min_val if min_val else metric_bounds(err_type)[0] * 0.95
        
        ax.set_ylim(bottom=lower_bound, top=upper_bound)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # sd_err fix
        sd_err = sub_df['sn_ratio'].unique()[0]
        nreps_vals = sub_df['nreps'].unique()
        nreps_str = ", ".join(map(str, nreps_vals))

        # --- Title & Labels ---
        ax.set_title(f"Observed Dimension: {nobs}\nLatent Dimension: {nsrc}\nSample Size: {nreps_str}")
        ax.set_xlabel(r"Number of unknown sources ($n_{\mathrm{u}}$)")
        ax.set_ylabel(f'{metric_label(err_type)}')
        
        # Mark the complete case boundary line
        ax.axvline(nobs, color="#FF3D00", linestyle=":", alpha=0.7, linewidth=2, zorder=4)
        ax.text(nobs + 0.2, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, 
                "↑ Complete Case Boundary", color="#FF3D00", fontweight="bold", rotation=90, alpha=0.8)
        
        ax.grid(True)

    # -------------------------------------------------
    # Remove unused subplots
    # -------------------------------------------------
    for i in range(len(keys), len(axes)):
        fig.delaxes(axes[i])

    # -------------------------------------------------
    # Shared legend
    # -------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_location, bbox_to_anchor=legend_anchor)
    plt.tight_layout()
    plt.show()
      
      
      
def plot_EM_results_boxplot(df, err_type, only_best_init=True, best_init_metric='X_err', 
                         max_val=None, min_val=None, plot_individual_runs=False):
    """
    Plot EM results in a grid over (nobs, nsrc) using box plots.

    Each subplot:
    - x-axis: n_unknown (nsrc - n_known_src)
    - y-axis: boxplot of the errors across initializations/runs
    - optional faint lines: individual initializations/runs
    """

    df = df.copy()

    # -------------------------------------------------
    # Unique experiment settings
    # -------------------------------------------------
    groups = df.groupby(["nobs", "nsrc", "err_correlation_type", "nreps"])
    keys = list(groups.groups.keys())

    n_plots = len(keys)

    # -------------------------------------------------
    # Create grid
    # -------------------------------------------------
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    if n_cols == 1 and n_rows == 1:
        size_multiplier = 2
    else:
        size_multiplier = 1
        
    if lower_is_better(metric=err_type):
        legend_location = 'upper right'
        legend_anchor = (1, 0.9)
    else:
        legend_location = "upper right"
        legend_anchor = (1, 0.9)
        
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(5 * n_cols * size_multiplier, 4 * n_rows * size_multiplier),
                             squeeze=False)
    axes = axes.reshape(-1)

    # 1. Define a unified master style registry
    color_map = {k: v["color"] for k, v in METHOD_STYLES.items()}
    
    # -------------------------------------------------
    # Loop over each (nobs, nsrc)
    # -------------------------------------------------
    for ax, (nobs, nsrc, err_correlation_type, nreps) in zip(axes, keys):

        sub_df = groups.get_group((nobs, nsrc, err_correlation_type, nreps))

        # --- Aggregate over n_known_src and method ---
        # Step 1: pick best configuration per init
        group_cols = ["init_data", "method", "n_known_src"]
        valid_methods = sub_df.groupby("method")[best_init_metric].apply(lambda x: x.notna().all())
        valid_methods = valid_methods[valid_methods].index
        
        if only_best_init:
            eligible = sub_df[sub_df["method"].isin(valid_methods)]
            best_per_init = eligible.loc[
                eligible.groupby(group_cols)[best_init_metric].idxmin()
            ]
            methods = best_per_init['method'].unique()
            source_df = best_per_init.copy()
        else: 
            methods = sub_df["method"].unique()
            source_df = sub_df.copy()

        # Convert to n_unknown for the x-axis
        source_df["n_unknown"] = nsrc - source_df["n_known_src"]
        
        # -------------------------------------------------
        # Calculate Boxplot Offsets for Side-by-Side Plotting
        # -------------------------------------------------
        n_methods = len(methods)
        if n_methods > 1:
            box_width = 0.6 / n_methods
            offsets = np.linspace(-0.3 + box_width/2, 0.3 - box_width/2, n_methods)
        else:
            box_width = 0.4
            offsets = [0]

        # -------------------------------------------------
        # Loop over methods separately 
        # -------------------------------------------------
        for i, method in enumerate(methods):
            method_color = color_map[method]
            offset = offsets[i]
            
            method_df = source_df[source_df["method"] == method]
            
            # --- Optional Faint Individual Lines ---
            if plot_individual_runs:
                for init_val, init_df in method_df.groupby("init_data"):
                    init_df = init_df.sort_values("n_unknown")
                    x_ind = init_df["n_unknown"].values
                    y_ind = init_df[err_type].values
                    
                    ax.plot(
                        x_ind + offset, # Apply offset so lines intersect their respective boxes
                        y_ind,
                        color=method_color,
                        alpha=0.05,
                        linewidth=1.5,
                        zorder=1
                    )

            # --- Box Plot Preparation ---
            x_vals = sorted(method_df["n_unknown"].unique())
            data_to_plot = []
            positions = []
            
            custom_mean_props = {
                'marker': method_marker(method),
                'markerfacecolor': method_color,
                'markeredgecolor': '#000',#method_color,
                'markersize': 6
            }
            
            for x_val in x_vals:
                y_data = method_df[method_df["n_unknown"] == x_val][err_type].dropna().values
                if len(y_data) > 0:
                    data_to_plot.append(y_data)
                    positions.append(x_val + offset)
            
            # Draw Boxplot
            if data_to_plot:
                bp = ax.boxplot(
                    data_to_plot,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False, # Keep custom x-axis mapping
                    zorder=3,
                    showmeans=True,
                    meanprops = custom_mean_props
                )
                
                # Apply Styles to Boxplot Elements
                for box in bp['boxes']:
                    box.set_facecolor(method_color)
                    box.set_alpha(0.6) 
                    box.set_edgecolor(method_color)
                    box.set_linewidth(1.2)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(1.5)
                for whisker in bp['whiskers']:
                    whisker.set_color(method_color)
                    whisker.set_linewidth(1.2)
                    whisker.set_linestyle('--')
                for cap in bp['caps']:
                    cap.set_color(method_color)
                    cap.set_linewidth(1.2)
                for flier in bp['fliers']:
                    flier.set(marker='o', color=method_color, alpha=0.3, markersize=3, markeredgecolor=method_color)

            # Dummy patch for the legend
            # Use a zero-size Rectangle proxy so Matplotlib can calculate paths safely
            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0), 0, 0, 
                    facecolor=method_color, 
                    alpha=0.6, 
                    label=f"{metric_label(err_type)} ({method_label(method)})"
                )
            )
            #ax.add_patch(mpatches.Patch(color=method_color, alpha=0.6, label=f"{metric_label(err_type)} ({method_label(method)})"))

        # -------------------------------------------------
        # Axis Limits and Formatting
        # -------------------------------------------------
        upper_bound = max_val if max_val else metric_bounds(err_type)[1] * 1.05
        lower_bound = min_val if min_val else metric_bounds(err_type)[0] * 0.95
        
        ax.set_ylim(bottom=lower_bound, top=upper_bound)
        
        # Lock discrete x-ticks to the unique n_unknown values
        all_n_unknown = sorted(source_df["n_unknown"].unique())
        ax.set_xticks(all_n_unknown)
        ax.set_xticklabels(all_n_unknown)
        if len(all_n_unknown) > 0:
            ax.set_xlim(min(all_n_unknown) - 0.6, max(all_n_unknown) + 0.6)

        # Labels setup
        sd_err = sub_df['sn_ratio'].unique()[0]
        nreps_vals = sub_df['nreps'].unique()
        nreps_str = ", ".join(map(str, nreps_vals))

        ax.set_title(f"Observed Dimension: {nobs}\nLatent Dimension: {nsrc}\nSample Size: {nreps_str}")
        ax.set_xlabel(r"Number of unknown sources ($n_{\mathrm{u}}$)")
        ax.set_ylabel(f'{metric_label(err_type)}')
        
        # Complete case boundary line mapping
        boundary_val =  nobs
        ax.axvline(boundary_val, color="#FF3D00", linestyle=":", alpha=0.7, linewidth=2, zorder=4)
        ax.text(boundary_val + 0.1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, 
                "↑ Complete Case Boundary", color="#FF3D00", fontweight="bold", rotation=90, alpha=0.8)
        
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # -------------------------------------------------
    # Remove unused subplots
    # -------------------------------------------------
    for i in range(len(keys), len(axes)):
        fig.delaxes(axes[i])

    # -------------------------------------------------
    # Shared legend
    # -------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_location, bbox_to_anchor=legend_anchor)
    plt.tight_layout()
    plt.show()
    

def plot_proportion_unknown_to_nobs(df, err_type="A_err", best_init_metric="X_err", remove_complete_cases = False):

    df = df.copy()
    df["alpha"] = (df["nsrc"] - df["n_known_src"]) / df["nobs"]
    #df["alpha"] = (df["nsrc"] - df["n_known_src"]) / df["nobs"]
    complete_mask = df['nsrc'] == df['nobs']
    # Select best initialization
    group_cols = get_columns_by_type('base_identifier') + get_columns_by_type('filter_col') + ['alpha']
    exclude = {'init_data', 'data_seed'}
    across_data_init_group = [col for col in group_cols if col not in exclude]
    
    if remove_complete_cases:
        df_complete = df[complete_mask].copy()
        df = df[~complete_mask]
    else:
        df_complete = df.copy()
    
    if lower_is_better(best_init_metric):
        best_per_init = df.loc[
            df.groupby(group_cols)[best_init_metric].idxmin()
        ]
    else:
        best_per_init = df.loc[
            df.groupby(group_cols)[best_init_metric].idxmax()
        ]

    # Average over datasets
    grouped = (
        best_per_init
        .groupby(across_data_init_group)[err_type]
        .mean()
        .reset_index()
    )

    # One figure per nobs
    for nobs, df_nobs in grouped.groupby("nobs"):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot all methods
        for method, df_method in df_nobs.groupby("method"):
                

            # faint individual curves for each nsrc
            for nsrc, sub_df in df_method.groupby("nsrc"):

                sub_df = sub_df.sort_values("alpha")

                ax.plot(
                    sub_df["alpha"],
                    sub_df[err_type],
                    color = method_color(method),
                    linestyle = method_linestyle(method),
                    alpha=0.05,
                )

                #ax.scatter(
                #    sub_df["alpha"],
                #    sub_df[err_type],
                #    color = method_color(method),
                #    marker = method_marker(method),
                #    alpha=0.1,
                #)

               # # complete-case reference line
               # if nsrc == nobs:
               #     mask = np.isclose(sub_df["alpha"], 1)
#
               #     if mask.any():
               #         ax.axhline(
               #             y=sub_df.loc[mask, err_type].iloc[0],
               #             linestyle="--",
               #             alpha=0.3,
               #         )

            # -----------------------------------------
            # Average collapse curve for this method
            # -----------------------------------------
            avg_curve = (
                df_method
                .groupby("alpha")[err_type]
                .mean()
                .reset_index()
                .sort_values("alpha")
            )

            ax.plot(
                avg_curve["alpha"],
                avg_curve[err_type],
                linewidth=2,
                label=method_label(method),
                color = method_color(method),
                linestyle = method_linestyle(method),
                marker = method_marker(method)
            )
            
                
        # ---------------------------------------------
        # Formatting
        # ---------------------------------------------
        ax.axvline(x=1, linestyle=":", color="black")

        ax.invert_xaxis()

        ax.set_xlabel(
            r"$\alpha = (n_{src}-n_{known})/d_{x}$"
        )

        ax.set_ylabel(metric_label(err_type))

        direction = (
            "↓ smaller is better"
            if lower_is_better(err_type)
            else "↑ greater is better"
        )

        ax.set_title(
            f"nobs = {nobs} ({direction})"
        )

        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()    

def plot_alpha_contrast(df, method="standard"):
    """
    Collapse curves using alpha = n_known_src / nsrc.
    Find the difference between curves that have the same number of sources
    """

    df = df.copy()

    # ---------------------------------------------
    # Step 1: filter method
    # ---------------------------------------------
    df = df[df["method"] == method]

    # ---------------------------------------------
    # Step 2: define alpha
    # ---------------------------------------------
    df["alpha"] = (df["nsrc"] - df["n_known_src"]) / df["nobs"]

    # ---------------------------------------------
    # Step 3: average over initializations
    # ---------------------------------------------

    best_per_init = df.loc[
        # ---------------------------------------------
            df.groupby(["nobs", "nsrc", "init_data", "method", "n_known_src"])["X_err"].idxmin()
        ]


    grouped = (
        best_per_init.groupby(["nobs", "nsrc", "alpha"])["A_err"]
        .mean()
        .reset_index()
    )

    # ---------------------------------------------
    # Step 4: plot all curves
    # ---------------------------------------------
    plt.figure(figsize=(8, 6))

    for (nobs, nsrc), sub_df in grouped.groupby(["nobs", "nsrc"]):
        if nobs == nsrc or nobs == nsrc-1:
            plt.plot(
                sub_df["alpha"],
                sub_df["A_err"],
                alpha=0.4,
                label=f"({nobs},{nsrc})"
            )
        
    plt.axvline(x = 1, linestyle = ":")

    # ---------------------------------------------
    # Optional: global average curve
    # ---------------------------------------------
    avg_curve = (
        grouped.groupby("alpha")["A_err"]
        .mean()
        .reset_index()
    )

    #plt.plot(
    #    avg_curve["alpha"],
    #    avg_curve["A_err"],
    #    color="black",
    #    linewidth=3,
    #    label="Average"
    #)
    plt.gca().invert_xaxis()
    # ---------------------------------------------
    # Plot formatting
    # ---------------------------------------------
    plt.xlabel(r"$\alpha = (n_{src} - n_{known}) / d_{x}$")
    plt.ylabel("A_err")
    plt.title(f"Alpha-collapse ({method})")

    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_difference_from_baseline(df, err_type='A_err', method=None):
    """
    Plots the difference in err_type from the complete case (nobs == nsrc)
    for each nobs value, only considering alpha <= 1.

    Parameters
    ----------
    df       : DataFrame with columns [nobs, nsrc, alpha, err_type, method]
    err_type : str   — column to plot
    method   : str   — filter to a specific method, or None for all
    """
    df = df.copy()

    # Optional method filter
    if method is not None:
        df = df[df['method'] == method]

    # Only consider alpha <= 1 (overcomplete regime)
    df["alpha"] = (df["nsrc"] - df["n_known_src"]) / df["nobs"]
    df = df[df['alpha'] <= 1.0]
    
    
    best_per_init = df.loc[
        # ---------------------------------------------
            df.groupby(["nobs", "nsrc", "init_data", "method", "n_known_src"])["X_err"].idxmin()
        ]

    df = best_per_init.copy()
    grouped = (
        best_per_init.groupby(["nobs", "nsrc", "alpha"])["A_err"]
        .mean()
        .reset_index()
    )


    # ── Build baseline: complete case (nobs == nsrc) per nobs ─────────────────
    baseline = (
        df[df['nobs'] == df['nsrc']]
        .groupby(['nobs', 'alpha'])[err_type]
        .mean()
        .reset_index()
        .rename(columns={err_type: 'baseline'})
    )

    if baseline.empty:
        print('No complete case (nobs == nsrc) found in data.')
        return

    # ── Merge baseline into main df ───────────────────────────────────────────
    df = df[df['nobs'] != df['nsrc']]   # exclude complete case from difference plot
    df = df.merge(baseline[['nobs', 'alpha', 'baseline']], on=['nobs', 'alpha'], how='left')
    df['diff'] = df[err_type] - df['baseline']

    # ── Plot ──────────────────────────────────────────────────────────────────
    unique_nobs = sorted(df['nobs'].unique())
    n_plots     = len(unique_nobs)
    n_cols      = int(np.ceil(np.sqrt(n_plots)))
    n_rows      = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                              squeeze=False)
    axes = axes.reshape(-1)

    for ax, nobs in zip(axes, unique_nobs):
        sub = df[df['nobs'] == nobs]

        for nsrc, grp in sub.groupby('nsrc'):
            # Average over initialisations
            mean_diff = grp.groupby('alpha')['diff'].mean()
            std_diff  = grp.groupby('alpha')['diff'].std()

            ax.plot(
                mean_diff.index,
                mean_diff.values,
                label=f'nsrc={nsrc}',
                marker='o'
            )
            ax.fill_between(
                mean_diff.index,
                mean_diff.values - std_diff.values,
                mean_diff.values + std_diff.values,
                alpha=0.2
            )

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8,
                   label='baseline (complete case)')
        ax.set_title(f'nobs={nobs}')
        ax.set_xlabel('alpha')
        ax.set_ylabel(f'{metric_label(err_type)} for (overcomplete - complete case)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.invert_xaxis()


    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    title = f'Difference from complete case | {metric_label(err_type)}'
    if method:
        title += f' | method={method}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    