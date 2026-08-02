import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import overcomplete_learning.data as ol_data
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import overcomplete_learning.metrics as ol_metric  
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import scipy.stats as stats


#####################################
# AESTHAETIC CHOICES - AND - HELPERS
#####################################
METHOD_STYLES = {
        # Standard Family (Blue / Solid)
        "standard":                 {"color": "tab:blue",   "linestyle": "-",  "marker": "o",   'label': 'SB-VEM', 'latex': r"$\GirolamiExtended$"},
        "standard_iid_err":         {"color": "tab:blue",   "linestyle": "--",  "marker": "x",  'label': 'SB-VEM', 'latex': r"$\GirolamiExtended$"}, #possibly use SB-VEM-iid
        "standard_iid":             {"color": "tab:blue",   "linestyle": "--",  "marker": "x",  'label': 'SB-VEM', 'latex': r"$\GirolamiExtended$"},

        # OLS Family (Green / Dashed)
        "OLS":                      {"color": "tab:green",  "linestyle": "-", "marker": "o",    'label': 'LS-B-VEM', 'latex': r"$\OLSGirolami$"},
        "OLS_iid":                  {"color": "tab:green",  "linestyle": "--", "marker": "x",   'label': 'LS-B-VEM', 'latex': r"$\OLSGirolami$"},               #possibly use B-VEM-iid
        "OLS_iid_fixed_SNR":        {"color": "tab:green",  "linestyle": "--", "marker": "x",   'label': 'LS-B-VEM Fixed SNR = True'},
        "OLS_iid_fixed_sd":         {"color": "tab:green",  "linestyle": "-", "marker": "o",   'label':  'LS-B-VEM Fixed Variance = True'},
        "OLS_debias":               {"color": "aquamarine",  "linestyle": "-.", "marker": "s",  'label': 'LS-B-VEM-+', 'latex': r"$\OLSDebias$"},
        "OLS_debias_iid":           {"color": "aquamarine",  "linestyle": "-.", "marker": "s",  'label': 'LS-B-VEM-+', 'latex': r"$\OLSDebias$"}, #possibly use B-VEM-
        # Baseline Abberations (Cyan / Wrong Math)
        "standard_wrong":           {"color": "tab:cyan",   "linestyle": ":",  "marker": "v",   'label': 'Girolami_misspecified',   'latex': r"$\GirolamiWrong$"},
        "OLS_wrong":                {"color": "tab:cyan",   "linestyle": ":", "marker": "s",    'label': 'LS -> Girolami_misspecified'},

        # Alternative Algorithms
        "sdp":                      {"color": "tab:orange", "linestyle": "-",  "marker": "o",   'label': 'OverICA',     'latex': r"$\OverICA$"},
        "OverICA":                  {"color": "tab:orange", "linestyle": "-",  "marker": "o",   'label': 'OverICA',     'latex': r"$\OverICA$"},
        "OverICA_fixed_sd":                  {"color": "tab:orange", "linestyle": "-",  "marker": "o",   'label': 'OverICA Fixed Variance = True',     'latex': r"$\OverICA$"},
        "OLS_oracle_fixed_SNR":     {"color": "tab:red",    "linestyle": "--",  "marker": "x",   'label':  'Oracle Fixed SNR = True'},
        "OLS_oracle_fixed_sd":      {"color": "tab:red",    "linestyle": "-", "marker": "o",   'label':  'Oracle Fixed Variance = True'},
        "fast_ICA":                 {"color": "tab:purple", "linestyle": "-.", "marker": "o",   'label': 'FastICA++',   'latex': r"$\FastEMDimReduc$"},

        # Control Baselines (Brown / Plus marker)
        "random":                   {"color": "tab:brown",  "linestyle": "-",  "marker": "P",   'label': 'Random',                          'latex': r"$\RANDOM$"},
        "Random_fixed_sd":                   {"color": "tab:brown",  "linestyle": "-",  "marker": "P",   'label': 'Random Fixed Variance = True',    'latex': r"$\RANDOM$"},
        "random_quasiorthogonal":   {"color": "tab:brown",  "linestyle": "-",  "marker": "P",   'label': 'Random Quasiorthogonal Matrix',   'latex': r"$\RANDOM$"}}

def method_label(method):
    return METHOD_STYLES[method]['label']
def method_linestyle(method):
    return METHOD_STYLES[method]['linestyle']
def method_color(method):
    return METHOD_STYLES[method]['color']
def method_marker(method):
    return METHOD_STYLES[method]['marker']
def method_latex(method):
    return METHOD_STYLES[method]['latex']

#####################################
# ERROR PROPERTIES AND HELPERS
#####################################
ERROR_PROPERTIES = {
        'A_err':                    {'val_min': 0,  'val_max': 1,   'lower_is_better': True,    'latex': r"$\ferr$",        'label': 'Column Scaled RMSE'},
        'A_err_angle_mean':         {'val_min': 0,  'val_max': 1,   'lower_is_better': True,    'latex': r"$\aerrmean$",    'label': 'Mean Angular Error'}, #(rad)*pi/2)
        'S_err_MSE':                {'val_min': 0,  'val_max': 1,   'lower_is_better': True,    'latex': r"$\serrMSE$",     'label': 'Source Relative L2 Error'},
        'A_err_angle_max':          {'val_min': 0,  'val_max': 1,   'lower_is_better': True,    'latex': r"$\aerrmax$",     'label': 'Maximum Angular Error'}, #(rad)*pi/2)
        'S_err_MCC':                {'val_min': 0,  'val_max': 1,   'lower_is_better': False,   'latex': r"$\serrMCC$",     'label': 'Source MCC'}, #Mean Correlation Coefficient
        'X_err':                    {'val_min': 0,  'val_max': 1,   'lower_is_better': True,    'latex': r"$\xerrMSE$",     'label': 'Reconstruction MSE'},
        'elapsed':                  {'val_min': 0,  'val_max': 3,   'lower_is_better': True,    'latex': r"$Time Elapsed (s)$",     'label': 'Elapsed Time'},
        'log_elapsed':              {'val_min': 0,  'val_max': 3,   'lower_is_better': True,    'latex': r"$log - Time Elapsed (s)$",     'label': 'Elapsed log Time'},
        'convergence_iteration':    {'val_min': 0,  'val_max': 200, 'lower_is_better': True,    'latex': r"Iterations",     'label': 'Total Number of Iterations'},
        'seconds_per_it':           {'val_min': 0,  'val_max': 1,   'lower_is_better': True,    'latex': r"Seconds Per Iteration",     'label': 'Seconds Per Iteration'},
        'ratio':                    {'val_min': 0,  'val_max': 3,   'lower_is_better': True,    'latex': r"Ratio",     'label': 'Ratio'},
        
}
def lower_is_better(metric):
    return ERROR_PROPERTIES[metric]['lower_is_better']

def metric_label(metric):
    return ERROR_PROPERTIES[metric]['label']

def metric_latex(metric):
    return ERROR_PROPERTIES[metric]['latex']

def metric_bounds(metric):
    return (ERROR_PROPERTIES[metric]['val_min'], ERROR_PROPERTIES[metric]['val_max'])

#####################################
# COLUMNS PROPERTIES AND HELPERS
#####################################
COLUMN_PROPERTIES = {
    #settings
    'nsrc':             {'label': r'$d_{S}$',                       'type': 'base_identifier'},
    'nobs':             {'label': r'$d_{X}$',                       'type': 'base_identifier'},
    'n_known_src':      {'label': 'Number of Known Sources',        'type': 'base_identifier'},
    'n_unknown':        {'label': 'Number of Unknown Sources',      'type': 'base_identifier'},
    'data_seed':        {'label': 'Seed for Data Generation',       'type': 'base_identifier'},      
    'method':           {'label': 'Algortihm',                      'type': 'base_identifier'},
    
    #init data and info
    'init':             {'label': 'Parameter Initialization Identifier',    'type': 'initialization_details'},
    'init_data':        {'label': 'Data Generation Identifier',             'type': 'initialization_details'},
    'init_seed':        {'label': 'Seed For Parameter Initialization',      'type': 'initialization_details'}, 
    'converged':        {'label': 'Flag For Algorithm Convergence',         'type': 'initialization_details'},
    
    #variable identifiers
    'nreps':            {'label': 'Sample Size',                                 'type': 'filter_col'},
    'err_correlation_type': {'label': 'Additive Error Correlation Type',    'type': 'filter_col'},
    'normalize_A':      {'label': f'Normalizing $A_t$ ',                    'type': 'filter_col'},
    'sn_ratio'   :      {'label': 'Signal-to-Noise Ratio',                  'type': 'legacy_identifier'},    
    'sd_err'     :      {'label': 'Noise Std. Deviation',                   'type': 'legacy_identifier'},
    #metrics
    'convergence_iteration': {'label': metric_label('convergence_iteration'),        'type': 'error_metric'},
    'elapsed':          {'label': metric_label('elapsed'),              'type': 'error_metric'},
    'A_err':            {'label': metric_label('A_err'),                'type': 'error_metric'},
    'A_err_angle_mean': {'label': metric_label('A_err_angle_mean'),     'type': 'error_metric'}, 
    'A_err_angle_max':  {'label': metric_label('A_err_angle_max'),      'type': 'error_metric'},
    'S_err_MSE':        {'label': metric_label('S_err_MSE'),            'type': 'error_metric'},
    'S_err_MCC':        {'label': metric_label('S_err_MCC'),            'type': 'error_metric'},
    'X_err':            {'label': metric_label('X_err'),                'type': 'error_metric'},
}

def col_label(col):
    return COLUMN_PROPERTIES[col]['label']
def col_type(col):
    return COLUMN_PROPERTIES[col]['type']
def get_columns_by_type(target_type):
    """
    Extracts column names that match a specific type.
    
    Parameters:
        target_type (str): The type to filter by (e.g., 'error_metric').
        
    Returns:
        list: A list of column name strings.
    """
    return [col for col, metadata in COLUMN_PROPERTIES.items() if metadata.get('type') == target_type]

#####################################
# PLOTTING FUNCTIONALITIES
#####################################
##################
#VISUALIZE DATASET
##################
def plot_gram_matrix(
    A: np.ndarray,
    remove_diagonal: bool = False,
    title = None,
) -> None:
    """
    Plot the Gram matrix G = A^T A.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (n, p).
    remove_diagonal : bool
        If True, set diagonal to zero for better visualization
        of off-diagonal correlations.
    title : str
        Title of the plot.
    """

    if title is None:
        title = f"Gram Matrix"
    mu = ol_data.coherence(A)
    title = f'{title} with coherence: {round(mu, ndigits=2)}'
    # Step 1: Compute Gram matrix
    G = A.T @ A

    # Step 2: Optionally remove diagonal
    if remove_diagonal:
        G = G.copy()
        np.fill_diagonal(G, 0)

    # Step 3: Plot heatmap
    plt.figure(figsize=(6, 5))
    im = plt.imshow(G, cmap="viridis", aspect="auto")

    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("Column index")
    plt.ylabel("Column index")

    plt.tight_layout()
    plt.show()

def plot_latents(S, n, type:str = 'latent sources'):
    nrep, nsrc = S.shape
    n = min(n, nsrc)  # guard against requesting more than available
    
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]  # ensure iterable when n=1
    
    for i, ax in enumerate(axes): # type: ignore
        #ax.scatter(range(nrep), S[:, i], alpha=0.4, s=10, color='steelblue')
        ax.plot(range(nrep), S[:, i], alpha=0.6, linewidth=0.8, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'Source {i+1}', fontsize=11)
        ax.set_xlabel('Sample index', fontsize=9)
        if i == 0:
            ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'First {n} {type}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
def plot_em_data(
    data,
    estimate = None,
    initial_A = None,
    arrow_scale=3.0,
    point_alpha=0.3,
    point_size=10,
    ax=None,
):
    """
    Plot observations X and columns of A as arrows from the origin.

    Parameters
    ----------
    data : EMData
        EMData object.
    arrow_scale : float
        Scaling applied to arrows for visualization.
    point_alpha : float
        Transparency of data points.
    point_size : float
        Scatter point size.
    ax : matplotlib axis, optional
        Existing axis.
    """

    if data.X.shape[1] != 2:
        raise ValueError(
            f"plot_em_data currently requires nobs=2, "
            f"got nobs={data.X.shape[1]}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # --------------------------------------------------
    # Scatter observations
    # --------------------------------------------------
    ax.scatter(
        data.X[:, 0],
        data.X[:, 1],
        alpha=point_alpha,
        s=point_size,
        zorder=1,
    )

    # --------------------------------------------------
    # Plot columns of A
    # --------------------------------------------------
    def plot(A, col = '#000000'):
        for j in range(A.shape[1]):
            dx = arrow_scale * A[0, j]
            dy = arrow_scale * A[1, j]

            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(0.0, 0.0),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2,
                    color = col
                ),
                zorder=2,
            )

            ax.text(
                dx,
                dy,
                f"$a_{j+1}$",
                fontsize=10,
            )    
    plot(data.A)
    
    if estimate is not None:
        A_est = estimate['A_est']
        B, _, _, _, _ = ol_data.best_permutation_match_sign_flips(A = data.A, B = A_est)
        plot(A=ol_data.normalize_columns(B), col = '#FF0000')
    if initial_A is not None:
        B, _, _, _, _ = ol_data.best_permutation_match_sign_flips(A = data.A, B = initial_A)
        
        plot(A=ol_data.normalize_columns(B), col = "#D8860B")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Observations and Mixing Directions")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    return ax

##################
#VISUALIZE RESULTS
###################

def plot_column_embeddings(A, B):
    """
    Takes in two matrices A and B of shape (D, N) and plots their columns 
    as arrows. Embeds the D dimensions into 2D plots by pairing the rows. 
    If D is odd, the final plot is a 1D representation.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
        
    D, N = A.shape
    num_plots = math.ceil(D / 2)
    
    # Create subplots side by side
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]
        
    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        dim1 = plot_idx * 2
        dim2 = dim1 + 1
        
        is_2d = dim2 < D
        
        # Plot arrows for A (Blue) and B (Red)
        for j in range(N):
            # Matrix A
            x_A = A[dim1, j]
            y_A = A[dim2, j] if is_2d else 0
            ax.arrow(0, 0, x_A, y_A, color='blue', alpha=0.7, 
                     head_width=0.08, head_length=0.15, length_includes_head=True)
            
            # Matrix B
            x_B = B[dim1, j]
            y_B = B[dim2, j] if is_2d else 0
            ax.arrow(0, 0, x_B, y_B, color='red', alpha=0.7, 
                     head_width=0.08, head_length=0.15, length_includes_head=True)
            
        # Plot formatting and labels
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Calculate limits to keep origin centered or visible
        all_x = np.concatenate([A[dim1, :], B[dim1, :], [0]])
        x_max = np.max(np.abs(all_x)) * 1.2 or 1
        ax.set_xlim(-x_max, x_max)
        
        if is_2d:
            ax.set_title(f"Dimensions {dim1} vs {dim2} (2D)")
            ax.set_xlabel(f"Row {dim1}")
            ax.set_ylabel(f"Row {dim2}")
            
            all_y = np.concatenate([A[dim2, :], B[dim2, :], [0]])
            y_max = np.max(np.abs(all_y)) * 1.2 or 1
            ax.set_ylim(-y_max, y_max)
        else:
            ax.set_title(f"Dimension {dim1} (1D)")
            ax.set_xlabel(f"Row {dim1}")
            ax.set_ylim(-1, 1)
            ax.set_yticks([]) # Hide y-axis for the 1D plot
            ax.axhline(0, color='black', linewidth=1) # Draw a solid x-axis

    # Add a global legend
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
    fig.legend(custom_lines, ['Matrix A', 'Matrix B'], loc='upper right', bbox_to_anchor=(1.0, 1.1))
    
    plt.tight_layout()
    plt.show()

#–---------------------
# fixed dimensionality
#----------------------
def plot_EM_results_grid(df, err_type, plot_type='trajectory', only_best_init=True, 
                         best_init_metric='X_err', max_val=None, min_val=None, mean_or_median = 'mean',
                         plot_individual_runs=False):
    """
    Plot EM results in a grid over (nobs, nsrc) using either continuous trajectories
    or side-by-side discrete boxplots.

    Parameters:
    -----------
    plot_type : str
        'trajectory' -> plots mean line with shaded standard deviation bands.
        'boxplot'    -> plots grouped boxplots with custom mean markers per method.
    """
    if plot_type not in ['trajectory', 'boxplot']:
        raise ValueError("plot_type must be either 'trajectory' or 'boxplot'")

    df = df.copy()

    # -------------------------------------------------
    # Unique experiment settings & grid setup
    # -------------------------------------------------
    groups = df.groupby(["nobs", "nsrc", "err_correlation_type", "nreps"])
    keys = list(groups.groups.keys())
    n_plots = len(keys)

    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    size_multiplier = 1.4 if (n_cols == 1 and n_rows == 1) else 1
        
    # Legend layout anchors
    if lower_is_better(metric=err_type):
        legend_location, legend_anchor = 'upper left', (0.1, 0.85)
    else:
        legend_location, legend_anchor = 'upper right', (0.98, 0.85)
        
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(4* n_cols * size_multiplier, 3 * n_rows * size_multiplier),
                             squeeze=False)
    axes = axes.reshape(-1)

    # Styles mapping
    color_map     = {k: v["color"] for k, v in METHOD_STYLES.items()}
    linestyle_map = {k: v["linestyle"] for k, v in METHOD_STYLES.items()}
    marker_map    = {k: v["marker"] for k, v in METHOD_STYLES.items()}
    
    # -------------------------------------------------
    # Loop over each (nobs, nsrc) grid cell
    # -------------------------------------------------
    for ax, (nobs, nsrc, err_correlation_type, nreps) in zip(axes, keys):
        sub_df = groups.get_group((nobs, nsrc, err_correlation_type, nreps))

        # --- Filter Best Initialization Paths ---
        group_cols = ["init_data", "method", "n_known_src"]
        valid_methods = sub_df.groupby("method")[best_init_metric].apply(lambda x: x.notna().all())
        valid_methods = valid_methods[valid_methods].index
        
        if only_best_init:
            eligible = sub_df[sub_df["method"].isin(valid_methods)]
            best_per_init = eligible.loc[eligible.groupby(group_cols)[best_init_metric].idxmin()]
            methods = best_per_init['method'].unique()
            source_df = best_per_init.copy()
        else: 
            methods = sub_df["method"].unique()
            source_df = sub_df.copy()

        # Standardize x-axis conversion to n_unknown up front
        source_df["n_unknown"] = nsrc - source_df["n_known_src"]
        
        # --- Calculate Offsets if Boxplot is Active ---
        n_methods = len(methods)
        if plot_type == 'boxplot':
            box_width = 0.6 / n_methods if n_methods > 1 else 0.4
            offsets = np.linspace(-0.3 + box_width/2, 0.3 - box_width/2, n_methods) if n_methods > 1 else [0]
        else:
            box_width = 0.0
            offsets = [0] * n_methods

        # -------------------------------------------------
        # Loop over methods separately 
        # -------------------------------------------------
        for i, method in enumerate(methods):
            method_color = color_map[method]
            offset = offsets[i]
            method_df = source_df[source_df["method"] == method]
            
            # --- 1. Optional Faint Individual Run Lines ---
            if plot_individual_runs:
                for init_val, init_df in method_df.groupby("init_data"):
                    init_df = init_df.sort_values("n_unknown")
                    ax.plot(
                        init_df["n_unknown"].values + offset,
                        init_df[err_type].values,
                        color=method_color,
                        linestyle=linestyle_map[method] if plot_type == 'trajectory' else '-',
                        alpha=0.05 if plot_type == 'boxplot' else 0.1,
                        linewidth=1.5,
                        zorder=1
                    )

            # --- 2. Render Trajectory Layer ---
            if plot_type == 'trajectory':
                # Group by unified x-axis component
                stats = method_df.groupby("n_unknown")[err_type].agg(['mean', 'std', 'min', 'max', 'median']).sort_index()
                x_vals = stats.index.values
                
                if mean_or_median == 'mean':
                    y_mean = stats['mean'].values
                    y_std = stats['std'].fillna(0).values
                
                    ax.plot(
                        x_vals, y_mean,
                        marker=marker_map[method],
                        color=method_color,
                        linestyle=linestyle_map[method],
                        label=f"{method_label(method)}",#{metric_label(err_type)} (
                        linewidth=2,
                        zorder=3
                    )
                    ax.fill_between(
                        x_vals, y_mean - y_std, y_mean + y_std,
                        color=method_color, alpha=0.1, zorder=2
                    )
                else:
                    y_mean = stats['median'].values
                    y_min = stats['min'].fillna(0).values
                    y_max = stats['max'].fillna(0).values
                    ax.plot(
                        x_vals, y_mean,
                        marker=marker_map[method],
                        color=method_color,
                        linestyle=linestyle_map[method],
                        label=f"{method_label(method)}",
                        #{metric_label(err_type)} (
                        linewidth=2,
                        zorder=3
                    )
                    ax.fill_between(
                        x_vals, y_min, y_max,
                        color=method_color, alpha=0.1, zorder=2
                    )
                

            # --- 3. Render Boxplot Layer ---
            elif plot_type == 'boxplot':
                x_vals = sorted(method_df["n_unknown"].unique())
                data_to_plot, positions = [], []
                
                for x_val in x_vals:
                    y_data = method_df[method_df["n_unknown"] == x_val][err_type].dropna().values
                    if len(y_data) > 0:
                        data_to_plot.append(y_data)
                        positions.append(x_val + offset)
                
                if data_to_plot:
                    bp = ax.boxplot(
                        data_to_plot, positions=positions, widths=box_width,
                        patch_artist=True, manage_ticks=False, zorder=3,
                        showmeans=True,
                        meanprops={
                            'marker': method_marker(method),
                            'markerfacecolor': method_color,
                            'markeredgecolor': '#000',
                            'markersize': 6
                        }
                    )
                    
                    # Style assignments
                    for box in bp['boxes']:
                        box.set(facecolor=method_color, alpha=0.6, edgecolor=method_color, linewidth=1.2)
                    for median in bp['medians']:
                        median.set(color='black', linewidth=1.5)
                    for whisker in bp['whiskers']:
                        whisker.set(color=method_color, linewidth=1.2, linestyle='--')
                    for cap in bp['caps']:
                        cap.set(color=method_color, linewidth=1.2)
                    for flier in bp['fliers']:
                        flier.set(marker='o', color=method_color, alpha=0.3, markersize=3, markeredgecolor=method_color)

                # Safe zero-size patch reference tracking for global legends
                ax.add_patch(
                    mpatches.Rectangle(
                        (0, 0), 0, 0, facecolor=method_color, alpha=0.6, 
                        label=f"{method_label(method)}" #{metric_label(err_type)} ()
                    )
                )

        # -------------------------------------------------
        # Axis Limits and Formatting
        # -------------------------------------------------
        upper_bound = max_val if max_val else metric_bounds(err_type)[1] * 1.05
        lower_bound = min_val if min_val else metric_bounds(err_type)[0] * 0.95
        ax.set_ylim(bottom=lower_bound, top=upper_bound)
        
        all_n_unknown = sorted(source_df["n_unknown"].unique())
        
        if plot_type == 'boxplot':
            ax.set_xticks(all_n_unknown)
            ax.set_xticklabels(all_n_unknown)
            if len(all_n_unknown) > 0:
                ax.set_xlim(min(all_n_unknown) - 0.6, max(all_n_unknown) + 0.6)
            ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.grid(True)

        # Meta strings parsing
        nreps_vals = sub_df['nreps'].unique()
        nreps_str = ", ".join(map(str, nreps_vals))

        ax.set_title(f"Obs. Dim: {nobs}, Source Dim.: {nsrc}, Sample Size: {nreps_str}", fontsize = 14)
        ax.set_xlabel(r"Number of unknown sources ($n_{\mathrm{u}}$)", fontsize = 14)   
        ax.set_ylabel(f'{metric_label(err_type)}', fontsize = 14)
        
        # Complete case line configuration
        
        boundary_val = nobs
        
        ax.axvline(boundary_val, color="#000000", linestyle=":", alpha=0.6, linewidth=2.5, zorder=2)
     #   ax.text(boundary_val + 0.15, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, 
     #           "↑ Complete Case Boundary", color="dimgrey", fontweight="bold", rotation=90, alpha=0.6, zorder = 2)
     #   # Assuming 'ax' is your matplotlib axes object, and nobs = 4
        y_pos = upper_bound*0.9 if lower_is_better(err_type) else lower_bound*1.1  # Adjust this to sit right near the bottom of your y-axis

        # Draw the spanning arrow
        ax.annotate(
            '', 
            xy=(0, y_pos), 
            xytext=(nobs, y_pos), 
            xycoords='data', 
            textcoords='data',
            arrowprops=dict(arrowstyle='|-|', linestyle = ':', color='#000000', lw=1.5, alpha = 0.6), # '|-|' gives nice flat end-caps
        )

        # Place the text centered exactly halfway across the span
        ax.text(
            nobs / 2, 
            y_pos + 0.01, # Slightly above the line
            r'(Under)Complete ($n_{\mathrm{u}}\leq d_{x}$)', 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            fontweight='bold', 
            color='#000000',
            alpha = 0.6
)
    # Clean empty frames
    for i in range(len(keys), len(axes)):
        fig.delaxes(axes[i])

    # Shared unified legend compilation
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_location, bbox_to_anchor=legend_anchor)
    plt.tight_layout()
    plt.show()

    return fig

#–---------------------
# fixed complexity
#----------------------

def plot_fixed_unknown_vs_known(df, err_type, best_init_metric, mean_or_median, method=None, add_lin_reg=True, 
                                max_val = None, min_val = None,
                                center_X = True, add_infobox = True, make_square = False, ):
    """
    Plots error metrics against the number of known sources, grouping lines
    by a fixed number of unknown sources.
    
    Generates a grid of subplots for each method, evaluating one `nobs` per figure.
    """
    df = df.copy()
  
    # 1. Standardize 'method' into a list for consistent iteration
    if method is not None:
        if isinstance(method, str):
            methods_to_plot = [method]
        elif isinstance(method, (list, tuple, np.ndarray)):
            methods_to_plot = list(method)
        else:
            raise ValueError("Method parameter must be a string or a list/iterable of strings.")
        
        # Filter dataframe up front for only the requested methods
        df = df[df["method"].isin(methods_to_plot)]
    else:
        # If no method is specified, grab all unique methods present in the data
        methods_to_plot = sorted(df["method"].dropna().unique())
   
    grouped = best_init_path(df = df, best_init_metric=best_init_metric)
    grouped_mean = across_data_stat(df = df, err_type=err_type, best_init_metric = best_init_metric, stat = mean_or_median)
    unique_nobs = sorted(grouped["nobs"].unique())
    num_methods = len(methods_to_plot)
    fig_list = []
    # LOOP 1: Iterate over Observation counts (nobs) - One Figure per nobs
    for nobs in unique_nobs:
        df_nobs = grouped[grouped["nobs"] == nobs]
        df_nobs_mean = grouped_mean[grouped_mean["nobs"] == nobs]
        # 3. Determine Grid Geometry (Targeting a square aspect ratio)
        if num_methods <= 1:
            n_cols, n_rows = 1, 1
            if make_square:
                fig, axes = plt.subplots(figsize=(5, 5))
            else:
                fig, axes = plt.subplots(figsize=(11, 6))
            axes = np.array([axes]) 
        else:
            n_cols = int(np.ceil(np.sqrt(num_methods)))
            n_rows = int(np.ceil(num_methods / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharex=False, sharey=False)
            axes = axes.flatten() 

        global_handles = []
        global_labels = []
    
        # LOOP 2: Iterate over Methods - Map to specific subplots
        for idx, meth in enumerate(methods_to_plot):
            ax = axes[idx]
            df_meth = df_nobs[df_nobs["method"] == meth]
            df_meth_mean = df_nobs_mean[df_nobs_mean["method"] == meth]
            if df_meth.empty:
                ax.text(0.5, 0.5, f"No Data for {meth}", ha='center', va='center')
                ax.set_title(f"Method: {method_label(meth)} |"+ r" $d_{x}$ " + f"= {nobs} | Statistic: {mean_or_median}" )
                ax.axis('off')
                continue

            # Identify curves excluding the boundary curve for the colormap
            other_unknowns = sorted([u for u in df_meth["n_unknown"].unique() if u != nobs])
            colors = plt.cm.viridis(np.linspace(0, 0.85, max(1, len(other_unknowns))))
            color_idx = 0
            
            boundary_x, boundary_y = None, None
            all_x_values = sorted(df_meth["n_known_src"].unique())
            
            # 1. Plot all non-boundary curves first
            for n_un in sorted(df_meth["n_unknown"].unique()):
                sub_df = df_meth[df_meth["n_unknown"] == n_un].sort_values("n_known_src")
                sub_df_mean = df_meth_mean[df_meth_mean["n_unknown"] == n_un].sort_values("n_known_src")
                if len(sub_df) == 0:
                    continue
                    
                if n_un == nobs:
                    # Save complete-case baseline data for later overlay processing
                    boundary_x = sub_df["n_known_src"].values
                    boundary_y = sub_df[err_type].values
                    boundary_x_mean = sub_df_mean["n_known_src"].values
                    boundary_y_mean = sub_df_mean[err_type].values
                    continue
                
                ### Render using the number of unknowns as a text string marker
                ax.plot(
                    sub_df_mean["n_known_src"], sub_df_mean[err_type],
                    marker=r"$n_{\mathrm{u}}$", markersize=0, linewidth=1.5, alpha=0.7,
                    color=colors[color_idx], 
                    #label=r"$n_{\mathrm{u}}$"+f" = {n_un}"
                )
                
                            # 3. ALTERNATIVE: Micro-Jittered Points (Strategy D)
                # Plotting raw points with tiny size, low alpha, and slight horizontal shift
                jitter = np.random.uniform(-0.2, 0.2, size=len(sub_df))
                ax.scatter(
                    sub_df["n_known_src"] + jitter,
                    sub_df[err_type],
                    color=colors[color_idx],
                    s=5,          # Micro-markers
                    alpha=0.3,   # Highly transparent
                    edgecolors='none',
                    #label=r"$n_{\mathrm{u}}$"+f" = {n_un}",
                    zorder=0      # Send to the background
                )
                
                
                color_idx += 1
                
                if add_lin_reg:
                    Y = sub_df[err_type].values
                    X = sub_df["n_known_src"].values
                    if len(X) < 3:
                        continue
                    x_max = max(X)                    
                    
                    
                    
                    
                    # 1. Save the midpoint explicitly to map back to raw X space
                    midpoint = (X.max() + X.min()) / 2
                    X_centered = X - midpoint 
                    
                    if(center_X):
                        # Simple linear regression: Y = slope * X + intercept
                        slope, intercept = np.polyfit(X_centered, Y, 1)
                    else:
                        slope, intercept = np.polyfit(X, Y, 1)
                    
                    # 2. Plot Midpoint Marker
                    ax.plot(
                        midpoint, intercept, 
                        marker=f"${n_un}$", markersize=11, alpha=0.9,
                        color=colors[color_idx - 1], zorder=6,
                        label=r"$n_{\mathrm{u}}$"+f" = {n_un}",
                        markeredgecolor = (0,0,0,0),
                        markeredgewidth=0.2,
                    )
                    
                    # 3. Force X-Limits calculation safely based on data
                    x_range = np.array([X.min(), X.max() * 1.05])
                    x_range = np.clip(x_range, a_min=0, a_max=x_max)
                    
                    # 4. Map the line equation back to raw X coordinates
                    y_range = slope * (x_range - midpoint) + intercept
                    
                    # Plot the continuous trend line
                    ax.plot(
                        x_range, y_range, 
                        linestyle="--", linewidth=1.5, alpha=1,
                        color=colors[color_idx - 1]
                    )

            # 2. Handle the "Complete" boundary curve (n_unknown == nobs)
            if boundary_x is not None and len(boundary_x) > 0:
                boundary_color = "#FF3D00" 
                
                ## Render complete case line using nobs as the text marker
                ax.plot(
                    boundary_x_mean, boundary_y_mean,
                    marker=f"${nobs}$", markersize=0, 
                    linewidth=3.5, color=boundary_color, zorder=5,
                    #label=r"$n_{\mathrm{u}}$"+f" = {nobs} (Complete)"
                )
                jitter = np.random.uniform(-0.2, 0.2, size=len(boundary_x))
                
                ax.scatter(
                    boundary_x+jitter,
                    boundary_y,
                    color=boundary_color,
                    s=5,          # Micro-markers
                    alpha=0.5,   # Highly transparent
                    edgecolors='none',
                    zorder=0,      # Send to the background
                    #label=r"$n_{\mathrm{u}}$"+f" = {nobs} (Complete)",
                    )

                
                
                if add_lin_reg and len(boundary_x) >= 2:
                    Y = boundary_y
                    X = boundary_x
                    x_max = max(X)                    
                    midpoint = (X.max() + X.min()) / 2
                    X_centered = X -  midpoint
                    if(center_X):
                        # Simple linear regression: Y = slope * X + intercept
                        slope, intercept = np.polyfit(X_centered, Y, 1)
                    else:
                        slope, intercept = np.polyfit(X, Y, 1)
                    
                    
                    ax.plot(
                        midpoint, intercept, 
                        marker=f"${nobs}$", markersize=13,
                        alpha=0.9,
                        color=boundary_color, zorder=6,
                        label=r"$n_{\mathrm{u}}$"+f" = {nobs} (complete)",
                        markeredgecolor = (0,0,0,0),
                        markeredgewidth=0.2,
                    )
                    
                    x_range = np.array([X.min(), X.max() * 1.05])
                    x_range = np.clip(x_range, a_min=0, a_max=x_max)
                    y_range = slope * (x_range - midpoint) + intercept
                    
                    ax.plot(
                        x_range, y_range, 
                        linestyle="--", linewidth=1.5, alpha=1,
                        color=boundary_color
                    )
                
                # 3. Dynamic background shading extended across the full x-axis width
                extended_x = np.array(all_x_values)
                extended_y = np.interp(extended_x, boundary_x, boundary_y)
                
                is_lower_better = lower_is_better(err_type)
                
                # Safely set y-limits for the fill
                bounds = metric_bounds(err_type)
                if not (max_val is None):
                    bounds = bounds[0], max_val
                if not (min_val is None):
                    bounds = min_val, bounds[1]
                
                
                ax.set_ylim(bottom=bounds[0]*1.05, top=bounds[1]*1.05)
                current_ylim = ax.get_ylim()
                x_center = np.mean(extended_x)
                
                if is_lower_better:
                    #ax.fill_between(extended_x, extended_y, current_ylim[0], color='#E8F5E9', alpha=0.5, zorder=0)
                    ax.text(x_center, (np.mean(extended_y) + current_ylim[0]) / 2, 
                            r"UNDERCOMPLETE REGIME $n_{\mathrm{u}}\leq d_{x}$", color="#2E7D32", fontsize=10, fontweight="bold", ha="center", alpha=0.6)
                    ax.text(x_center, (np.mean(extended_y) + current_ylim[1]) / 3, 
                            r"OVERCOMPLETE REGIME ($n_{\mathrm{u}}> d_{x}$)", color="#C62828", fontsize=10, fontweight="bold", ha="center", alpha=0.6)
                else:
                    #ax.fill_between(extended_x, extended_y, current_ylim[1], color='#E8F5E9', alpha=0.5, zorder=0)
                    ax.text(x_center, (np.mean(extended_y) + current_ylim[1]) / 2, 
                            r"UNDERCOMPLETE REGIME ($n_{\mathrm{u}}\leq d_{x}$)", color="#2E7D32", fontsize=10, fontweight="bold", ha="center", alpha=0.6)
                    ax.text(x_center, (np.mean(extended_y) + current_ylim[0]) / 3, 
                            r"OVERCOMPLETE REGIME ($n_{\mathrm{u}}> d_{x}$)", color="#C62828", fontsize=10, fontweight="bold", ha="center", alpha=0.6)
                
                ax.set_ylim(current_ylim)
                

            # ---------------------------------------------
            # Formatting & Cleanup per Subplot (ax)
            # ---------------------------------------------
            ax.set_xlabel(r"Number of Known Sources ($n_{\mathrm{k}}$)", fontsize=11)
            ax.set_ylabel(metric_label(err_type), fontsize=11)
            
            direction = "↓ decreasing is better" if lower_is_better(err_type) else "↑ increasing is better"
            ax.set_title(f"Method: {method_label(meth)} | $d_{{x}}$ = {nobs} | Statistic = {mean_or_median}\n"
                         f"Performance trajectory ({direction})", fontsize=12, fontweight='bold')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, linestyle=":", alpha=0.5)
            
            # Legend placement adjusted for subplots
            handles, labels = ax.get_legend_handles_labels()
            total_items = len(handles)
            max_nrow = 15
            adaptive_ncol = int(np.ceil(total_items / max_nrow)) if total_items > 0 else 1
            
            # Place legend nicely within or outside the specific ax
            #ax.legend(title="Unobserved Sources", loc="upper right", fontsize=8, ncol=adaptive_ncol)
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > len(global_handles):
                global_handles = handles
                global_labels = labels
            # Only add infobox once or modify it to attach to `ax` specifically
        # Cleanup unused axes in the grid
        for empty_idx in range(num_methods, len(axes)):
            fig.delaxes(axes[empty_idx])

        # Compress the subplots to the left to leave 20% of the figure empty on the right
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        
        # Place the legend on the SAME axes as the filterbox to guarantee alignment
        if add_infobox :
            legend_location = 'lower left'
            legend_anchor = (1.1, 0.4)
        else:
            if lower_is_better(err_type):
                legend_anchor = (1,1)
                legend_location = 'upper right'
            else:
                legend_anchor = (1,0)
                legend_location = 'lower right'
                #legend_location = 'upper right'
            
        if global_handles:
            max_nrow = 15
            adaptive_ncol = int(np.ceil(len(global_handles) / max_nrow)) if len(global_handles) > 0 else 1

            axes[-1].legend(
                global_handles, global_labels,
                title=r"Unobserved Sources ($n_{\mathrm{u}}$)",
                loc=legend_location,           # Anchor the bottom-left corner of the legend...
                bbox_to_anchor=legend_anchor,  # ...at X=1.1, and Y=0.4 (safely above the infobox's 0.1)
                fontsize=9,
                ncol=adaptive_ncol
            )
            if add_infobox:
                try:
                    add_filter_infobox(
                        ax=axes[-1],                # Anchor it to the exact same axes
                        df=df_nobs, 
                        properties=COLUMN_PROPERTIES,
                        x_pos=1.1,                  # Perfectly aligns with the legend's left edge
                        y_pos=0.1,                  # Sits at the bottom, below the legend
                    )
                except NameError:
                    pass
            


        # Render the figure for the current `nobs`
        plt.tight_layout()
        plt.show()
        fig_list.append(fig)
    return fig_list
        
        
def plot_error_decay_slopes_old(df, err_type, best_init_metric, mean_or_median, center_X = True, plot_info_box = True):
    """
    Fits a linear model (err ~ baseline + slope * n_known) for each n_unknown strata,
    then plots slopes and intercepts in separate subplots against n_unknown.
    
    Features dynamic background text annotations representing "improving" vs "worsening" 
    trajectories and baseline values based on the metric evaluation rules.
    """
    df = df.copy()
    fig_list = []
    #DATA SETS PREPARATION
    # Isolate the best initialization paths
    group_cols = get_columns_by_type('base_identifier')
    if lower_is_better(best_init_metric):
        best_per_init = df.loc[df.groupby(group_cols)[best_init_metric].idxmin()]
    else:
        best_per_init = df.loc[df.groupby(group_cols)[best_init_metric].idxmax()]
    
    exclude = {'init_data', 'data_seed'}
    across_data_init_group = [col for col in group_cols if col not in exclude]
    
    # Average out individual dataset variation
    grouped = best_init_path(df = df, best_init_metric=best_init_metric)
    #across_data_stat(df = df, err_type=err_type, best_init_metric = best_init_metric, stat = mean_or_median)


    #FITTING LINEAR MODEL        
    # Fit linear models per strata to extract both slopes and intercepts
    trend_results = []
    
    for (method, nobs, n_un), sub_df in grouped.groupby(["method", "nobs", "n_unknown"]):
        sub_df = sub_df.sort_values("n_known_src")
        
        # We need at least 2 points to fit a reliable trajectory line
        if len(sub_df) > 1:
            Y = sub_df[err_type].values
            X = sub_df["n_known_src"].values
            if(center_X):
                X_centered = X - (X.max() + X.min()) / 2 
                # Simple linear regression: Y = slope * X + intercept
                slope, intercept = np.polyfit(X_centered, Y, 1)
            else:
                slope, intercept = np.polyfit(X, Y, 1)
            
            trend_results.append({
                "method": method,
                "nobs": nobs,
                "n_unknown": n_un,
                "slope": slope,
                "intercept": intercept,
                'err': err_type
            })
            
    df_trends = pd.DataFrame(trend_results)
    
    if df_trends.empty:
        print("Insufficient data variance to compute linear model trends.")
        return
        
    # 4. Plotting - Separate figures per sensor dimension (nobs) with 2 subplots each
    unique_nobs = sorted(df_trends["nobs"].unique())
    
    for nobs in unique_nobs:
        df_nobs = df_trends[df_trends["nobs"] == nobs]
        
        # Create a side-by-side figure layout
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
        
        # Draw lines for each estimation method across both axes
        for method, df_method in df_nobs.groupby("method"):
            df_method = df_method.sort_values("n_unknown")
            
            # Formatting configurations
            marker = method_marker(method) if 'method_marker' in globals() else 'o'
            label = method_label(method) if 'method_label' in globals() else method
            color = method_color(method) if 'method_color' in globals() else None
            linestyle = method_linestyle(method) if 'method_linestyle' in globals() else '-'
            
            # Subplot 1: Slopes
            ax1.plot(
                df_method["n_unknown"],
                df_method["slope"],
                marker=marker,
                linewidth=2,
                label=label,
                color=color,
                linestyle=linestyle
            )
            
            # Subplot 2: Intercepts
            ax2.plot(
                df_method["n_unknown"],
                df_method["intercept"],
                marker=marker,
                linewidth=2,
                label=label,
                color=color,
                linestyle=linestyle
            )
         
   
        # =================================================================
        # SUBPLOT 1: Slopes Customizations
        # =================================================================
        # Reference line at zero
        ax1.axhline(0, color="black", linestyle="--", alpha=0.6, linewidth=1)
        
        # Mark the complete case boundary line
        ax1.axvline(nobs, color="#FF3D00", linestyle=":", alpha=0.7, linewidth=2)
        ax1.text(nobs +0.2, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.1, 
                 "↑ Complete Case Boundary", color="#FF3D00", fontweight="bold", rotation=90, alpha=0.8)
        
        # Dynamic Background Watermark Text for Slopes
        ymin1, ymax1 = ax1.get_ylim()
        xmin1, xmax1 = ax1.get_xlim()
        x_center1 = (xmin1 + xmax1) / 2
        
        if lower_is_better(err_type):
            above_text, above_color = "WORSENING PERFORMANCE", "#C62828"
            below_text, below_color = "IMPROVING PERFORMANCE", "#2E7D32"
        else:
            above_text, above_color = "IMPROVING PERFORMANCE", "#2E7D32"
            below_text, below_color = "WORSENING PERFORMANCE", "#C62828"
            
        if ymax1 > 0:
            ax1.text(x_center1, ymax1 * 0.45, above_text, color=above_color,
                    fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        if ymin1 < 0:
            ax1.text(x_center1, ymin1 * 0.45, below_text, color=below_color,
                    fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        
        ax1.set_xlabel("Number of Unknown Sources ($n_{\mathrm{u}}$)", fontsize=11)
        ax1.set_ylabel(f"Slope of {metric_label(err_type) if 'metric_label' in globals() else err_type} per Known Source", fontsize=11)
        
        slope_direction = "↓ negative is better" if lower_is_better(err_type) else "↑ positive is better"
        ax1.set_title(f"First-Order Performance Improvement For Fixed Latent Dimension\n({slope_direction})", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.5)

        # =================================================================
        # SUBPLOT 2: Intercepts Customizations
        # =================================================================
        ax2.axvline(nobs, color="#FF3D00", linestyle=":", alpha=0.7, linewidth=2)
        ax2.text(nobs+0.2, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.1, 
                 "↑ Complete Case Boundary", color="#FF3D00", rotation=90, alpha=0.8, fontweight = 'bold')
        # Define watermark evaluation strings for Intercepts
        if lower_is_better(err_type):
            int_above_text, int_above_color = "WORSE PERFORMANCE", "#C62828"
            int_below_text, int_below_color = "BETTER PERFORMANCE", "#2E7D32"
            intercept_direction = "↓ lower is better"
        else:
            int_above_text, int_above_color = "BETTER PERFORMANCE", "#2E7D32"
            int_below_text, int_below_color = "WORSE PERFORMANCE", "#C62828"
            intercept_direction = "↑ greater is better"
            

            
        # Using axes coordinate transform (0 to 1 scale) ensures watermarks display reliably
        ax2.text(0.5, 0.80, int_above_text, color=int_above_color, transform=ax2.transAxes,
                fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        ax2.text(0.5, 0.20, int_below_text, color=int_below_color, transform=ax2.transAxes,
                fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        
        ax2.set_xlabel("Number of Unknown Sources ($n_{\mathrm{u}}$)", fontsize=11)
        intercept_label = r"Expected Performance" if center_X else "Zero-Knowledge Baseline"
        ax2.set_ylabel(f"{metric_label(err_type) if 'metric_label' in globals() else err_type}", fontsize=11)
        ax2.set_title(f"{intercept_label}  \n({intercept_direction})", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.5)
        if plot_info_box:
            add_filter_infobox(ax = ax2, df = df, properties=COLUMN_PROPERTIES)
            ax2.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left')
        else:
            if lower_is_better(err_type):
                ax2.legend(bbox_to_anchor=(0, 1), loc='upper left')
            else:
                ax2.legend(bbox_to_anchor=(1, 1), loc='upper right')
        
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Overall figure styling adjustments
        fig.suptitle(f"Performance Trajectory ($d_{{x}} = {nobs}$) \n Statistic: {mean_or_median}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.show()
        fig_list.append(fig)
    return fig_list
def plot_error_decay_slopes(
    df, 
    err_type, 
    best_init_metric, 
    mean_or_median, 
    center_X=True, 
    plot_info_box=True, 
    error_bars='ci',  # <-- Options: None, 'se' (Standard Error), 'ci' (95% Confidence Interval),
    jitter = 0.15,
    error_alpha = 0.35
):
    """
    Fits a linear model (err ~ baseline + slope * n_known) for each n_unknown strata,
    then plots slopes and intercepts in separate subplots against n_unknown.
    Optionally displays Standard Errors or 95% Confidence Intervals as error bars.
    
    Returns:
        fig_list (list): A list of Matplotlib figure objects.
        df_coefs (pd.DataFrame): Clean table of coefficients, SEs, and 95% CI margins.
    """
    # 1. Isolate the best initialization paths and average dataset variation
    grouped = best_init_path(df=df, best_init_metric=best_init_metric)

    # 2. Fit linear models per strata
    trend_results = []
    for (method, nobs, n_un), sub_df in grouped.groupby(["method", "nobs", "n_unknown"]):
        n_samples = len(sub_df)
        if n_samples > 2:  # Requires N > 2 for degrees of freedom > 0
            sub_df = sub_df.sort_values("n_known_src")
            X = sub_df["n_known_src"].values
            Y = sub_df[err_type].values
            n_samples = len(sub_df)

            if center_X:
                X = X - (X.max() + X.min()) / 2 
                
            res = stats.linregress(X, Y)
            
            # Calculate 95% CI margins using Student's t-distribution
            df_deg = n_samples - 2
            t_crit = stats.t.ppf(0.975, df=df_deg)
            slope_ci = t_crit * res.stderr
            intercept_ci = t_crit * res.intercept_stderr
            
            trend_results.append({
                "method": method,
                "nobs": nobs,
                "n_unknown": n_un,
                "slope": res.slope,
                "intercept": res.intercept,
                "slope_se": res.stderr,
                "intercept_se": res.intercept_stderr,
                "slope_ci_margin": slope_ci,
                "intercept_ci_margin": intercept_ci,
                "n":n_samples
            })
        elif n_samples == 2:  # Exact fit fallback (undefined SE/CI)
            sub_df = sub_df.sort_values("n_known_src")
            X = sub_df["n_known_src"].values
            Y = sub_df[err_type].values
            if center_X:
                X = X - (X.max() + X.min()) / 2
            slope, intercept = np.polyfit(X, Y, 1)
            trend_results.append({
                "method": method,
                "nobs": nobs,
                "n_unknown": n_un,
                "slope": slope,
                "intercept": intercept,
                "slope_se": np.nan,
                "intercept_se": np.nan,
                "slope_ci_margin": np.nan,
                "intercept_ci_margin": np.nan,
                "n": n_samples
            })
            
    df_trends = pd.DataFrame(trend_results)
    if df_trends.empty:
        print("Insufficient data variance to compute linear model trends.")
        cols = ["method", "nobs", "nUnknown", "alpha", "alpha_se", "alpha_ci", "beta", "beta_se", "beta_ci"]
        return [], pd.DataFrame(columns=cols)
        
    # Helpers for dynamic global formatting fallbacks
    get_marker = lambda m: globals().get('method_marker', lambda x: 'o')(m)
    get_label = lambda m: globals().get('method_label', lambda x: x)(m)
    get_color = lambda m: globals().get('method_color', lambda x: None)(m)
    get_style = lambda m: globals().get('method_linestyle', lambda x: '-')(m)
    
    is_lower_better = lower_is_better(err_type)
    metric_name = globals().get('metric_label', lambda x: x)(err_type)
    fig_list = []
    
    # 3. Plotting - Separate figures per sensor dimension (nobs)
    for nobs in sorted(df_trends["nobs"].unique()):
        df_nobs = df_trends[df_trends["nobs"] == nobs]
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        
        # Identify unique methods to calculate a systematic horizontal offset
        unique_methods = sorted(df_nobs["method"].unique())
        num_methods = len(unique_methods)
        
        for idx, (method, df_method) in enumerate(df_nobs.groupby("method")):
            df_method = df_method.sort_values("n_unknown")
            # Calculate a unique X offset (jitter) for this method
            if num_methods > 1:
                offset = (idx - (num_methods - 1) / 2.0) * jitter
            else:
                offset = 0.0
            x_coords = df_method["n_unknown"] + offset
            
            style_kwargs = {
                "marker": get_marker(method),
                "label": get_label(method),
                "color": get_color(method),
                "linestyle": get_style(method),
                "linewidth": 2
            }
            
            # Determine error boundaries to plot
            yerr1, yerr2 = None, None
            if error_bars == 'se':
                yerr1 = df_method.get("slope_se")
                yerr2 = df_method.get("intercept_se")
            elif error_bars == 'ci':
                yerr1 = df_method.get("slope_ci_margin")
                yerr2 = df_method.get("intercept_ci_margin")
                
            if error_bars in ['se', 'ci'] and yerr1 is not None:
                #ax1.errorbar(x_coords, df_method["slope"], yerr=yerr1, capsize=4, elinewidth=1, **style_kwargs, alpha = 0.1)
                #ax2.errorbar(x_coords, df_method["intercept"], yerr=yerr2, capsize=4, elinewidth=1, **style_kwargs, alpha = 0.1)
                # 1. Generate the plots and capture the ErrorbarContainers
                eb1 = ax1.errorbar(x_coords, df_method["slope"], yerr=yerr1, capsize=4, elinewidth=1.5, **style_kwargs)
                eb2 = ax2.errorbar(x_coords, df_method["intercept"], yerr=yerr2, capsize=4, elinewidth=1.5, **style_kwargs)
                
                # 2. Modify transparency for the first subplot (Slopes)
                # eb1[1] contains the cap lines; eb1[2] contains the whisker collections
                for cap in eb1[1]:
                    cap.set_alpha(error_alpha)
                for bar in eb1[2]:
                    bar.set_alpha(error_alpha)
                    
                # 3. Modify transparency for the second subplot (Intercepts)
                for cap in eb2[1]:
                    cap.set_alpha(error_alpha)
                for bar in eb2[2]:
                    bar.set_alpha(error_alpha)
                
                
            else:
                ax1.plot(x_coords, df_method["slope"], **style_kwargs)
                ax2.plot(x_coords, df_method["intercept"], **style_kwargs)
         
        # Common layout and boundary styling
        for ax in (ax1, ax2):
            ax.axvline(nobs, color="#FF3D00", linestyle=":", alpha=0.7, linewidth=2)
            ax.text(nobs + 0.2, 0.1, "↑ Complete Case Boundary", color="#FF3D00", 
                    fontweight="bold", rotation=90, alpha=0.4, transform=ax.get_xaxis_transform(), zorder = 2)
            ax.set_xlabel("Number of Unknown Sources ($n_{\mathrm{u}}$)", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # --- Subplot 1: Slopes Specifics ---
        ax1.axhline(0, color="black", linestyle="--", alpha=0.6, linewidth=1)
        above_text, above_color = ("WORSENING PERFORMANCE", "#C62828") if is_lower_better else ("IMPROVING PERFORMANCE", "#2E7D32")
        below_text, below_color = ("IMPROVING PERFORMANCE", "#2E7D32") if is_lower_better else ("WORSENING PERFORMANCE", "#C62828")
        
        ymin1, ymax1 = ax1.get_ylim()
        lims = max([np.abs(ymin1), np.abs(ymax1)])
        ax1.set_ylim((-lims, lims))
        
        if ymax1 > 0.0:
            ax1.text(0.5, 0.75, above_text, color=above_color, transform=ax1.transAxes,
                     fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        if ymin1 < 0:
            ax1.text(0.5, 0.25, below_text, color=below_color, transform=ax1.transAxes,
                     fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        
        slope_dir = "↓ negative is improvement" if is_lower_better else "↑ positive is improvement"
        ax1.set_ylabel(f"Slope of {metric_name} per Known Source", fontsize=11)
        ax1.set_title(f"First-Order Performance Improvement For Fixed Latent Dimension " + r"($\beta_{n_{\mathrm{u}}}$)" + f"\n({slope_dir})", fontsize=12, fontweight="bold")

        # --- Subplot 2: Intercepts Specifics ---
        int_above_text, int_above_color = ("WORSE PERFORMANCE", "#C62828") if is_lower_better else ("BETTER PERFORMANCE", "#2E7D32")
        int_below_text, int_below_color = ("BETTER PERFORMANCE", "#2E7D32") if is_lower_better else ("WORSE PERFORMANCE", "#C62828")
        intercept_direction = "↓ lower is better" if is_lower_better else "↑ greater is better"
        
        ax2.text(0.5, 0.80, int_above_text, color=int_above_color, transform=ax2.transAxes,
                 fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        ax2.text(0.5, 0.20, int_below_text, color=int_below_color, transform=ax2.transAxes,
                 fontsize=12, fontweight="bold", alpha=0.4, ha="center", va="center", zorder=0)
        
        intercept_label = r"Expected Performance ($\alpha_{n_{\mathrm{u}}}$)" if center_X else "Zero-Knowledge Baseline"
        ax2.set_ylabel(metric_name, fontsize=11)
        ax2.set_title(f"{intercept_label}\n({intercept_direction})", fontsize=12, fontweight="bold")
        
        if plot_info_box:
            add_filter_infobox(ax=ax2, df=df, properties=COLUMN_PROPERTIES)
            ax2.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left')
        else:
            loc, bbox = ('upper left', (0, 1)) if is_lower_better else ('upper right', (1, 1))
            ax2.legend(bbox_to_anchor=bbox, loc=loc)
        
        #fig.suptitle(f"Performance Trajectory ($d_{{x}} = {nobs}$) \n Statistic: {mean_or_median}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.show()
        fig_list.append(fig)
        
    # 4. Generate final coefficient dataframe with standard errors and 95% CI margins
    df_coefs = df_trends.rename(columns={
        "n_unknown": "nUnknown",
        "intercept": "alpha",
        "slope": "beta",
        "intercept_se": "alpha_se",
        "slope_se": "beta_se",
        "intercept_ci_margin": "alpha_ci",
        "slope_ci_margin": "beta_ci"
    })[["method", "nobs", "nUnknown", "alpha", "alpha_se", "alpha_ci", "beta", "beta_se", "beta_ci", "n"]]
    
    return fig_list, df_coefs
def plot_convergence_by_xxx(df, x_axis, err_type, err_filter_selector=None, min_val = None, max_val = None):
    """
    Plots convergence of err_type across nreps per method.

    Parameters
    ----------
    df                  : DataFrame 
    x_axis              : str   - column name of the property to plot on x-axis
    err_type            : str   — column name of the error metric to plot
    err_filter_selector : dict  — optional filter e.g. {'n_known_src': 0, 'nobs': 10}
    """
    

    df = df.copy()

    # ── Optional filter ───────────────────────────────────────────────────────
    if err_filter_selector is not None:
        for col, val in err_filter_selector.items():
            df = df[df[col] == val]

    if df.empty:
        print('No data after filtering.')
        return

    # ── Step 1 & 2: best err_type per (method, init_data, nreps) ─────────────
    agg_fn  = 'min' if lower_is_better(err_type) else 'max'
    best_per_init = (
        df.groupby(['method', 'init_data', x_axis])[err_type]
        .agg(agg_fn)
        .reset_index()
    )

    # Average and std across init_data
    stats = (
    best_per_init.groupby(['method', x_axis])
    .agg(mean=(err_type, 'mean'), std=(err_type, 'std'))
    .reset_index()
)

    # ── Step 3 & 4: plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, grp in stats.groupby('method'):
        grp  = grp.sort_values(x_axis)
        x    = grp[x_axis].values
        y    = grp['mean'].values
        yerr = grp['std'].values

        style = METHOD_STYLES.get(method, {'color': 'grey', 'linestyle': '-'})
        color     = style.get('color',     'grey')
        linestyle = style.get('linestyle', '-')
        marker    = style.get('marker',    'o')

        ax.scatter(x, y,
                color=color, 
                #linestyle=linestyle, 
                marker=marker,
                label=method, 
                #linewidth=1.8, 
                #markersize=8
                )

        ax.fill_between(x,
                        y - yerr,
                        y + yerr,
                        color=color, alpha=0.15)

        lower_bound = metric_bounds(err_type)[0]*1.05 if min_val is None else min_val
        upper_bound = metric_bounds(err_type)[1]*1.05 if max_val is None else max_val

        ax.set_ylim(bottom = lower_bound,
                    top = upper_bound)
    # ── Formatting ────────────────────────────────────────────────────────────
    direction = '↓ smaller is better' if lower_is_better(err_type) else '↑ greater is better'
    filter_str = ''
    if err_filter_selector:
        filter_str = ' | ' + ', '.join(f'{k}={v}' for k, v in err_filter_selector.items())

    ax.set_xlabel(x_axis, fontsize=11)
    ax.set_ylabel(metric_label(err_type), fontsize=11)
    ax.set_title(f'{metric_label(err_type)} vs {x_axis} ({direction}){filter_str}', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc='right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
#HELPERS
def add_filter_infobox(ax, df, properties, x_pos=1.05, y_pos=1.0):
    """
    Scans the dataframe for 'filter_col' properties and generates an infobox 
    placed OUTSIDE the right edge of the axes. Alerts the user visually 
    if any filter contains mixed data.
    """
    filter_cols = [col for col, meta in properties.items() if meta.get('type') == 'filter_col']
    
    lines = []
    any_mixed = False
    
    lines.append(f"Dataset Filters (N={len(df)}):")
    lines.append("-" * 25)
    
    for col in filter_cols:
        if col in df.columns:
            label = properties.get(col, {}).get('label', col)
            unique_vals = df[col].dropna().unique()
            
            if len(unique_vals) == 0:
                val_str = "Empty/NaN"
            elif len(unique_vals) == 1:
                val_str = str(unique_vals[0])
            else:
                val_str = ", ".join(map(str, unique_vals))
                val_str = f"⚠️ {val_str} [MIXED]"
                any_mixed = True
                
            lines.append(f"{label}: {val_str}")
            
    text_str = "\n".join(lines)
    
    if any_mixed:
        box_props = dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1.5, alpha=0.9)
    else:
        box_props = dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CED4DA', linewidth=1, alpha=0.8)
    ax.text(x_pos, y_pos, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left', bbox=box_props, zorder=10)
   
   

def best_init_path(df, best_init_metric) -> pd.DataFrame: 
     # Isolate the best initialization paths
    group_cols = get_columns_by_type('base_identifier') + get_columns_by_type('filter_col')

   
    if lower_is_better(best_init_metric):
        best_per_init = df.loc[df.groupby(group_cols)[best_init_metric].idxmin()]
    else:
        best_per_init = df.loc[df.groupby(group_cols)[best_init_metric].idxmax()]
    
    
        
    return best_per_init
 
def across_data_stat(df: pd.DataFrame, best_init_metric: str, err_type: str = None, stat: str = 'mean') -> pd.DataFrame:
    """
    Isolates the best initialization paths, then calculates a specified 
    statistic to average out individual dataset variations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw experiment results DataFrame.
    best_init_metric : str
        The metric column used to determine the 'best' initialization run.
    err_type : str or list, optional
        Specific error column(s) to isolate. If None, computes the statistic 
        across all numeric columns.
    stat : str, default 'mean'
        The pandas GroupBy statistic to compute. 
        Supported values include: 'mean', 'median', 'std', 'min', 'max', 'var'.
        
    Returns:
    --------
    pd.DataFrame
        The aggregated statistic grouped by the non-seed identifier columns.
    """
    # 1. Identify the structural identifier columns, excluding seed variations
    group_cols = get_columns_by_type('base_identifier') + get_columns_by_type('filter_col')
    exclude = {'init_data', 'data_seed'}
    across_data_init_group = [col for col in group_cols if col not in exclude]
    
    # 2. Extract the best initialization paths using our robust helper function
    # We pass best_init_metric and let it select the optimal seed runs
    df_best = best_init_path(df=df, best_init_metric=best_init_metric)
     
    # 3. Create the GroupBy object
    groupby_obj = df_best.groupby(across_data_init_group)
    
    # 4. Dynamically retrieve and apply the requested statistic
    if err_type:
        # Isolate the specific error column(s), retrieve the statistic method, and call it
        stat_func = getattr(groupby_obj[err_type], stat)
        grouped = stat_func().reset_index()
    else:
        # Retrieve the statistic method for the whole group, enforcing numeric-only calculations
        stat_func = getattr(groupby_obj, stat)
        grouped = stat_func(numeric_only=True).reset_index()
        
    return grouped
     
