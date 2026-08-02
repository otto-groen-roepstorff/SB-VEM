import pandas as pd
import numpy as np
import overcomplete_learning.metrics as ol_metric
import overcomplete_learning.plotting   as ol_plot

def generate_error_latex_table(df: pd.DataFrame, error_metrics: list, err_type, best_init_metric) -> str:
    """
    Generates a LaTeX table summarizing error metrics per method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing a 'method' column and error metric columns.
        Should only contain one parameter initialization per data initialization
    error_metrics : list of str
        A list of the column names you want to summarize (e.g., ['A_err', 'S_err_MSE']).
        
    Returns:
    --------
    str
        The LaTeX code for the summary table.
    """
        
    # 2. Define the statistics we want to calculate
    # We map pandas built-in strings to their names, and include our custom IQR function
    agg_functions = {
        'mean': 'mean',
        'median': 'median',
        'min': 'min',
        'max': 'max',
        'std': 'std',
        ol_metric.calculate_iqr: 'IQR'
    }
    
    # 3. Group by 'method' and calculate the statistics for the specified error metrics
    # We use a list of functions so pandas applies all of them to each column
    funcs_to_apply = [np.mean, np.median, np.min, np.max, np.std, ol_metric.calculate_iqr]

    summary_df = df.groupby('method')[error_metrics].agg(funcs_to_apply)
    
    # 4. Reshape the DataFrame for better readability in LaTeX
    # This moves the error metrics into the rows (grouped under each method)
    # and keeps the statistics as the columns.
    summary_df = summary_df.stack(level=0, future_stack=True)
    summary_df.index.names = ['Method', 'Error Metric']
    
    # Rename the statistic columns for a cleaner table header
    summary_df.columns = ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'IQR']
    
    # 5. Format the numbers to 4 decimal places to keep the LaTeX table neat
    styled_df = summary_df.style.format("{:.4f}")
    
    # 6. Convert to LaTeX using the pandas Styler
    # 'hrules=True' adds professional-looking top and bottom borders (requires \usepackage{booktabs} in your LaTeX document)
    latex_string = styled_df.to_latex(hrules=True)
    
    return latex_string

def generate_condensed_latex_table(df: pd.DataFrame, error_metrics: list) -> str:
    """
    Generates a LaTeX table summarizing error metrics per method.
    Outputs three columns: 'Median (Min, Max)', 'Mean (\\pm Std)', and 'IQR'.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing a 'method' column and error metric columns.
    error_metrics : list of str
        A list of the column names you want to summarize.
        
    Returns:
    --------
    str
        The LaTeX code for the condensed summary table.
    """
    # 1. Ensure the error metrics are strictly numeric
    for col in error_metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 2. Calculate raw statistics grouped by method
    # Pandas will name the custom function column after the function's name (e.g., 'calculate_iqr')
    raw_stats = df.groupby('method')[error_metrics].agg(
        ['mean', 'median', 'min', 'max', 'std', ol_metric.calculate_iqr]
    )
    
    # Extract the name pandas gave the IQR column dynamically
    iqr_col_name = ol_metric.calculate_iqr.__name__
    
    formatted_columns = []
    
    # 3. Process each error metric to build the formatted string columns
    for metric in error_metrics:
        if metric not in raw_stats.columns.levels[0]:
            continue
            
        # Isolate the stats for this specific metric
        stats = raw_stats[metric].copy()
        
        label = ol_plot.metric_latex(metric)
        
        # Format every number to 2 decimal places, handling potential NaNs safely
        for col in stats.columns:
            stats[col] = stats[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "NaN")
            
        # 4. Construct the custom string formats
        # Using LaTeX math mode ($\pm$) for the plus-minus symbol
        median_combined = stats['median'] + " (" + stats['min'] + ", " + stats['max'] + ")"
        mean_combined   = stats['mean'] + " ($\\pm$ " + stats['std'] + ")"
        
        # Pull the correctly formatted IQR data from the stats dataframe
        IQR_comb        = stats[iqr_col_name]
        
        # Store them in a temporary DataFrame with a MultiIndex so we can reshape later
        metric_df = pd.DataFrame({
            ('Median (Min, Max)', label): median_combined,
            ('Mean ($\\pm$ Std)', label): mean_combined,
            ('IQR', label)              : IQR_comb
        })
        formatted_columns.append(metric_df)
        
    # 5. Bring all formatted metrics together
    final_df = pd.concat(formatted_columns, axis=1)
    
    # 6. Reshape so Methods and Error Metrics are on the rows
    final_df.columns.names = ['Statistic', 'Error Metric']
    reshaped_df = final_df.stack(level='Error Metric', future_stack=True)
    
    
    # Ensure columns are in the exact order requested, now including 'IQR'
    reshaped_df = reshaped_df[['Median (Min, Max)', 'Mean ($\\pm$ Std)', 'IQR']]
    reshaped_df = reshaped_df.rename_axis(index={'method': 'Method', 'Statistic': ''})
    reshaped_df = reshaped_df.rename(index=ol_plot.method_latex, level='Method')
    # 7. Convert to LaTeX
    # Because our data is already heavily formatted strings, we just output directly
    latex_string = reshaped_df.style.to_latex(hrules=True, column_format='llrrr')
    
    
    return latex_string
def generate_coefficient_latex_table(df_coefs: pd.DataFrame, decimal_places: int = 3) -> str:
    """
    Generates a LaTeX table summarizing linear model coefficients (alpha, beta) 
    and their standard errors combined into a single column "Estimate (SE)".
    
    Parameters:
    -----------
    df_coefs : pd.DataFrame
        The input coefficient dataframe returned by `plot_error_decay_slopes`.
    decimal_places : int
        The precision formatting for numerical values.
        
    Returns:
    --------
    str
        The compiled LaTeX tabular code with combined mathematical column headers.
    """
    # 1. Protect original data
    df = df_coefs.copy()
    
    # Handle minor spelling differences in columns safely
    if 'nUknown' in df.columns and 'nUnknown' not in df.columns:
        df = df.rename(columns={'nUknown': 'nUnknown'})
        
    # 2. Sort numerically BEFORE converting numeric types to formatted strings.
    sort_cols = [col for col in ['method', 'nobs', 'nUnknown'] if col in df.columns]
    df = df.sort_values(by=sort_cols)
    
    # 3. Map method labels using global lookups if they exist
    try:
        get_label = ol_plot.method_latex
        df['method'] = df['method'].apply(get_label)
    except NameError:
        pass  # Fallback if ol_plot is not defined globally
    
    # 4. Format and Combine Estimate + SE into a single "Value (SE)" string
    fmt = f"{{:.{decimal_places}f}}"
    
    def combine_val_se(val, se):
        if pd.isnull(val) or np.isnan(val):
            return "-"
        val_str = fmt.format(val)
        if pd.isnull(se) or np.isnan(se):
            return val_str  # Fallback if SE is not available (e.g., exact N=2 fits)
        return f"{val_str} ({fmt.format(se)})"

    # Safely create combined columns
    if 'alpha' in df.columns:
        alpha_se_col = 'alpha_se' if 'alpha_se' in df.columns else None
        df['alpha_combined'] = df.apply(
            lambda r: combine_val_se(r['alpha'], r[alpha_se_col] if alpha_se_col else None), 
            axis=1
        )
        
    if 'beta' in df.columns:
        beta_se_col = 'beta_se' if 'beta_se' in df.columns else None
        df['beta_combined'] = df.apply(
            lambda r: combine_val_se(r['beta'], r[beta_se_col] if beta_se_col else None), 
            axis=1
        )
    
    # Ensure parameter boundaries render as clean integers
    for col in ['nobs', 'nUnknown']:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{int(x)}" if pd.notnull(x) else "-")
    
    # 5. Map columns to merged LaTeX math headers
    col_mapping = {
        'method': 'Method',
        'nobs': r'$\obsdim$',
        'nUnknown': r'$\nUnknown$',
        'alpha_combined': r'$\alpha_{\nUnknown} \ (\text{SE}_{\alpha_{\nUnknown}})$',
        'beta_combined': r'$\beta_{\nUnknown} \ (\text{SE}_{\beta_{\nUnknown}})$',
        'Error': 'Error',
        "n" :   "n"
    }
    
    df = df.rename(columns=col_mapping)
    
    # 6. Keep only target columns in logical layout order
    target_cols = ['method', 'nobs', 'nUnknown', 'alpha_combined', 'beta_combined', 'Error', "n"]
    ordered_cols = [
        col_mapping[c] 
        for c in target_cols 
        if c in col_mapping and col_mapping[c] in df.columns
    ]
    df = df[ordered_cols]
    
    # 7. Build column alignments (l = left, r = right)
    align_map = {
        'Method': 'l',
        'Error': 'l',
        r'$\obsdim$': 'r',
        r'$\nUnknown$': 'r',
        r'$\alpha_{\nUnknown} \ (\text{SE}_{\alpha_{\nUnknown}})$': 'r',
        r'$\beta_{\nUnknown} \ (\text{SE}_{\beta_{\nUnknown}})$': 'r'
    }
    col_format = "".join([align_map.get(col, 'l') for col in ordered_cols])
    
    # 8. Convert to clean LaTeX with booktabs rules, hiding the row index
    latex_table = df.style.hide(axis='index').to_latex(hrules=True, column_format=col_format)
    
    return latex_table
    """
    Generates a LaTeX table summarizing linear model coefficients (alpha, beta) 
    and their standard errors (alpha_se, beta_se) per method and latent/unknown 
    source dimensions.
    
    Parameters:
    -----------
    df_coefs : pd.DataFrame
        The input coefficient dataframe returned by `plot_error_decay_slopes`.
    decimal_places : int
        The precision formatting for numerical values.
        
    Returns:
    --------
    str
        The compiled LaTeX tabular code with customized mathematical column headers.
    """
    # 1. Protect original data
    df = df_coefs.copy()
    
    # Handle minor spelling differences in columns safely
    if 'nUknown' in df.columns and 'nUnknown' not in df.columns:
        df = df.rename(columns={'nUknown': 'nUnknown'})
        
    # 2. FIX: Sort numerically BEFORE converting numeric types to formatted strings.
    # (Otherwise, alphabetical sorting places "11" before "2")
    sort_cols = [col for col in ['method', 'nobs', 'nUnknown'] if col in df.columns]
    df = df.sort_values(by=sort_cols)
    
    # 3. Map method labels using global lookups if they exist
    try:
        get_label = ol_plot.method_latex
        df['method'] = df['method'].apply(get_label)
    except NameError:
        # Fallback if ol_plot is not defined globally
        pass
    
    # 4. Format numeric values for display
    fmt = f"{{:.{decimal_places}f}}"
    for col in ['alpha', 'alpha_se', 'beta', 'beta_se']:
        if col in df.columns:
            df[col] = df[col].map(lambda x: fmt.format(x) if pd.notnull(x) else "-")
    
    # Ensure parameter boundaries render as clean integers
    for col in ['nobs', 'nUnknown']:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{int(x)}" if pd.notnull(x) else "-")
    
    # 5. Map columns to LaTeX command structures (Including Standard Errors)
    col_mapping = {
        'method': 'Method',
        'nobs': r'$\obsdim$',
        'nUnknown': r'$\nUnknown$',
        'alpha': r'$\alpha_{\nUnknown}$',
        'alpha_se': r'$\text{SE}(\alpha_{\nUnknown})$',
        'beta': r'$\beta_{\nUnknown}$',
        'beta_se': r'$\text{SE}(\beta_{\nUnknown})$',
        'Error': 'Error'
    }
    
    df = df.rename(columns=col_mapping)
    
    # 6. Keep only target columns in the requested logical order
    target_cols = ['method', 'nobs', 'nUnknown', 'alpha', 'alpha_se', 'beta', 'beta_se', 'Error']
    ordered_cols = [
        col_mapping[c] 
        for c in target_cols 
        if c in col_mapping and col_mapping[c] in df.columns
    ]
    df = df[ordered_cols]
    
    # 7. Dynamically build column format alignments (l = left, r = right)
    align_map = {
        'Method': 'l',
        'Error': 'l',
        r'$\obsdim$': 'r',
        r'$\nUnknown$': 'r',
        r'$\alpha_{\nUnknown}$': 'r',
        r'$\text{SE}(\alpha_{\nUnknown})$': 'r',
        r'$\beta_{\nUnknown}$': 'r',
        r'$\text{SE}(\beta_{\nUnknown})$': 'r'
    }
    col_format = "".join([align_map.get(col, 'l') for col in ordered_cols])
    
    # 8. Convert to clean LaTeX with booktabs rules, hiding the row index
    latex_table = df.style.hide(axis='index').to_latex(hrules=True, column_format=col_format)
    
    return latex_table

def generate_multi_coefficient_latex_table(
    dfs, 
    decimal_places: int = 2, 
    include_se: bool = False
) -> str:
    """
    Combines multiple trajectory coefficient DataFrames, pivots them by their 'Error' 
    type, and generates a hierarchical multi-column LaTeX table using standard Pandas.
    
    Parameters:
    -----------
    dfs : list of pd.DataFrame or pd.DataFrame
        A single DataFrame, or list of DataFrames to be combined.
    decimal_places : int
        Decimal formatting precision.
    include_se : bool
        If True, displays values as 'Estimate (SE)'. If False, displays raw Estimates.
    """
    # 1. Combine data if a list is provided
    df = pd.concat(dfs, ignore_index=True) if isinstance(dfs, list) else dfs.copy()

    # Safety rename for minor column spelling variations
    if 'nUknown' in df.columns and 'nUnknown' not in df.columns:
        df = df.rename(columns={'nUknown': 'nUnknown'})
        
    # 2. Map Method labels using your global plotting lookups
    try:
        get_label = ol_plot.method_latex
        df['method'] = df['method'].apply(get_label)
    except NameError:
        pass  # Fallback if ol_plot isn't available in scope
    
    # Sort logically before converting everything to text strings
    df = df.sort_values(by=['method', 'nobs', 'nUnknown'])

    # 3. Format value cells (and optionally combine with standard errors)
    dec_fmt = f"{{:.{decimal_places}f}}"
    
    def format_cell(val, se):
        if pd.isnull(val) or np.isnan(val): 
            return "-"
        val_str = dec_fmt.format(val)
        if not include_se or pd.isnull(se) or np.isnan(se):
            return val_str
        return f"{val_str} ({dec_fmt.format(se)})"

    alpha_se = 'alpha_se' if 'alpha_se' in df.columns else None
    beta_se = 'beta_se' if 'beta_se' in df.columns else None
    
    df['alpha_fmt'] = df.apply(lambda r: format_cell(r['alpha'], r[alpha_se] if alpha_se else None), axis=1)
    df['beta_fmt'] = df.apply(lambda r: format_cell(r['beta'], r[beta_se] if beta_se else None), axis=1)

    # 4. Standard Pivot & Level Restructuring
    # This automatically groups columns by Error type and nests alpha/beta underneath them!
    pivoted = df.pivot(
        index=['method', 'nobs', 'nUnknown', 'n'], 
        columns='Error', 
        values=['alpha_fmt', 'beta_fmt']
    )
    
    # Put 'Error' as the primary header level and parameter (alpha/beta) as the sub-header
    pivoted.columns = pivoted.columns.reorder_levels([1, 0])
    pivoted = pivoted.sort_index(axis=1, level=0)

    # Rename parameters to their LaTeX math equivalents
    pivoted = pivoted.rename(columns={
        'alpha_fmt': r'$\alpha_{\nUnknown}$',
        'beta_fmt': r'$\beta_{\nUnknown}$'
    }, level=1)

    # 5. Flatten the structure and rename the index columns
    reset_df = pivoted.reset_index()
    reset_df = reset_df.rename(columns={
        'method': 'Method',
        'nobs': r'$\obsdim$',
        'nUnknown': r'$\nUnknown$'
    }, level=0)

    # 6. Build the column alignments (lcc for the descriptors, rr for every error sub-column)
    num_errors = len(df['Error'].unique())
    col_format = "lrrr" + "rr" * num_errors

    # 7. Use Pandas Styler to output beautiful, compilable LaTeX tables
    latex_table = reset_df.style.hide(axis='index').to_latex(
        hrules=True, 
        column_format=col_format
    )
    
    return latex_table