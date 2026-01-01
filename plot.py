import os
import numpy as np
#import pandas as pd
#import random
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from lifelines import KaplanMeierFitter
plt.rcParams['text.usetex'] = True

# Define directory paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
RESULTS_DIR = os.path.join(PATH, "results")
FIGURES_DIR = os.path.join(PATH, "figures")

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)



def KM_survival(time, status):
    # Create the Kaplan-Meier fitter object
    kmf = KaplanMeierFitter()

    # Fit the Kaplan-Meier estimator to the data
    kmf.fit(durations=time, event_observed=status)
    
    # Compute the survival function
    survival_function = kmf.survival_function_

    # Estimate the mean survival time (area under the curve)
    # Compute the area under the survival curve using numerical integration
    x_values = survival_function.index
    y_values = survival_function['KM_estimate']

    # Use the trapezoidal rule to estimate the area under the curve
    mean_survival_time = 0
    for x_i in range(len(x_values)-1):
        mean_survival_time += y_values[x_values[x_i]] * (x_values[x_i+1]-x_values[x_i])

    print(f"Mean survival time (area under the survival curve): {mean_survival_time:.1f}")

    # Plot the survival curve
    kmf.plot_survival_function()
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Survival Curve")
    plt.show()
    return mean_survival_time

def plot_cumu_regrets():
    cumu_regrets_largest_colab = np.loadtxt(os.path.join(RESULTS_DIR, 'cumu_regrets_largest_prediction.csv'), delimiter=',')
    
    cumu_regrets_largest_solo = np.loadtxt(os.path.join(RESULTS_DIR, 'cumu_regrets_solo_largest_prediction.csv'), delimiter=',')
	
    row_size = cumu_regrets_largest_colab.shape[0]
    fig = plt.figure(figsize=(15,9))
    ax_list = []
    rows = 2
    cols = 5
    tickLabel_FS = 13
    axisLabel_FS = 16
    title_FS = 18
    legendLabel_FS = 14
    
    for i in range(cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax_list.append(ax)
    for i in range(cols):
        ax = fig.add_subplot(rows, cols, cols + (i+1))
        ax_list.append(ax)


    ylim_min, ylim_max = 0, 2.9  
    for i in range(row_size):
        # pick the axes
        ax = ax_list[i]
		
        # x-axis for plotting
        trials = list(range(cumu_regrets_largest_colab.shape[1]))
        trials = [i+1 for i in trials]
        ax.plot(trials, np.cumsum(cumu_regrets_largest_colab[i,:]),
				marker='o', label='Collaborative')

		# Plot cumulative sum of regrets for solo
        ax.plot(trials, np.cumsum(cumu_regrets_largest_solo[i,:]),
				marker='s', label='Non-collaborative')
		
        if i >= 5:
            ax.set_xlabel("Number of Trials", fontsize=axisLabel_FS)
        
        if i == 0 or i == 5:
            ax.set_ylabel("Cumulative Regrets", fontsize=axisLabel_FS)
        
        ax.set_title("Printer ${}$".format(i+1), fontsize=title_FS)
        ax.set_xlim(0, 20)
        ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True)
		
        if i == 5:
            ax.legend(loc='upper left', fontsize=legendLabel_FS)
        
        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=tickLabel_FS)
        ax.tick_params(axis='both', which='minor', labelsize=tickLabel_FS)
			
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'Seq_matrix_completion_case_1.png'), format='png', dpi=400)

def plot_subset_seq_mc():
    cumu_regrets_largest_colab = np.loadtxt(os.path.join(RESULTS_DIR, 'sub_cumu_regrets_largest_prediction.csv'), delimiter=',')
    cumu_regrets_largest_solo = np.loadtxt(os.path.join(RESULTS_DIR, 'sub_cumu_regrets_solo_largest_prediction.csv'), delimiter=',')
    chosen_indicator_matrix = np.loadtxt(os.path.join(RESULTS_DIR, 'sub_cumu_chosen.csv'), delimiter=',')
    chosen_indicator_matrix_no_colab = np.loadtxt(os.path.join(RESULTS_DIR, 'sub_cumu_chosen_no_colab.csv'), delimiter=',')
	
    row_size = cumu_regrets_largest_colab.shape[0]
    fig = plt.figure(figsize=(15,9)) # 15,11
    ax_list = []
    rows = 2
    cols = 5
    tickLabel_FS = 13
    axisLabel_FS = 16
    title_FS = 18
    legendLabel_FS = 14
    
    for i in range(cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax_list.append(ax)
    for i in range(cols):
        ax = fig.add_subplot(rows, cols, cols + (i+1))
        ax_list.append(ax)
    
    ylim_min, ylim_max = 0, 1.5
    for i in range(row_size):
		# pick the axes
        ax = ax_list[i]
		
		# x-axis for plotting
        trials = list(range(cumu_regrets_largest_colab.shape[1]))
        trials = [i+1 for i in trials]
		
		# Plot cumulative sum of regrets for collaboration
        # Create full regret array with forward-fill for not-selected trials
        regret_colab_full = np.zeros(len(trials))
        last_regret_colab = 0
        for trial_idx in range(len(trials)):
            if chosen_indicator_matrix[i, trial_idx] == 1:
                # Selected: use actual cumulative regret
                regret_colab_full[trial_idx] = cumu_regrets_largest_colab[i, trial_idx]
                last_regret_colab = regret_colab_full[trial_idx]
            else:
                # Not selected: carry forward last regret value
                regret_colab_full[trial_idx] = last_regret_colab

        # Separate chosen and not-chosen trials for marker styling
        chosen_idxs_colab = np.where(chosen_indicator_matrix[i,:] == 1)[0]
        not_chosen_idxs_colab = np.where(chosen_indicator_matrix[i,:] == 0)[0]

        # Plot all trials with line
        ax.plot(trials, regret_colab_full, '-', color='tab:blue', linewidth=1)
        # Plot chosen trials with solid circles
        ax.plot(chosen_idxs_colab + 1, regret_colab_full[chosen_idxs_colab],
                'o', color='tab:blue', markerfacecolor='tab:blue', markersize=6, label='Collab.')
        # Plot not-chosen trials with hollow circles (no label - will be in custom legend)
        ax.plot(not_chosen_idxs_colab + 1, regret_colab_full[not_chosen_idxs_colab],
                'o', color='tab:blue', markerfacecolor='none', markeredgewidth=1.5, markersize=6)

		# Plot cumulative sum of regrets for solo
        # Create full regret array with forward-fill for not-selected trials
        regret_solo_full = np.zeros(len(trials))
        last_regret_solo = 0
        for trial_idx in range(len(trials)):
            if chosen_indicator_matrix_no_colab[i, trial_idx] == 1:
                # Selected: use actual cumulative regret
                regret_solo_full[trial_idx] = cumu_regrets_largest_solo[i, trial_idx]
                last_regret_solo = regret_solo_full[trial_idx]
            else:
                # Not selected: carry forward last regret value
                regret_solo_full[trial_idx] = last_regret_solo

        # Separate chosen and not-chosen trials for marker styling
        chosen_idxs_solo = np.where(chosen_indicator_matrix_no_colab[i,:] == 1)[0]
        not_chosen_idxs_solo = np.where(chosen_indicator_matrix_no_colab[i,:] == 0)[0]

        # Plot all trials with line
        ax.plot(trials, regret_solo_full, '-', color='tab:orange', linewidth=1)
        # Plot chosen trials with solid squares
        ax.plot(chosen_idxs_solo + 1, regret_solo_full[chosen_idxs_solo],
                's', color='tab:orange', markerfacecolor='tab:orange', markersize=6, label='Non-collab.')
        # Plot not-chosen trials with hollow squares (no label - will be in custom legend)
        ax.plot(not_chosen_idxs_solo + 1, regret_solo_full[not_chosen_idxs_solo],
                's', color='tab:orange', markerfacecolor='none', markeredgewidth=1.5, markersize=6)
		
		
        if i >= 5:
            ax.set_xlabel("Number of Trials", fontsize=axisLabel_FS)
        
        if i == 0 or i == 5:
            ax.set_ylabel("Cumulative Regrets", fontsize=axisLabel_FS)
        
        ax.set_title("Printer ${}$".format(i+1), fontsize=title_FS)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xlim(0,20)
        ax.grid(True)

        if i == 5:
            # Create custom legend with combined "Not Selected" entry using HandlerTuple

            # Create line handles for the legend
            h_collab = Line2D([0], [0], marker='o', color='tab:blue', linestyle='-',
                             markerfacecolor='tab:blue', markersize=6)
            h_noncollab = Line2D([0], [0], marker='s', color='tab:orange', linestyle='-',
                                markerfacecolor='tab:orange', markersize=6)
            h_not_selected_circle = Line2D([0], [0], marker='o', color='tab:blue', linestyle='none',
                                          markerfacecolor='none', markeredgewidth=1.5, markersize=6)
            h_not_selected_square = Line2D([0], [0], marker='s', color='tab:orange', linestyle='none',
                                          markerfacecolor='none', markeredgewidth=1.5, markersize=6)

            # Create legend with tuple for "Not Selected"
            ax.legend(
                handles=[h_collab, h_noncollab, (h_not_selected_circle, h_not_selected_square)],
                labels=['Collab.', 'Non-collab.', 'Not Selected'],
                handler_map={tuple: HandlerTuple(ndivide=None)},
                handlelength=4,
                loc='upper left',
                fontsize=legendLabel_FS
            )
        
        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=tickLabel_FS)
        ax.tick_params(axis='both', which='minor', labelsize=tickLabel_FS)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'Sub_Seq_matrix_completion_case_2.png'), format='png', dpi=400)


def KM_survival_seq_mc():
    t_collab = np.array([19, 5, 1, 18, 14, 19, 1, 1, 9, 5])
    e_collab = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1]) # 1 = event observed, 0 = censored
    mu_hat = KM_survival(t_collab, e_collab)
    
    t_noncollab = np.array([19, 2, 19, 12, 15, 14, 14, 19, 19, 19])
    e_noncollab = np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 0]) # 1 = event observed, 0 = censored
    mu_hat = KM_survival(t_noncollab, e_noncollab)

def process_column_for_km(data_column, M):
    """
    Process a specific column for Kaplan-Meier survival analysis.

    Parameters:
    - data_column: Column from imported data (e.g., data1[:, 4])
    - M: Value to replace zeros in t_vec

    Returns:
    - mean_survival_time: Calculated mean survival time from KM_survival
    """
    # Step 1: Define t_vec and create e_vec (ones vector of same size)
    t_vec = data_column.copy()
    e_vec = np.ones(t_vec.shape, dtype=int)

    # Step 2: Find locations where t_vec is zero
    zero_locations = np.where(t_vec == 0)[0]
    # Replace zeros with M
    t_vec[zero_locations] = M
    # Change corresponding locations in e_vec to zero (censored)
    e_vec[zero_locations] = 0
    # Just in-case: handle scenarios when the optimal is included in the training set
    # This should never happen
    predone_locations = np.where(t_vec == -1)[0]
    t_vec[predone_locations] = 1

    # Step 3: Calculate mean_survival_time
    kmf = KaplanMeierFitter()
    kmf.fit(durations=t_vec, event_observed=e_vec)

    survival_function = kmf.survival_function_
    x_values = survival_function.index
    y_values = survival_function['KM_estimate']

    mean_survival_time = 0
    for x_i in range(len(x_values)-1):
        mean_survival_time += y_values[x_values[x_i]] * (x_values[x_i+1]-x_values[x_i])

    # Add the final segment from last observed time to M
    # The survival probability remains constant after the last event
    if len(x_values) > 0 and x_values[-1] < M:
        mean_survival_time += y_values[x_values[-1]] * (M - x_values[-1])

    return mean_survival_time


def plot_combined_sim_cases(configs):
    """
    Create 3 horizontally stacked subplots with shared y-axes.

    Parameters:
    -----------
    configs : list of dict
        List of 4 configuration dictionaries, each containing:
        - file1, file12, file2, file22, file3, file32: data files
        - xlabel, ylabel1, ylabel2: axis labels
        - xtick, xticklabels: x-axis tick positions and labels
        - M: list of M values
        - name: name for this subplot
    """
    # Font sizes optimized for A4 paper with 1-inch margins
    tickLabel_FS = 12#10
    axisLabel_FS = 14#12
    legendLabel_FS = 10#9

    # A4 paper dimensions with 1-inch margins
    # A4 page: 8.27 x 11.69 inches
    # With 1-inch margins: 6.27 x 9.69 inches usable
    fig_width = 10.0
    fig_height = 4.0

    # Create figure with 3 subplots - adjust size for A4 page
    # Each subplot shares y-axes
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=False)

    # We'll track y-axis limits to enforce sharing manually
    y1_min, y1_max = float('inf'), float('-inf')
    y2_min, y2_max = float('inf'), float('-inf')

    # Define colors
    color_collab = 'tab:blue'
    color_noncollab = 'tab:orange'

    # First pass: create all plots and collect y-axis limits
    ax2_list = []
    plot_data = []

    for idx, config in enumerate(configs[:3]):  # Only process first 3 configs
        ax1 = axes[idx]

        # Load data from results directory
        data1 = np.loadtxt(os.path.join(RESULTS_DIR, config['file1']), delimiter=',')
        data1_true = np.loadtxt(os.path.join(RESULTS_DIR, config['file1_t']), delimiter=',')
        data2 = np.loadtxt(os.path.join(RESULTS_DIR, config['file2']), delimiter=',')
        data2_true = np.loadtxt(os.path.join(RESULTS_DIR, config['file2_t']), delimiter=',')
        data3 = np.loadtxt(os.path.join(RESULTS_DIR, config['file3']), delimiter=',')
        data3_true = np.loadtxt(os.path.join(RESULTS_DIR, config['file3_t']), delimiter=',')

        # Extract columns for plotting
        collab_axis1 = [data1[:, 0], data1_true[:, 0], data2[:, 0], data2_true[:, 0], data3[:, 0], data3_true[:, 0]]
        noncollab_axis1 = [data1[:, 1], data2[:, 1], data3[:, 1]]

        # Second axis data
        collab_axis2 = []
        collab_axis2.append(process_column_for_km(data1[:, 2], config['M'][0]))
        collab_axis2.append(process_column_for_km(data1_true[:, 2], config['M'][0]))
        collab_axis2.append(process_column_for_km(data2[:, 2], config['M'][1]))
        collab_axis2.append(process_column_for_km(data2_true[:, 2], config['M'][1]))
        collab_axis2.append(process_column_for_km(data3[:, 2], config['M'][2]))
        collab_axis2.append(process_column_for_km(data3_true[:, 2], config['M'][2]))

        noncollab_axis2 = []
        noncollab_axis2.append(process_column_for_km(data1[:, 3], config['M'][0]))
        noncollab_axis2.append(process_column_for_km(data2[:, 3], config['M'][1]))
        noncollab_axis2.append(process_column_for_km(data3[:, 3], config['M'][2]))

        # Store plot data for second pass
        plot_data.append({
            'collab_axis1': collab_axis1,
            'noncollab_axis1': noncollab_axis1,
            'collab_axis2': collab_axis2,
            'noncollab_axis2': noncollab_axis2
        })

        # Update y-axis limits
        for data_list in collab_axis1 + noncollab_axis1:
            y1_min = min(y1_min, np.min(data_list))
            y1_max = max(y1_max, np.max(data_list))

        for val in collab_axis2 + noncollab_axis2:
            y2_min = min(y2_min, val)
            y2_max = max(y2_max, val)

    # Add some padding to y-axis limits
    y1_range = y1_max - y1_min
    y1_min -= 0.05 * y1_range
    y1_max += 0.05 * y1_range

    y2_range = y2_max - y2_min
    y2_min -= 0.05 * y2_range
    y2_max += 0.05 * y2_range

    # Second pass: create actual plots with shared y-axis limits
    for idx, config in enumerate(configs[:3]):  # Only process first 3 configs
        ax1 = axes[idx]
        data = plot_data[idx]

        # Define positions for box plots - adjusted for better spacing
        # Spread out boxes more to reduce crowding around each tick
        positions_collab = [1.5, 1.75, 3.5, 3.75, 5.5, 5.75]
        positions_noncollab = [2.25, 4.25, 6.25]  # Only 3 positions now

        # Plot boxplots on first y-axis
        all_data = [data['collab_axis1'][0], data['collab_axis1'][1], data['collab_axis1'][2],
                    data['collab_axis1'][3], data['collab_axis1'][4], data['collab_axis1'][5],
                    data['noncollab_axis1'][0], data['noncollab_axis1'][1], data['noncollab_axis1'][2]]
        all_positions = positions_collab + positions_noncollab

        bp1 = ax1.boxplot(all_data, positions=all_positions, widths=0.2, patch_artist=True,
                          showfliers=False, whis=(0, 100))

        # Style boxes
        for i in range(9):  # Now only 9 boxes total: 6 collab + 3 noncollab
            if i < 6:  # Collab boxes
                color = color_collab
                edge_color = 'darkblue'
                alpha = 0.7 if i % 2 == 0 else 0.4
            else:  # Noncollab boxes (indices 6, 7, 8)
                color = color_noncollab
                edge_color = 'darkorange'
                alpha = 0.7  # All noncollab boxes have same alpha

            bp1['boxes'][i].set_facecolor(color)
            bp1['boxes'][i].set_alpha(alpha)
            bp1['medians'][i].set_color(edge_color)
            bp1['medians'][i].set_linewidth(2)
            bp1['whiskers'][i*2].set_color(color)
            bp1['whiskers'][i*2+1].set_color(color)
            bp1['caps'][i*2].set_color(color)
            bp1['caps'][i*2+1].set_color(color)

        # Set x-axis labels
        ax1.set_xticks(config['xtick'])
        ax1.set_xticklabels(config['xticklabels'])
        ax1.tick_params(axis='x', labelcolor='black', labelsize=tickLabel_FS)
        ax1.set_xlabel(config['xlabel'], fontsize=axisLabel_FS)

        # Set y-axis properties for first axis
        ax1.set_ylim(y1_min, y1_max)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=tickLabel_FS)

        # Only show y-label on leftmost subplot
        if idx == 0:
            ax1.set_ylabel(config['ylabel1'], color='black', fontsize=axisLabel_FS)
        else:
            ax1.set_yticklabels([])  # Hide y-tick labels for non-leftmost plots

        # Create second y-axis
        ax2 = ax1.twinx()
        ax2_list.append(ax2)

        # Plot scatter points on second y-axis
        ax2.plot([positions_collab[0], positions_collab[2], positions_collab[4]],
                 [data['collab_axis2'][0], data['collab_axis2'][2], data['collab_axis2'][4]],
                 'o', color=color_collab, markersize=5, markeredgewidth=1,
                 markeredgecolor='black', markerfacecolor=color_collab, alpha=0.8) #darkblue

        ax2.plot([positions_collab[1], positions_collab[3], positions_collab[5]],
                 [data['collab_axis2'][1], data['collab_axis2'][3], data['collab_axis2'][5]],
                 'o', color=color_collab, markersize=5, markeredgewidth=1,
                 markeredgecolor='black', markerfacecolor=color_collab, alpha=0.4) #darkblue

        ax2.plot([positions_noncollab[0], positions_noncollab[1], positions_noncollab[2]],
                 [data['noncollab_axis2'][0], data['noncollab_axis2'][1], data['noncollab_axis2'][2]],
                 'o', color=color_noncollab, markersize=5, markeredgewidth=1,
                 markeredgecolor='black', markerfacecolor=color_noncollab, alpha=0.8) #darkorange

        # Set y-axis properties for second axis
        ax2.set_ylim(y2_min, y2_max)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=tickLabel_FS)

        # Only show y-label on rightmost subplot
        if idx == 2:  # Changed from 3 to 2 for 3 subplots
            ax2.set_ylabel(config['ylabel2'], color='black', fontsize=axisLabel_FS)
        else:
            ax2.set_yticklabels([])  # Hide y-tick labels for non-rightmost plots

        # Add grid
        ax1.grid(True, alpha=0.3)

        # Add subplot title (a), (b), (c)
        subplot_labels = [r'$(a)$', r'$(b)$', r'$(c)$']
        ax1.set_title(subplot_labels[idx], fontsize=axisLabel_FS, pad=10)

    # Create custom legend (shared across all subplots)
    legend_elements = [
        Patch(facecolor=color_collab, alpha=0.7, edgecolor='darkblue', linewidth=1.5, label=r'Collab. w/ $r_{\mathrm{est.}}$'),
        Patch(facecolor=color_collab, alpha=0.4, edgecolor='darkblue', linewidth=1.5, label=r'Collab. w/ $r_{\mathrm{true}}$'),
        Patch(facecolor=color_noncollab, alpha=0.7, edgecolor='darkorange', linewidth=1.5, label=r'Non-collab.'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_noncollab,
               markeredgecolor='black', markeredgewidth=1, markersize=4, #darkorange
               alpha=0.8, label=r'Non-collab. ($\hat{\mu}$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_collab,
               markeredgecolor='black', markeredgewidth=1, markersize=4, #darkblue
               alpha=0.8, label=r'Collab. w/ $r_{\mathrm{est.}}$ ($\hat{\mu}$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_collab,
               markeredgecolor='black', markeredgewidth=1, markersize=4, #darkblue
               alpha=0.4, label=r'Collab. w/ $r_{\mathrm{true}}$ ($\hat{\mu}$)'),
    ]

    # Place legend below the plots, centered, in 2 rows
    # This is optimal for horizontal 3-subplot layout
    fig.legend(handles=legend_elements,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=3,
              fontsize=legendLabel_FS,
              frameon=True,
              fancybox=True,
              shadow=True)

    plt.tight_layout()
    # Adjust layout to make room for legend at bottom
    plt.subplots_adjust(bottom=0.25)

    # Save the figure
    filename = 'Seq_MC_sim_combined_cases.png'
    fig.savefig(os.path.join(FIGURES_DIR, filename), format='png', dpi=400, bbox_inches='tight')

    return fig

if __name__ == '__main__':
    plot_cumu_regrets()
    plot_subset_seq_mc()

    # Filename Format: r_K_l_r-true_r-exp_zeta_M
    configs = []

    # Case 1: rank
    configs.append({
        'file1': "FRPy_result_tab_50_50_5_4_60_20.csv",
        'file1_t': "FRPy_result_tab_50_50_5_5_60_20.csv",
        'file2': "FRPy_result_tab_50_50_10_9_60_20.csv",
        'file2_t': "FRPy_result_tab_50_50_10_10_60_20.csv",
        'file3': "FRPy_result_tab_50_50_15_11_60_20.csv",
        'file3_t': "FRPy_result_tab_50_50_15_15_60_20.csv",
        'xlabel': "$r$",
        'ylabel1': "$\sum_{t=1}^{M} \mathrm{regret}^{(t)}_{k}$", #Terminal Cumulative Regret
        'ylabel2': "$\hat{\mu}$",
        'xtick': [2, 4, 6],
        'xticklabels': ["$5$", "$10$", "$15$"],
        'M': [20, 20, 20],
        'name': 'rank'
    })

    # Case 2: K
    configs.append({
        'file1': "FRPy_result_tab_30_50_10_8_60_20.csv",
        'file1_t': "FRPy_result_tab_30_50_10_10_60_20.csv",
        'file2': "FRPy_result_tab_50_50_10_9_60_20.csv",
        'file2_t': "FRPy_result_tab_50_50_10_10_60_20.csv",
        'file3': "FRPy_result_tab_100_50_10_10_60_20.csv",
        'file3_t': "FRPy_result_tab_100_50_10_10_60_20.csv",
        'xlabel': "$K$",
        'ylabel1': "$\sum_{t=1}^{M} \mathrm{regret}^{(t)}_{k}$", #Cumulative Regret
        'ylabel2': "$\hat{\mu}$",
        'xtick': [2, 4, 6],
        'xticklabels': ["$30$", "$50$", "$100$"],
        'M': [20, 20, 20],
        'name': 'K'
    })

    # Case 3: missing
    configs.append({
        'file1': "FRPy_result_tab_50_50_15_12_50_15.csv",
        'file1_t': "FRPy_result_tab_50_50_15_15_50_15.csv",
        'file2': "FRPy_result_tab_50_50_15_11_60_15.csv",
        'file2_t': "FRPy_result_tab_50_50_15_15_60_15.csv",
        'file3': "FRPy_result_tab_50_50_15_7_80_15.csv",
        'file3_t': "FRPy_result_tab_50_50_15_15_80_15.csv",
        'xlabel': "$\zeta$ ($\%$)",
        'ylabel1': "$\sum_{t=1}^{M} \mathrm{regret}^{(t)}_{k}$", #Cumulative Regret
        'ylabel2': "$\hat{\mu}$",
        'xtick': [2, 4, 6],
        'xticklabels': ["$50$", "$60$", "$80$"],
        'M': [15, 15, 15],
        'name': 'missing'
    })

    fig_combined = plot_combined_sim_cases(configs)