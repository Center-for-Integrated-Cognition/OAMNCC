import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

# --- Constants ---
# Scenario Geography and Setup
TARGET_NAMES = ["Target Alpha", "Target Bravo", "Target Charlie", "Target Delta"]
NAVAL_SHIP_START_COORDS = (45.0, 75.0)
SHIPPING_LANE_X_MIN = 30.0
SHIPPING_LANE_X_MAX = 40.0
SHIPPING_LANE_Y_MIN = 0.0
SHIPPING_LANE_Y_MAX = 100.0
MIN_INTER_TARGET_DISTANCE = 20.0 # nm
PIRATE_START_X = 30.0 # The width of the shipping lane (position of a merchant in it) determines possible "warning time" before boarding attempts. (ranges from being completely surprised to having some warning and being closer to naval ship.)

# Time and Speed Conversion
CLOSING_SPEED_KNOTS = 160.0 # nm per hour -- reflecting Sikorsky MH-60 jayhawk but like, also weather.
EXPECTED_CREW_CLOSING_SPEED_KNOTS = 160.0 #We imagine the impact of a crew that is less efficient as a discrepancy between expected/simulated closing speed when deciding how to respond and real closing speed.
PIRATE_CLOSING_SPEED_KNOTS = 15.0 
MINUTES_PER_HOUR = 60.0
MINUTES_TO_TRAVEL_ONE_NM = MINUTES_PER_HOUR / CLOSING_SPEED_KNOTS # 60 / 30 = 2 minutes/nm
PIRATE_MINUTES_TO_TRAVEL_ONE_NM = MINUTES_PER_HOUR / PIRATE_CLOSING_SPEED_KNOTS
EXPECTED_MINUTES_TO_TRAVEL_ONE_NM = MINUTES_PER_HOUR / CLOSING_SPEED_KNOTS

# Pirate Engagement Parameters
PIRATE_GIVE_UP_TIME_MINUTES = 30.0
NONINTERVENTION_PIRATE_SUCCESS_PROBABILITY = 0.05
# Derived from P(Failure)^30 = 0.05 -> P(Failure) = 0.05^(1/30)
PROB_PIRATE_SUCCESS_PER_MINUTE = 1.0 - (NONINTERVENTION_PIRATE_SUCCESS_PROBABILITY ** (1.0 / PIRATE_GIVE_UP_TIME_MINUTES))

# Global list to collect all generated ransom values for the new plot
ALL_GENERATED_RANSOM_VALUES = []

# --- Helper Functions ---
def calculate_distance(coords1, coords2):
    return math.sqrt((coords2[0] - coords1[0])**2 + (coords2[1] - coords1[1])**2)

def generate_and_normalize_ransom_values(target_names_list, total_sum=500.0, min_val=50, max_val=200):
    num_targets = len(target_names_list)
    if num_targets == 0: return {}

    raw_values = [random.uniform(min_val, max_val) for _ in range(num_targets)]
    current_sum = sum(raw_values)
    
    scale_factor = total_sum / current_sum if current_sum > 0 else 0
    normalized_values = [val * scale_factor for val in raw_values]
   
    global ALL_GENERATED_RANSOM_VALUES
    ALL_GENERATED_RANSOM_VALUES.extend(normalized_values)

    return {name: round(val, 2) for name, val in zip(target_names_list, normalized_values)}

def generate_merchant_locations_in_lane(num_targets, min_dist):
    locations = {}
    target_coords_list = []
    
    for i in range(num_targets):
        attempts = 0
        while attempts < 1000: # Safety break
            x = random.uniform(SHIPPING_LANE_X_MIN, SHIPPING_LANE_X_MAX)
            y = random.uniform(SHIPPING_LANE_Y_MIN, SHIPPING_LANE_Y_MAX)
            new_coords = (x, y)
            
            is_too_close = False
            for existing_coords in target_coords_list:
                if calculate_distance(new_coords, existing_coords) < min_dist:
                    is_too_close = True
                    break
            if not is_too_close:
                break
            attempts += 1
        
        target_coords_list.append(new_coords)
        locations[TARGET_NAMES[i]] = new_coords
        
    return locations


def simulate_engagement_outcome(merchant_coords, merchant_ransom_value, naval_ship_intervention_coords):
    """
    Simulates the full engagement for one merchant and returns its outcome (ransom value or 0).
    """
    # Phase 1: Approach Times
    pirate_start_coords = (PIRATE_START_X, merchant_coords[1])
    pirate_distance_to_merchant = calculate_distance(pirate_start_coords, merchant_coords)
    time_pirate_reaches_merchant = pirate_distance_to_merchant * PIRATE_MINUTES_TO_TRAVEL_ONE_NM

    naval_ship_distance_to_merchant = calculate_distance(NAVAL_SHIP_START_COORDS, naval_ship_intervention_coords)
    time_naval_ship_arrives = naval_ship_distance_to_merchant * MINUTES_TO_TRAVEL_ONE_NM

    # Case 1: Naval ship arrives before the pirate even reaches the merchant.
    if time_naval_ship_arrives < time_pirate_reaches_merchant:
        return merchant_ransom_value # Win: Pirate is scared off.

    # Phase 2: Boarding Attempt
    # The pirate attacks from the moment it arrives until the naval ship arrives OR it gives up.
    duration_of_boarding_attempt = min(
        PIRATE_GIVE_UP_TIME_MINUTES,
        time_naval_ship_arrives - time_pirate_reaches_merchant
    )

    # Simulate the per-minute coin flips for the duration of the boarding attempt.
    is_boarded = False
    for minute in range(int(math.ceil(duration_of_boarding_attempt))):
        if random.random() < PROB_PIRATE_SUCCESS_PER_MINUTE:
            is_boarded = True
            break
    
    # Phase 3: Determine Outcome
    if is_boarded:
        return 0.0 # Loss: Merchant was boarded before naval ship arrived or pirate gave up.
    else:
        # If not boarded, it's a win. This can be because:
        # 1. The naval ship arrived and interrupted the (so far unsuccessful) boarding.
        # 2. The pirate's 30-minute timer ran out before the naval ship arrived.
        return merchant_ransom_value

def calculate_s3_estimated_marginal_gain(merchant_coords, naval_ship_start_coords):
    """
    Calculates the estimated marginal gain for Strategy 3.
    This is the agent's "mental simulation" to inform its choice.
    """
    # Agent's estimated time for pirate to reach merchant
    pirate_start_coords = (PIRATE_START_X, merchant_coords[1])
    pirate_dist = calculate_distance(pirate_start_coords, merchant_coords)
    est_t_pirate_arrival = pirate_dist * PIRATE_MINUTES_TO_TRAVEL_ONE_NM

    # --- P(Success WITH Intervention) ---
    # Agent's estimated time for naval ship to arrive
    naval_dist = calculate_distance(naval_ship_start_coords, merchant_coords)
    est_t_hub_arrival = naval_dist * EXPECTED_MINUTES_TO_TRAVEL_ONE_NM

    p_success_with_intervention = 0.0
    if est_t_hub_arrival < est_t_pirate_arrival:
        p_success_with_intervention = 1.0
    else:
        # If hub arrives later, success is the chance the pirate fails every minute
        # for the duration the merchant is vulnerable.
        duration_at_risk = est_t_hub_arrival - est_t_pirate_arrival
        num_minutes_at_risk = min(PIRATE_GIVE_UP_TIME_MINUTES, duration_at_risk)
        
        prob_pirate_fails_per_minute = 1.0 - PROB_PIRATE_SUCCESS_PER_MINUTE
        p_success_with_intervention = prob_pirate_fails_per_minute ** num_minutes_at_risk

    # --- P(Success WITHOUT Intervention) ---
    # If not helped, the merchant is at risk for the full 30-minute give-up time.
    prob_pirate_fails_per_minute = 1.0 - PROB_PIRATE_SUCCESS_PER_MINUTE
    p_success_without_intervention = prob_pirate_fails_per_minute ** PIRATE_GIVE_UP_TIME_MINUTES

    marginal_gain = p_success_with_intervention - p_success_without_intervention
    return max(0.0, marginal_gain)


def run_strategies(trial_targets_data):
    """Determines the target choice for each of the three strategies."""
    # S1: Closest Target
    s1_choice = min(trial_targets_data, key=lambda t: t['distance_from_hub'])

    # S2: Highest Value Target
    s2_choice = max(trial_targets_data, key=lambda t: t['ransom_value'])

    # S3: Highest Marginal Utility (Ransom Value * Estimated Marginal Gain)
    s3_choice = max(trial_targets_data, key=lambda t: t['ransom_value'] * t['s3_marginal_gain_prob'])
    
    return s1_choice, s2_choice, s3_choice

def execute_scenario(scenario_params):
    desc = scenario_params["description"]
    num_trials = scenario_params["num_trials"]

    print(f"\n--- EXECUTING SCENARIO: {desc} ---")
    print(f"Naval Ship Start: {NAVAL_SHIP_START_COORDS}, Closing Speed: {CLOSING_SPEED_KNOTS} knots")
    print(f"Pirate Attack: Starts at x={PIRATE_START_X}, gives up after {PIRATE_GIVE_UP_TIME_MINUTES} mins")
    print(f"Running {num_trials} trials...")

    s1_raw_outcomes = []
    s2_raw_outcomes = []
    s3_raw_outcomes = []
    no_intervention_outcomes = []

    for _ in range(num_trials):
        merchant_locations = generate_merchant_locations_in_lane(len(TARGET_NAMES), MIN_INTER_TARGET_DISTANCE)
        ransom_values = generate_and_normalize_ransom_values(TARGET_NAMES)

        trial_targets_data_for_strategies = []
        for name in TARGET_NAMES:
            coords = merchant_locations[name]
            trial_targets_data_for_strategies.append({
                "name": name,
                "coords": coords,
                "ransom_value": ransom_values[name],
                "distance_from_hub": calculate_distance(NAVAL_SHIP_START_COORDS, coords),
                "s3_marginal_gain_prob": calculate_s3_estimated_marginal_gain(coords, NAVAL_SHIP_START_COORDS)
            })

        s1_choice, s2_choice, s3_choice = run_strategies(trial_targets_data_for_strategies)

        trial_baseline_outcome = 0
        for target in trial_targets_data_for_strategies:
            trial_baseline_outcome += NONINTERVENTION_PIRATE_SUCCESS_PROBABILITY * target['ransom_value']#simulate_engagement_outcome(
#                target['coords'], target['ransom_value'], (float('inf'), float('inf'))
#            )
        no_intervention_outcomes.append(trial_baseline_outcome)

        for strat_idx, chosen_target in enumerate([s1_choice, s2_choice, s3_choice]):
            current_trial_total_outcome = 0
            current_trial_total_outcome += simulate_engagement_outcome(
                chosen_target['coords'], chosen_target['ransom_value'], chosen_target['coords']
            )
            for target in trial_targets_data_for_strategies:
                if target['name'] != chosen_target['name']:
                    current_trial_total_outcome += NONINTERVENTION_PIRATE_SUCCESS_PROBABILITY * target['ransom_value'] #simulate_engagement_outcome(
#                        target['coords'], target['ransom_value'], (float('inf'), float('inf'))
#                    )
            
            if strat_idx == 0: s1_raw_outcomes.append(current_trial_total_outcome)
            elif strat_idx == 1: s2_raw_outcomes.append(current_trial_total_outcome)
            elif strat_idx == 2: s3_raw_outcomes.append(current_trial_total_outcome)

    return {
        "description": desc,
        "s1_raw_outcomes": s1_raw_outcomes,
        "s2_raw_outcomes": s2_raw_outcomes,
        "s3_raw_outcomes": s3_raw_outcomes,
        "no_intervention_outcomes": no_intervention_outcomes,
        "num_trials": num_trials
    }

def plot_scenario_outcomes(desc, num_trials, s1_data, s2_data, s3_data, 
                           plot_title_suffix="", y_axis_label="", filename_prefix=""):
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'font.serif': 'Times New Roman'})
    if num_trials == 0:
        print(f"Scenario '{desc}': num_trials is 0 for {filename_prefix} plot, skipping.")
        return

    data_to_plot = []
    labels = []
    if s1_data and len(s1_data) > 1: data_to_plot.append(s1_data); labels.append('Closest')
    else: s1_data = None
    if s2_data and len(s2_data) > 1: data_to_plot.append(s2_data); labels.append('Highest\nValue')
    else: s2_data = None
    if s3_data and len(s3_data) > 1: data_to_plot.append(s3_data); labels.append('Marginal\nGain')
    else: s3_data = None
       
    if not data_to_plot:
        print(f"Scenario '{desc}': Not enough valid data for {filename_prefix} plot or KS tests.")
        return

    fig, ax = plt.subplots(figsize=(3.27, 2.5))
    plot_positions = np.array(range(len(data_to_plot)))

    #bp = ax.boxplot(data_to_plot, labels=labels, showmeans=True, meanline=False,
    #                patch_artist=True, widths=0.15, showfliers=True,
    #                positions=plot_positions - 0.12)
    vp = ax.violinplot(data_to_plot, showmeans=True, showmedians=False, showextrema=False,
                       positions=plot_positions)

    violin_colors = ['skyblue', 'lightgreen', 'lightcoral']
    for i, pc in enumerate(vp['bodies']): pc.set_facecolor(violin_colors[i % len(violin_colors)]); pc.set_edgecolor('grey'); pc.set_alpha(0.7)
    if 'cmeans' in vp: vp['cmeans'].set_edgecolor('darkviolet'); vp['cmeans'].set_linewidth(2)
    box_colors = ['#E0E0E0', '#D0D0D0', '#C0C0C0']
    #for i, patch in enumerate(bp['boxes']): patch.set_facecolor(box_colors[i % len(box_colors)]); patch.set_alpha(0.9)

    #if bp['means']:
    #    for i, mean_line in enumerate(bp['means']):
    #        if i < len(data_to_plot) and data_to_plot[i]:
    #            mean_val = mean_line.get_ydata()[0]
    #            ax.text(plot_positions[i] - 0.12, mean_val + 5, f'Mean: {mean_val:.1f}', 
    #                    va='bottom', ha='center', color='black', fontweight='bold', fontsize=9,
    #                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

    #ax.set_title(f'Distribution of {y_axis_label} per Trial')
    ax.set_ylabel(y_axis_label)
    #ax.set_xlabel('Selection Strategy')
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(labels)
    plt.xticks(ha="center")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if "Ransom Avoided" in y_axis_label :
        all_relative_data = [item for sublist in data_to_plot if sublist for item in sublist]
        if all_relative_data:
            abs_max = max(abs(min(all_relative_data)), abs(max(all_relative_data)))
            padding = abs_max * 0.1 if abs_max > 0 else 10 
            #ax.set_ylim(-abs_max - padding, abs_max + padding) #min(all_relative_data)
            ax.set_ylim(min(0,min(all_relative_data)), abs_max + padding)
    else:
        ax.set_ylim(-5, 560)
       
    y_max_data_for_annotation = 0
    all_flat_data = [item for sublist in data_to_plot if sublist for item in sublist]
    if all_flat_data: y_max_data_for_annotation = np.percentile(all_flat_data, 98)
    else: y_max_data_for_annotation = ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > ax.get_ylim()[0] else 50

    current_y_lim_top = ax.get_ylim()[1]
    current_y_lim_bottom = ax.get_ylim()[0]
    annotation_y_level = y_max_data_for_annotation + (current_y_lim_top - y_max_data_for_annotation) * 0.1 
    if annotation_y_level >= current_y_lim_top * 0.95 : 
        annotation_y_level = current_y_lim_top * 0.85
    if annotation_y_level <= current_y_lim_bottom + (current_y_lim_top - current_y_lim_bottom) *0.1: 
        annotation_y_level = y_max_data_for_annotation + 10 
    
    plot_data_range = current_y_lim_top - y_max_data_for_annotation
    line_height_offset = plot_data_range * 0.05 if plot_data_range > 0 else 2
    text_v_offset = plot_data_range * 0.015 if plot_data_range > 0 else 1

    def format_p_value(p_val):
        stars = ""
        if p_val < 0.001: stars = "***"
        elif p_val < 0.01: stars = "**"
        elif p_val < 0.05: stars = "*"
        p_str = f"p < 0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        return f"{p_str}"#{stars}"

    s1_idx, s2_idx, s3_idx = -1, -1, -1
    if 'Closest' in labels: s1_idx = labels.index('Closest')
    if 'Highest\nValue' in labels: s2_idx = labels.index('Highest\nValue')
    if 'Marginal\nGain' in labels: s3_idx = labels.index('Marginal\nGain')

    y_pos_s1s2 = annotation_y_level-line_height_offset
    y_pos_s2s3 = annotation_y_level
    y_pos_s1s3 = annotation_y_level

    if s1_idx != -1 and s2_idx != -1:
        s1_actual_data = data_to_plot[s1_idx]
        s2_actual_data = data_to_plot[s2_idx]
        if len(s1_actual_data) > 1 and len(s2_actual_data) > 1:
            _, p_val = ks_2samp(s1_actual_data, s2_actual_data)
            x1, x2 = plot_positions[s1_idx], plot_positions[s2_idx]
            ax.plot([x1, x1, x2, x2], [y_pos_s1s2 - line_height_offset, y_pos_s1s2, y_pos_s1s2, y_pos_s1s2 - line_height_offset], lw=1.2, c='black')
            ax.text((x1 + x2) * 0.5, y_pos_s1s2 + text_v_offset, format_p_value(p_val), ha='center', va='bottom', fontsize=9, color='black')
            if s3_idx != -1: 
                y_pos_s2s3 = y_pos_s1s2 + 1*line_height_offset * 2.5 + text_v_offset*2 
                y_pos_s1s3 = y_pos_s2s3 + 3*line_height_offset * 2.5 + text_v_offset*2

    if s2_idx != -1 and s3_idx != -1:
        s2_actual_data = data_to_plot[s2_idx]
        s3_actual_data = data_to_plot[s3_idx]
        if len(s2_actual_data) > 1 and len(s3_actual_data) > 1:
            _, p_val = ks_2samp(s2_actual_data, s3_actual_data)
            x1, x2 = plot_positions[s2_idx], plot_positions[s3_idx]
            ax.plot([x1, x1, x2, x2], [y_pos_s2s3 - line_height_offset, y_pos_s2s3, y_pos_s2s3, y_pos_s2s3 - line_height_offset], lw=1.2, c='black')
            ax.text((x1 + x2) * 0.5, y_pos_s2s3 + text_v_offset, format_p_value(p_val), ha='center', va='bottom', fontsize=9, color='black')

    if s1_idx != -1 and s3_idx != -1:
        s1_actual_data = data_to_plot[s1_idx]
        s3_actual_data = data_to_plot[s3_idx]
        if len(s1_actual_data) > 1 and len(s3_actual_data) > 1:
            _, p_val = ks_2samp(s1_actual_data, s3_actual_data)
            x1, x2 = plot_positions[s1_idx], plot_positions[s3_idx]
            ax.plot([x1, x1, x2, x2], [y_pos_s1s3 - line_height_offset, y_pos_s1s3, y_pos_s1s3, y_pos_s1s3 - line_height_offset], lw=1.2, c='black')
            ax.text((x1 + x2) * 0.5, y_pos_s1s3 + text_v_offset, format_p_value(p_val), ha='center', va='bottom', fontsize=9, color='black')
   
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
   
    filename_desc = "".join(c if c.isalnum() else '_' for c in desc) 
    filename = f"scenario_outcomes_{filename_prefix}{filename_desc[:30]}.pdf"
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f" Plot saved as {filename}")
    except Exception as e: print(f" Could not save plot: {e}")
    plt.close(fig)

# --- Main Execution Function and Call ---
def main():
    # random.seed(42) # For reproducibility
       
    # This script is designed around a single, specific scenario as per the comments.
    # The framework can be extended to a list of scenarios if needed.
    scenario_to_run = {
        "description": "Naval Intervention in Shipping Lane",
        "num_trials": 1000,
        # All other parameters are defined as constants in this simpler script
    }
   
    print(f"\n{'='*25} PROCESSING SCENARIO {'='*25}")
    results = execute_scenario(scenario_to_run)
    print(f"{'='*30} SCENARIO COMPLETE {'='*30}\n")

    print("\n--- Generating Plots for Scenario Outcomes ---")
    if results["num_trials"] > 0 :
        # Calculate and plot relative outcomes (Value Added)
        baseline_outcomes = results["no_intervention_outcomes"]
        s1_relative = [r - b for r, b in zip(results["s1_raw_outcomes"], baseline_outcomes)]
        s2_relative = [r - b for r, b in zip(results["s2_raw_outcomes"], baseline_outcomes)]
        s3_relative = [r - b for r, b in zip(results["s3_raw_outcomes"], baseline_outcomes)]
       
        plot_scenario_outcomes(results["description"], results["num_trials"],
                               s1_relative, s2_relative, s3_relative,
                               plot_title_suffix=" (Relative to No Intervention)",
                               y_axis_label='Ransom Avoided',
                               filename_prefix="relative_")
    else:
        print(f"Skipping plots for '{results['description']}' as no trials were run.")
   
    # Plot overall distribution of generated Ransom Values
    #plot_overall_ransom_distribution(ALL_GENERATED_RANSOM_VALUES)

if __name__ == "__main__":
    main()
