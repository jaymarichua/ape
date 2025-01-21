import pandas as pd
import scipy.stats as stats
import numpy as np

# Data (from the 50 persona responses for the StarCraft II context)
data = {
    'PersonaID': [
        'SC2_Novice_1_RunA', 'SC2_Novice_2_RunB', 'SC2_Novice_3_RunC', 'SC2_Novice_4_RunD', 'SC2_Novice_5_RunE',
        'SC2_Novice_6_RunF', 'SC2_Novice_7_RunG', 'SC2_Novice_8_RunH', 'SC2_Novice_9_RunI', 'SC2_Novice_10_RunJ',
        'SC2_Expert_1_RunA', 'SC2_Expert_2_RunB', 'SC2_Expert_3_RunC', 'SC2_Expert_4_RunD', 'SC2_Expert_5_RunE',
        'SC2_Expert_6_RunF', 'SC2_Expert_7_RunG', 'SC2_Expert_8_RunH', 'SC2_Expert_9_RunI', 'SC2_Expert_10_RunJ',
        'SC2_Novice_11_RunK', 'SC2_Novice_12_RunL', 'SC2_Novice_13_RunM', 'SC2_Novice_14_RunN', 'SC2_Novice_15_RunO',
        'SC2_Novice_16_RunP', 'SC2_Novice_17_RunQ', 'SC2_Novice_18_RunR', 'SC2_Novice_19_RunS', 'SC2_Novice_20_RunT',
        'SC2_Expert_11_RunK', 'SC2_Expert_12_RunL', 'SC2_Expert_13_RunM', 'SC2_Expert_14_RunN', 'SC2_Expert_15_RunO',
        'SC2_Expert_16_RunP', 'SC2_Expert_17_RunQ', 'SC2_Expert_18_RunR', 'SC2_Expert_19_RunS', 'SC2_Expert_20_RunT',
        'SC2_Novice_21_RunU', 'SC2_Novice_22_RunV', 'SC2_Novice_23_RunW', 'SC2_Novice_24_RunX', 'SC2_Novice_25_RunY',
        'SC2_Novice_26_RunZ', 'SC2_Novice_27_RunAA', 'SC2_Novice_28_RunBB', 'SC2_Novice_29_RunCC', 'SC2_Novice_30_RunDD',
        'SC2_Expert_21_RunU', 'SC2_Expert_22_RunV', 'SC2_Expert_23_RunW', 'SC2_Expert_24_RunX', 'SC2_Expert_25_RunY',
        'SC2_Expert_26_RunZ', 'SC2_Expert_27_RunAA', 'SC2_Expert_28_RunBB', 'SC2_Expert_29_RunCC', 'SC2_Expert_30_RunDD',
        'SC2_Novice_31_RunEE', 'SC2_Novice_32_RunFF', 'SC2_Novice_33_RunGG', 'SC2_Novice_34_RunHH', 'SC2_Novice_35_RunII',
        'SC2_Novice_36_RunJJ', 'SC2_Novice_37_RunKK', 'SC2_Novice_38_RunLL', 'SC2_Novice_39_RunMM', 'SC2_Novice_40_RunNN',
        'SC2_Expert_31_RunEE', 'SC2_Expert_32_RunFF', 'SC2_Expert_33_RunGG', 'SC2_Expert_34_RunHH', 'SC2_Expert_35_RunII',
        'SC2_Expert_36_RunJJ', 'SC2_Expert_37_RunKK', 'SC2_Expert_38_RunLL', 'SC2_Expert_39_RunMM', 'SC2_Expert_40_RunNN'
    ],
    'Expertise': [
        'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice',
        'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert',
        'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice',
        'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert',
        'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice',
        'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert',
        'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice', 'Novice',
        'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert', 'Expert'
    ],
    'Disclosure': [
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman',
        'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman', 'No', 'Superhuman'
    ],
    'Toxicity': [
        4, 3, 5, 2, 4, 3, 5, 2, 4, 3,
        5, 4, 5, 3, 5, 3, 5, 2, 5, 3,
        3, 2, 4, 3, 4, 2, 5, 3, 4, 2,
        5, 3, 5, 2, 5, 3, 5, 2, 5, 3,
        4, 2, 5, 3, 4, 2, 5, 3, 4, 2,
        5, 3, 5, 2, 5, 3, 5, 2, 5, 3,
        4, 2, 4, 3, 4, 2, 5, 3, 4, 2,
        5, 3, 5, 2, 5, 3, 5, 2, 5, 3
    ],
    'Fairness': [
        2, 3, 1, 4, 2, 3, 1, 3, 2, 4,
        2, 3, 1, 4, 2, 3, 1, 4, 2, 3,
        3, 4, 2, 3, 2, 4, 1, 3, 2, 4,
        2, 3, 1, 4, 2, 3, 1, 4, 2, 3,
        2, 4, 1, 3, 2, 4, 1, 3, 2, 4,
        1, 4, 2, 3, 2, 3, 1, 4, 2, 3,
        3, 4, 2, 4, 3, 4, 1, 4, 2, 4,
        1, 4, 1, 3, 2, 4, 1, 4, 2, 3
    ],
    'Trust': [
        3, 5, 2, 4, 3, 4, 1, 5, 3, 4,
        2, 3, 1, 5, 3, 4, 1, 5, 2, 4,
        3, 5, 2, 4, 3, 5, 2, 4, 3, 5,
        2, 4, 1, 5, 3, 4, 1, 5, 2, 4,
        3, 4, 2, 4, 3, 5, 1, 4, 3, 5,
        2, 4, 1, 5, 3, 4, 1, 5, 2, 4,
        2, 5, 3, 4, 2, 5, 2, 4, 3, 5,
        1, 4, 1, 5, 2, 4, 1, 5, 2, 4
    ]
}

df = pd.DataFrame(data)

# --- Functions for statistical tests and effect sizes ---

def perform_t_test(group1, group2):
    """Performs an independent samples t-test, checks assumptions, and returns relevant statistics."""
    # Check for normality
    _, p_normality_group1 = stats.shapiro(group1)
    _, p_normality_group2 = stats.shapiro(group2)

    # Check for homogeneity of variances
    _, p_levene = stats.levene(group1, group2)

    # Choose the appropriate t-test
    if p_normality_group1 > 0.05 and p_normality_group2 > 0.05 and p_levene > 0.05:
        # Parametric t-test
        t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        test_type = "Standard Independent Samples T-test"
    elif p_normality_group1 > 0.05 and p_normality_group2 > 0.05 and p_levene < 0.05:
        # Welch's t-test (unequal variances)
        t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_type = "Welch's T-test"
    else:
        # Non-parametric Mann-Whitney U test
        t_statistic, p_value = stats.mannwhitneyu(group1, group2)
        test_type = "Mann-Whitney U Test"

    return t_statistic, p_value, test_type, p_normality_group1, p_normality_group2, p_levene

def calculate_cohens_d(group1, group2):
    """Calculates Cohen's d for effect size."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((np.std(group1, ddof=1) ** 2) + (np.std(group2, ddof=1) ** 2)) / 2)
    cohens_d = mean_diff / pooled_std
    return cohens_d

# --- Perform the statistical tests ---

results = []

for expertise in ['Novice', 'Expert']:
    for variable in ['Toxicity', 'Fairness', 'Trust']:
        group_no_disclosure = df[(df['Expertise'] == expertise) & (df['Disclosure'] == 'No')][variable]
        group_superhuman_disclosure = df[(df['Expertise'] == expertise) & (df['Disclosure'] == 'Superhuman')][variable]

        # Perform t-test (or Mann-Whitney U)
        t_statistic, p_value, test_type, p_normality_group1, p_normality_group2, p_levene = perform_t_test(group_no_disclosure, group_superhuman_disclosure)

        # Calculate Cohen's d
        cohens_d = calculate_cohens_d(group_no_disclosure, group_superhuman_disclosure)

        # Store results
        results.append({
            'Expertise': expertise,
            'Variable': variable,
            'Test Type': test_type,
            'T-statistic/U-statistic': t_statistic,
            'P-value': p_value,
            'Cohen\'s d': cohens_d,
            'Mean (No Disclosure)': np.mean(group_no_disclosure),
            'SD (No Disclosure)': np.std(group_no_disclosure, ddof=1),
            'Mean (Superhuman Disclosure)': np.mean(group_superhuman_disclosure),
            'SD (Superhuman Disclosure)': np.std(group_superhuman_disclosure, ddof=1),
            'N (No Disclosure)': len(group_no_disclosure),
            'N (Superhuman Disclosure)': len(group_superhuman_disclosure),
            'Shapiro-Wilk p-value (Group 1)': p_normality_group1,
            'Shapiro-Wilk p-value (Group 2)': p_normality_group2,
            'Levene\'s p-value': p_levene
        })

# --- Create a DataFrame for the results ---

results_df = pd.DataFrame(results)

# --- Display the results ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(results_df)
