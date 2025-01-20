import pandas as pd
import scipy.stats as stats
import numpy as np

# Data (from the 50 persona responses)
data = {
    'PersonaID': [
        'LLM_Novice_1_RunA', 'LLM_Novice_2_RunB', 'LLM_Novice_3_RunC', 'LLM_Novice_4_RunD', 'LLM_Novice_5_RunE',
        'LLM_Novice_6_RunF', 'LLM_Novice_7_RunG', 'LLM_Novice_8_RunH', 'LLM_Novice_9_RunI', 'LLM_Novice_10_RunJ',
        'LLM_Expert_1_RunA', 'LLM_Expert_2_RunB', 'LLM_Expert_3_RunC', 'LLM_Expert_4_RunD', 'LLM_Expert_5_RunE',
        'LLM_Expert_6_RunF', 'LLM_Expert_7_RunG', 'LLM_Expert_8_RunH', 'LLM_Expert_9_RunI', 'LLM_Expert_10_RunJ',
        'LLM_Novice_11_RunK', 'LLM_Novice_12_RunL', 'LLM_Novice_13_RunM', 'LLM_Novice_14_RunN', 'LLM_Novice_15_RunO',
        'LLM_Novice_16_RunP', 'LLM_Novice_17_RunQ', 'LLM_Novice_18_RunR', 'LLM_Novice_19_RunS', 'LLM_Novice_20_RunT',
        'LLM_Expert_11_RunK', 'LLM_Expert_12_RunL', 'LLM_Expert_13_RunM', 'LLM_Expert_14_RunN', 'LLM_Expert_15_RunO',
        'LLM_Expert_16_RunP', 'LLM_Expert_17_RunQ', 'LLM_Expert_18_RunR', 'LLM_Expert_19_RunS', 'LLM_Expert_20_RunT',
        'LLM_Novice_21_RunU', 'LLM_Novice_22_RunV', 'LLM_Novice_23_RunW', 'LLM_Novice_24_RunX', 'LLM_Novice_25_RunY',
        'LLM_Novice_26_RunZ', 'LLM_Novice_27_RunAA', 'LLM_Novice_28_RunBB', 'LLM_Novice_29_RunCC', 'LLM_Novice_30_RunDD',
        'LLM_Expert_21_RunU', 'LLM_Expert_22_RunV', 'LLM_Expert_23_RunW', 'LLM_Expert_24_RunX', 'LLM_Expert_25_RunY',
        'LLM_Expert_26_RunZ', 'LLM_Expert_27_RunAA', 'LLM_Expert_28_RunBB', 'LLM_Expert_29_RunCC', 'LLM_Expert_30_RunDD',
        'LLM_Novice_31_RunEE', 'LLM_Novice_32_RunFF', 'LLM_Novice_33_RunGG', 'LLM_Novice_34_RunHH', 'LLM_Novice_35_RunII',
        'LLM_Novice_36_RunJJ', 'LLM_Novice_37_RunKK', 'LLM_Novice_38_RunLL', 'LLM_Novice_39_RunMM', 'LLM_Novice_40_RunNN',
        'LLM_Expert_31_RunEE', 'LLM_Expert_32_RunFF', 'LLM_Expert_33_RunGG', 'LLM_Expert_34_RunHH', 'LLM_Expert_35_RunII',
        'LLM_Expert_36_RunJJ', 'LLM_Expert_37_RunKK', 'LLM_Expert_38_RunLL', 'LLM_Expert_39_RunMM', 'LLM_Expert_40_RunNN'
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
        2, 1, 3, 2, 2, 1, 4, 3, 2, 1,
        4, 5, 3, 4, 4, 5, 3, 4, 5, 4,
        2, 1, 3, 2, 2, 1, 3, 2, 2, 1,
        4, 5, 3, 4, 4, 5, 3, 4, 4, 3,
        2, 1, 3, 2, 2, 1, 3, 2, 2, 1,
        4, 5, 3, 4, 4, 4, 3, 4, 4, 3,
        2, 1, 3, 1, 2, 1, 3, 2, 2, 1,
        4, 5, 3, 4, 4, 5, 3, 4, 4, 3
    ],
    'Fairness': [
        3, 4, 2, 3, 3, 4, 2, 3, 4, 4,
        2, 1, 2, 2, 3, 1, 2, 1, 2, 3,
        3, 4, 3, 4, 3, 4, 2, 3, 4, 4,
        2, 1, 2, 1, 3, 1, 2, 1, 2, 3,
        3, 4, 3, 4, 3, 4, 2, 3, 4, 4,
        2, 1, 2, 1, 3, 2, 2, 1, 2, 3,
        4, 4, 3, 4, 4, 4, 3, 4, 4, 5,
        2, 1, 2, 1, 3, 1, 2, 1, 2, 3
    ],
    'Trust': [
        4, 5, 3, 4, 4, 5, 2, 3, 4, 4,
        2, 1, 3, 1, 2, 1, 2, 1, 1, 2,
        3, 5, 3, 4, 4, 5, 3, 4, 4, 5,
        2, 1, 2, 1, 2, 1, 2, 1, 2, 2,
        4, 5, 3, 4, 4, 5, 3, 4, 4, 5,
        2, 1, 2, 1, 2, 2, 2, 1, 2, 2,
        3, 5, 4, 5, 4, 5, 3, 4, 4, 5,
        2, 1, 2, 1, 2, 1, 2, 1, 2, 3
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
