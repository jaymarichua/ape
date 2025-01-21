# anova_sc2.py (example script for StarCraft II data)

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_anova_sc2():
    # ---------------------------
    # 1) Copy the SC2 dictionary from sc2_stats.py
    # ---------------------------
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
            'Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice',
            'Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert',
            'Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice',
            'Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert',
            'Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice',
            'Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert',
            'Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice','Novice',
            'Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert','Expert'
        ],
        'Disclosure': [
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman',
            'No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman','No','Superhuman'
        ],
        'Toxicity': [
            4,3,5,2,4,3,5,2,4,3,
            5,4,5,3,5,3,5,2,5,3,
            3,2,4,3,4,2,5,3,4,2,
            5,3,5,2,5,3,5,2,5,3,
            4,2,5,3,4,2,5,3,4,2,
            5,3,5,2,5,3,5,2,5,3,
            4,2,4,3,4,2,5,3,4,2,
            5,3,5,2,5,3,5,2,5,3
        ],
        'Fairness': [
            2,3,1,4,2,3,1,3,2,4,
            2,3,1,4,2,3,1,4,2,3,
            3,4,2,3,2,4,1,3,2,4,
            2,3,1,4,2,3,1,4,2,3,
            2,4,1,3,2,4,1,3,2,4,
            1,4,2,3,2,3,1,4,2,3,
            3,4,2,4,3,4,1,4,2,4,
            1,4,1,3,2,4,1,4,2,3
        ],
        'Trust': [
            3,5,2,4,3,4,1,5,3,4,
            2,3,1,5,3,4,1,5,2,4,
            3,5,2,4,3,5,2,4,3,5,
            2,4,1,5,3,4,1,5,2,4,
            3,4,2,4,3,5,1,4,3,5,
            2,4,1,5,3,4,1,5,2,4,
            2,5,3,4,2,5,2,4,3,5,
            1,4,1,5,2,4,1,5,2,4
        ]
    }

    df_sc2 = pd.DataFrame(data)

    # Convert Expertise and Disclosure to categorical
    df_sc2['Expertise'] = df_sc2['Expertise'].astype('category')
    df_sc2['Disclosure'] = df_sc2['Disclosure'].astype('category')

    # -----------------------------------------------
    # 2) For each DV, run a 2x2 ANOVA (Expertise × Disclosure)
    # -----------------------------------------------
    dependent_vars = ['Toxicity', 'Fairness', 'Trust']
    for dv in dependent_vars:
        formula = f'{dv} ~ C(Expertise) * C(Disclosure)'
        model = ols(formula, data=df_sc2).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print("==================================================")
        print(f"ANOVA for DV = {dv} (StarCraft II dataset)")
        print(anova_table)
        print("==================================================\n")

if __name__ == "__main__":
    run_anova_sc2()
