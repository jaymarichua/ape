#######################################################
# anova_llm.py
# Run a 2x2 ANOVA (Expertise × Disclosure)
# on the COMPLETE LLM dataset. Includes subgroup checks.
#######################################################

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_anova_llm():
    # ---------------------------
    # 1) Full LLM dataset
    # ---------------------------
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
            2,1,3,2,2,1,4,3,2,1,
            4,5,3,4,4,5,3,4,5,4,
            2,1,3,2,2,1,3,2,2,1,
            4,5,3,4,4,5,3,4,4,3,
            2,1,3,2,2,1,3,2,2,1,
            4,5,3,4,4,4,3,4,4,3,
            2,1,3,1,2,1,3,2,2,1,
            4,5,3,4,4,5,3,4,4,3
        ],
        'Fairness': [
            3,4,2,3,3,4,2,3,4,4,
            2,1,2,2,3,1,2,1,2,3,
            3,4,3,4,3,4,2,3,4,4,
            2,1,2,1,3,1,2,1,2,3,
            3,4,3,4,3,4,2,3,4,4,
            2,1,2,1,3,2,2,1,2,3,
            4,4,3,4,4,4,3,4,4,5,
            2,1,2,1,3,1,2,1,2,3
        ],
        'Trust': [
            4,5,3,4,4,5,2,3,4,4,
            2,1,3,1,2,1,2,1,1,2,
            3,5,3,4,4,5,3,4,4,5,
            2,1,2,1,2,1,2,1,2,2,
            4,5,3,4,4,5,3,4,4,5,
            2,1,2,1,2,2,2,1,2,2,
            3,5,4,5,4,5,3,4,4,5,
            2,1,2,1,2,1,2,1,2,3
        ]
    }
    df_llm = pd.DataFrame(data)

    # Convert factors to categorical
    df_llm['Expertise'] = df_llm['Expertise'].astype('category')
    df_llm['Disclosure'] = df_llm['Disclosure'].astype('category')

    # -----------------------------------------------------------------
    # DEBUG: Print out group counts + some stats to see if subgroups exist
    # -----------------------------------------------------------------
    print("\nGroup sizes by Expertise×Disclosure:")
    group_sizes = df_llm.groupby(['Expertise','Disclosure']).size()
    print(group_sizes, "\n")

    # Also check standard dev in each cell, to detect zero-variation groups
    for (exp, disc), subset in df_llm.groupby(['Expertise','Disclosure']):
        msg = f"{exp}, {disc}: n={len(subset)} "
        msg += f"| Tox.std={subset['Toxicity'].std(ddof=1):.2f}, "
        msg += f"Fair.std={subset['Fairness'].std(ddof=1):.2f}, "
        msg += f"Trust.std={subset['Trust'].std(ddof=1):.2f}"
        print(msg)
    print()

    # -----------------------------------------------
    # 2) For each DV, run a 2x2 ANOVA (Expertise × Disclosure)
    # -----------------------------------------------
    dependent_vars = ['Toxicity', 'Fairness', 'Trust']
    for dv in dependent_vars:
        formula = f'{dv} ~ C(Expertise) * C(Disclosure)'
        model = ols(formula, data=df_llm).fit()

        print("==========================================")
        print(f"ANOVA for DV = {dv}")
        try:
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)
        except Exception as e:
            print(f"ERROR running ANOVA on {dv}: {e}")
        print("==========================================\n")


if __name__ == "__main__":
    run_anova_llm()
