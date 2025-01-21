# anova_llm.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_anova_llm():
    # ---------------------------
    # 1) Copy the LLM dictionary from llm_stats.py
    # ---------------------------
    data = {
        'PersonaID': [...],  # truncated for brevity
        'Expertise': [...],
        'Disclosure': [...],
        'Toxicity': [...],
        'Fairness': [...],
        'Trust': [...]
    }
    df_llm = pd.DataFrame(data)

    # Convert Expertise and Disclosure to categorical
    df_llm['Expertise'] = df_llm['Expertise'].astype('category')
    df_llm['Disclosure'] = df_llm['Disclosure'].astype('category')

    # -----------------------------------------------
    # 2) For each DV, run a 2x2 ANOVA (Expertise × Disclosure)
    # -----------------------------------------------
    dependent_vars = ['Toxicity', 'Fairness', 'Trust']
    for dv in dependent_vars:
        formula = f'{dv} ~ C(Expertise) * C(Disclosure)'
        model = ols(formula, data=df_llm).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print("==================================================")
        print(f"ANOVA for DV = {dv} (LLM dataset)")
        print(anova_table)
        print("==================================================\n")

if __name__ == "__main__":
    run_anova_llm()
