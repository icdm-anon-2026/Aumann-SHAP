import pandas as pd

df = pd.read_csv(r'C:\Users\METASUP\Aumann-SHAP\experiments\german_credit\cache\global_long_GLOBAL_rs1_thr30_t080_diceN8000_m5_tau0005_seed123.csv')

def rank_flip(grp):
    eq = grp.sort_values('phi_eq', ascending=False)['feature'].values
    mi = grp.sort_values('S_micro', ascending=False)['feature'].values
    return not all(eq == mi)

g = df.groupby('idx').apply(rank_flip)
print('Rank-flip rate:', round(g.mean(), 3))
print('N instances:', len(g))