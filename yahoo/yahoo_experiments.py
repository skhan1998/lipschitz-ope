from sklearn.linear_model import LogisticRegression
import argparse
import sys
import seaborn as sns
import pickle
from tqdm.notebook import tqdm
import re
import pandas as pd
import numpy as np
#from plotnine import *
sys.path.insert(0, "../")
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed 

from ope_estimators import *

TIMESTAMP = "1241180700"

def load_data(ts):
    with open(f"ts_{ts}_date_20090501_clicks.pkl", "rb") as f:
        df = pickle.load(f)

    df["Y"] = df["Y"].astype(int)

    with open(f"ts_{ts}_date_20090501_articles.pkl", "rb") as f:
        article_df = pickle.load(f)

    article_df.columns = ["V0", "V1", "V2",
                          "V3", "V4", "V5", "A", "click_rate"]

    return df, article_df

def construct_ope_df(df, article_df):
    pool = df["pool"][0]

    threshold = np.median(article_df["V0"])

    subpool = article_df.query(f"V0 > {threshold}")["A"].values

    ope_df = pd.merge(df, article_df.filter(["A", "V0"]), on="A")
    ope_df["in_subpool"] = ope_df["V0"] > threshold

    ope_df = ope_df.filter(
        ["A", "Y", "X0", "X1", "X2", "X3", "X4", "in_subpool"])

    return ope_df, pool, subpool

def get_policy(cutoff, pool, subpool):
    def pi(row, a):
        if row["X3"] > cutoff:
            return 1/len(pool)
        elif int(a) in subpool:
            return 1/len(subpool)
        else:
            return 0 
        
    return pi 

def setup_ope_experiment(ope_df, cutoff_b, cutoff_e, pool, subpool):
    
    pi_b = get_policy(cutoff_b, pool, subpool)
    pi_e = get_policy(cutoff_e, pool, subpool)

    pi_b_dict = {action: ope_df.apply(lambda x: pi_b(
        x, action), axis=1).values for action in pool}
    pi_e_dict = {action: ope_df.apply(lambda x: pi_e(
        x, action), axis=1).values for action in pool}

    return pi_b_dict, pi_e_dict

def compute_unbiased_estimate(df, cutoff_e, pool, subpool):
    pi_e = get_policy(cutoff_e, pool, subpool)
    
    weights = df.apply(lambda x: pi_e(x, str(x["A"]))/0.05, axis=1).values
    estimate = np.mean(weights*df["Y"].values)
    sigma_hat = np.sqrt(np.var(weights*df["Y"].values)/len(df))
    
    return estimate, sigma_hat    

def run_ope_experiment(ope_df, D, pi_b_dict, pi_e_dict, model, pool):

    Estimator = MultiActionOPEEstimator(ope_df[["X0", "X1", "X2", "X3", "X4"]].values,
                                        ope_df["Y"].values,
                                        ope_df["A"].values,
                                        pi_b_dict,
                                        pi_e_dict,
                                        0.5,
                                        pool,
                                        "LipImputeBddRespEstimator",
                                        D = D,
                                        model = model,
                                        M_lower = 0,
                                        M_upper = 1)
    
    L_grid = list(np.linspace(0.2, 1, 50))
    L_grid += [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #L_grid = np.linspace(0.2, 0.75, 1)

    Analysis = SensitivityAnalysis(Estimator, L_grid)

    Analysis.compute_pid_intervals()

    return Analysis

def make_plot(Analysis, unbiased_estimate, sigma_hat, path):

    plot_df = Analysis.result_df

    plot_df["imputation"] = np.sum(
        [x for x in Analysis.estimator.imputation_values.values()])
    plot_df["unbiased"] = unbiased_estimate
    plot_df["ci_lower"] = unbiased_estimate-1.96*sigma_hat
    plot_df["ci_upper"] = unbiased_estimate+1.96*sigma_hat
    
    plot_df["manski_upper"] = np.sum([x for x in Analysis.estimator.manski_upper.values()])
    plot_df["manski_lower"] = np.sum([x for x in Analysis.estimator.manski_lower.values()])

    plot_df = pd.melt(plot_df, id_vars="L",
                  value_vars=["imputation", "unbiased", "psi_plus", "psi_minus", 
                              "ci_upper", "ci_lower", "manski_upper", "manski_lower"])
        
    # else:
    #     plot_df = pd.melt(plot_df, id_vars="L",
    #                   value_vars=["imputation", "unbiased", "psi_plus", "psi_minus", 
    #                               "ci_upper", "ci_lower"])

    def get_method(row):
        if row["variable"][0] == "p":
            return "lip"
        elif row["variable"][0] == "m":
            return "manski"
        elif row["variable"][0] == "c":
            return "unbiased"
        else:
            return row["variable"]

    def get_type(row):
        if row["method"] == "manski" or row["method"] == "lip":
            return "partial_id"
        elif row["variable"][0] == "c":
            return "ci"
        else:
            return "point_estim"

    plot_df["method"] = plot_df.apply(lambda row: get_method(row), axis=1)
    plot_df["type"] = plot_df.apply(lambda row: get_type(row), axis=1)

    with open(path+"_plot_df.pkl", "wb") as f:
        pickle.dump(plot_df, f)
        
    return plot_df 

    #p = (ggplot(plot_df, aes(x="L", group="variable", color="method", linetype="type"))
         #+ geom_line(aes(y="value"))
         #+ labs(x="Lipschitz constant, L", y="Off-policy value",
         #       color="Method", linetype="Approach")
         #+ theme_bw()
         #+ scale_color_manual({"imputation": "blue", "lip": "black",
         #                      "unbiased": "orange", "manski": "grey"})
         #+ scale_linetype_manual(["solid", "dashed"])
         #+ theme(legend_position="bottom", legend_box_spacing=.5))

    #ggsave(p, path, dpi=200)
    
    
def run_experiment(cutoff_e, ts, cutoff_b = 0.5):
    
    df, article_df = load_data(ts)
    ope_df, pool, subpool = construct_ope_df(df, article_df)

    ope_df = ope_df.query(f"in_subpool or X3 > {cutoff_b}")

    pi_b_dict, pi_e_dict = setup_ope_experiment(
        ope_df, cutoff_b, cutoff_e, pool, subpool)
    
    unbiased_estimate, sigma_hat = compute_unbiased_estimate(df, cutoff_e, pool, subpool)

    model = LogisticRegression()
    
    X = ope_df[["X0", "X1", "X2", "X3", "X4"]].values
    D = pairwise_distances(X)
        
    Analysis = run_ope_experiment(ope_df, D, pi_b_dict, pi_e_dict, model, pool)

    base_path = f"ts_{ts}_cutoff_e_{cutoff_e}_cutoff_b_{cutoff_b}"

    with open(base_path+"_analysis.pkl", "wb") as f:
        pickle.dump(Analysis, f)

    plot_df = make_plot(Analysis, unbiased_estimate, sigma_hat, base_path)
    
    plot_df["T"] = cutoff_e
    
    return plot_df


if __name__ == "__main__":
    cutoff_grid = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    result_dfs = Parallel(n_jobs=len(cutoff_grid), verbose = 10)(delayed(lambda c : run_experiment(c, TIMESTAMP))(cutoff) for cutoff in cutoff_grid)   
    
    with open("yahoo_results_final.pkl", "wb") as f:
        pickle.dump(pd.concat(result_dfs), f)
    

