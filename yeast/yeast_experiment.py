from plotnine import *
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pickle
import seaborn as sns
import sys
from siuba import group_by, summarize, filter, mutate, arrange, spread, gather, _
from siuba.experimental.pivot import pivot_wider
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

sys.path.insert(0, "../")
from ope_estimators import *

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()


def load_data():
    with open("yeast.table", "r") as f:
        df = pd.read_table(f, header=None, delim_whitespace=True)

    df.columns = ["Name", "X1", "X2", "X3",
                  "X4", "X5", "X6", "X7", "X8", "Outcome"]

    df = (df >> filter(_.Outcome != "ERL")
          >> filter(_.Outcome != "POX")
          >> filter(_.Outcome != "VAC")
          >> filter(_.Outcome != "EXC")
          >> filter(_.Outcome != "ME1")
          >> filter(_.Outcome != "ME2"))

    K = df["Outcome"].nunique()
    n = len(df)

    classes = df["Outcome"].unique()
    class_labels = [i for i in range(K)]

    label_map = {classes[i]: class_labels[i] for i in range(K)}

    df["Y"] = df.apply(lambda r: label_map[r["Outcome"]], axis=1)

    return df, class_labels


def behavior_transformation(policy):
    new_policy = np.array([x if x > 0.05 else 0 for x in policy])
    new_policy = new_policy/sum(new_policy)

    return new_policy


def generate_ope_problem(n, df):
    model = LogisticRegression(penalty="none", n_jobs=1, solver="newton-cg")

    df_sample = df.sample(n, replace=True)
    X = df_sample[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values
    Y = df_sample["Y"].values

    model.fit(X, Y)

    K = df["Outcome"].nunique()

    class_labels = [i for i in range(K)]

    eval_probs = model.predict_proba(X)

    pi_e = {i: eval_probs[:, i] for i in range(K)}

    behavior_probs = np.apply_along_axis(
        behavior_transformation, 1, eval_probs)

    pi_b = {i: behavior_probs[:, i] for i in range(K)}

    A = np.array([np.random.choice(class_labels, p=behavior_probs[i, :])
                  for i in range(n)])

    rewards = (A == Y).astype(int)

    pi_e_value = 0
    for i in range(n):
        pi_e_value += eval_probs[i, :][Y[i]]/n

    return X, rewards, A, pi_b, pi_e, pi_e_value


def run_trial(n, L_grid, df, trial):
    X, rewards, A, pi_b, pi_e, pi_e_value = generate_ope_problem(n, df)

    K = df["Outcome"].nunique()
    class_labels = [i for i in range(K)]

    D = pairwise_distances(X)
    result_df = []

    for L in L_grid:
        model = LogisticRegression()

        Estimator = MultiActionOPEEstimator(X, rewards, A, pi_b, pi_e, L, class_labels,
                                            "LipImputeBddRespEstimator", D=D,
                                            model=model, M_lower=0, M_upper=1)
        lower, upper = Estimator.psi_hat()

        coverage = (lower - 0.01 < pi_e_value) and (upper + 0.01 > pi_e_value)

        result = {"trial": trial, "n": n, "L": str(L), "lower": lower, "upper": upper,
                  "cover": coverage.astype(int), "pi_e_value": pi_e_value}
        result_df.append(result)

    manski_lower, manski_upper = sum(Estimator.manski_lower.values()), sum(
        Estimator.manski_upper.values())
    coverage = (manski_lower - 0.01 <
                pi_e_value) and (manski_upper + 0.01 > pi_e_value)

    result = {"trial": trial, "n": n, "L": "inf", "lower": manski_lower, "upper": manski_upper,
              "cover": coverage.astype(int), "pi_e_value": pi_e_value}
    result_df.append(result)

    return pd.DataFrame.from_records(result_df)


def run_simulation(n, L_grid, df, trials=10000):
    logger.info(f"Running n={n}")
    result_dfs = Parallel(n_jobs=15, verbose=10)(delayed(lambda x: run_trial(n, L_grid, df, x))(trial)
                                                 for trial in range(trials))

    return pd.concat(result_dfs)


if __name__ == "__main__":

    df, class_labels = load_data()

    n_grid = [1000, 2000, 3000, 4000, 5000, 10000]
    L_grid = [1, 2, 3, 4, 5]

    result_dfs = []

    for n in n_grid:
        logger.info(f"Running n={n}")
        result = run_simulation(n, L_grid, df)
        result_dfs.append(result)

        with open(f"result_df_n_{n}_trials_10000.pkl", "wb") as f:
            pickle.dump(result, f)

    with open("result_df_full_trials_10000.pkl", "wb") as f:
        pickle.dump(pd.concat(result_dfs), f)
