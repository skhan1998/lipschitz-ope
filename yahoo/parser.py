import numpy as np 
import pandas as pd
import re 
import pickle 
import argparse

ftr_re = re.compile("\d\.\d*")
id_re = re.compile("^\d*")

def read_logs(target_ts):
    click_logs = [] 
    article_logs = [] 
    articles_parsed = False
    pool = [] 

    with open("ydata-fp-td-clicks-v1_0.20090501", "r") as f:
        while True:        
            line = f.readline()
            if not line:
                break 
            
            split_line = line.split("|")
            click_data, user_data, articles = split_line[0], split_line[1], split_line[2:]
            timestamp, action, response = click_data.split()

            if timestamp == target_ts:
                user_features = np.array(ftr_re.findall(user_data)).astype(float)
                if not articles_parsed:
                    for article in articles: 
                        article_id = id_re.search(article).group(0)
                        pool.append(article_id)

                        article_features = np.array(ftr_re.findall(article)).astype(float)
                        article_dict = {f"V{i}":article_features[i] for i in range(len(article_features))}
                        article_dict.update({"id":article_id})

                        article_logs.append(article_dict)

                    articles_parsed = True 
                        
                click_dict = {"ts":timestamp, "A":action, "Y":response, "pool":pool}
                click_dict.update({f"X{i}":user_features[i] for i in range(len(user_features))})

                click_logs.append(click_dict) 

    click_df = pd.DataFrame(click_logs)
    click_df["Y"] = click_df["Y"].astype(int)
    click_df["A"] = click_df["A"].astype(int)

    article_df = pd.DataFrame(article_logs)        
    article_df["id"] = article_df["id"].astype(int)

    click_rates = (click_df
        .groupby(["A"])
        .agg({"Y":"mean"})
        .reset_index())

    click_rates.columns = ["id", "click_rate"]
    article_df = pd.merge(article_df, click_rates, on = "id")

    
    (article_df
    .corr(numeric_only=True)
    .to_csv(f"article_corrs/ts_{target_ts}_date_20090501_article_corr.txt", header=True, index=True, sep='\t', float_format = "%.3f"))

    (click_df
    .corr(numeric_only = True)
    .to_csv(f"click_corrs/ts_{target_ts}_date_20090501_click_corr.txt", header=True, index=True, sep='\t', float_format = "%.3f"))

    (click_df
    .describe()
    .to_csv(f"summaries/ts_{target_ts}_date_20090501_summary.txt", header=True, index=True, sep='\t', float_format = "%.3f"))

    with open(f"dfs/ts_{target_ts}_date_20090501_clicks.pkl", "wb") as f:
        pickle.dump(click_df, f)   

    with open(f"dfs/ts_{target_ts}_date_20090501_articles.pkl", "wb") as f:
        pickle.dump(article_df, f)  


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("ts", type=str)
    args = parser.parse_args()

    read_logs(args.ts)

 
