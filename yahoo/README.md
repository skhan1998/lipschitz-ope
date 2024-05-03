This folder contains code to replicate the experiments on the Yahoo! Webscope dataset in Section 5 of the main paper. The files it contains are:

- parser.py, which parses the raw data in the file ydata-fp-td-clicks-v1_0.20090501.gz downloaded from dataset R6A at https://webscope.sandbox.yahoo.com/catalog.php?datatype=r

- the data files ts_1241180700_date_20090501_articles.pkl and ts_1241180700_date_20090501_clicks.pkl, which are generated by running the command python3 parser.py '1241180700'

- yahoo_experiment.py,  which runs the main experiment on the yeast data and generates the file yahoo_results_final.pkl

- yahoo_plots.ipynb, which generates Figures 3 and 5 in the paper, as well as summary statistics reported in Section 5 