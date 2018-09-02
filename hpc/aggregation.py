# Aggregate the results and generate a plot
#
# Authors: Parisa Zahedi
#
# Dependencies: sklearn, numpy
# License: BSD-3-Clause

# pylint: disable=C0321

import os
import re
import glob
import warnings

from sklearn import metrics
import numpy as np
import pandas as pd
import json
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# store the result
column_names = [
    'T', 'tn', 'fp', 'fn', 'tp', 'allowed_FN', 'init_included_papers',
    'dataset', 'seed', 'pred', 'training_size'
]
df_results = pd.DataFrame(columns=column_names)
# save the result to a file
for fp in glob.glob(os.path.join('output', 'sr_lstm*.json')):
    with open(fp) as json_file:
        # load the result into memory
        json_data = json.load(json_file)
        json_data['pred'] = [json_data['pred']]
    df = pd.DataFrame(json_data)
    df_results = df_results.append(df)

df_result['recall'] = df_result['tp'] / (df_result['tp'] + df_result['fn'])
df_result['fp_rate'] = df_result['fp'] / (df_result['fp'] + df_result['fn'])

selected_col_names = [
    'T', 'allowed_FN', 'init_included_papers', 'dataset', 'fp_rate', 'recall'
]

df_csv = df_results.loc[:, df_results.columns != 'pred']
df_csv.to_csv(os.path.join('output', 'sr_lstm_results.csv'))

df_bad = df_results[df_results['T'] == 475]
d = df_bad.iloc[0]['pred']
pd.DataFrame(d).to_csv(os.path.join('output', 'badTp475.csv'))

# prediction = np.loadtxt(fp)
# regexp = re.search('.*digits_svm.*\_C\_(.*)\_gamma\_(.*)\.txt', fp)
# c = regexp.group(1)
# gamma = regexp.group(2)
# f1score = metrics.f1_score(data_test_target, prediction, average='macro')

# result.append({'f1': f1score, 'cost': c, 'gamma': gamma})

# optimal = max(result, key=lambda x: x['f1'])
# print("Grid length:", len(result))
# print("Optimal settings:", optimal)

# # make a plot
# import pandas as pd
# import seaborn as sns

# df_result = pd.DataFrame(result)
# df_result['cost'] = df_result['cost'].astype(float)
# df_result['gamma'] = df_result['gamma'].astype(float)
# df_result = df_result.pivot(index='cost', columns='gamma', values='f1')
# sns_fig = sns.heatmap(df_result)
# figure = sns_fig.get_figure()
# figure.savefig(os.path.join("output", "digits_f1_plot.pdf"))
