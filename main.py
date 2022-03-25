from CTAB_GAN.model.ctabgan import CTABGAN
# Importing the evaluation metrics
from CTAB_GAN.model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
# Importing standard libraries
import numpy as np
import pandas as pd
import glob

# Specifying the replication number
num_exp = 1
# Specifying the name of the dataset used
dataset = "Czech_Bank"
# Specifying the path of the dataset used
real_path = "./src/main/resources/trans.csv"
# Specifying the root directory for storing generated data
fake_file_root = "fake_dataset"

synthesizer = CTABGAN(raw_csv_path=real_path,
                      test_ratio=0.20,
                      categorical_columns=['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account'],
                      log_columns=['amount', 'balance'],
                      mixed_columns={'k_symbol': [0.0], 'bank': [0.0], 'account': [0.0]},
                      integer_columns=['year', 'month', 'day', 'dayofweek', 'amount', 'balance'],
                      problem_type={"Classification": 'type'},
                      epochs=100)


if __name__ == '__main__':
    # Fitting the synthesizer to the training dataset and generating synthetic data
    for i in range(num_exp):
        synthesizer.fit()
        syn = synthesizer.generate_samples()
        syn.to_csv(fake_file_root + "/" + dataset + "/" + dataset + "_fake_{exp}.csv".format(exp=i), index=False)
