# Sampling

import pandas as pd

def simple_random_sample(dataset, sample):
    """
    Simple random sample
    - replace = False: not repeat register
    - same result, use a seed in random_state
    """
    return dataset.sample(n=sample, replace=False, random_state=1)


# Loading data
dataset = pd.read_csv("data/raw_data/census.csv")

# Creating a sample 
df_random_sample = simple_random_sample(dataset, 100)
