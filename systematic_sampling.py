# systematic sampling

import pandas as pd
import random
import numpy as np



def systematic_sampling(dataset, sampling):
    random.seed(1)
    range_dataset = len(dataset) //  sampling
    first_number = random.randint(0, range_dataset)
    return np.arange(first_number, len(dataset), step=range_dataset)


# Loading data
dataset = pd.read_csv("data/raw_data/census.csv")
systematic_sampling = systematic_sampling(dataset, 100)

