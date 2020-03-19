"""

Script to normalize the data in /Data and 

Author: jweber
Date: 09.03.2020
"""

import pandas as pd 
import numpy as np 

def load_normalize_data(normalize=True):

    data = pd.read_csv('../Data/Table_alpha_Data.txt', header=0, dtype=np.float64)
    assert(np.any(data.isna()) == False), "There are NaN's in the dataframe - please check this!"

    descriptors = {
        "min": data.min().values,
        "max": data.max().values
    }

    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
!

    return (data, descriptors)

