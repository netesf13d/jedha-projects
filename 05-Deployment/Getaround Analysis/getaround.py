# -*- coding: utf-8 -*-
"""
Script for Getaround project.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# %% Loading
"""
## <a name="loading"></a> Data loading
"""

##
pp_df = pd.read_csv('./get_around_pricing_project.csv')
pp_df = pp_df.drop('Unnamed: 0', axis=1)
pp_df

##
da_df = pd.read_excel('./get_around_delay_analysis.xlsx')
da_df

"""
The dataset is split in two parts, with 4843 observations each.
- The first dataframe, `pp_df`, contains the information that we have *before* a car is rent.
We will use these features as predictors for our machine learning model
- The second dataframe contains information relative to the *outcome* of a rental. This data cannot be used to make predictions.
However, we will use it to build target feature to train our ML model.

We should include all the data available in our exploratory data analysis. we build a combined dataframe `eda_df` for this purpose.
"""
eda_df = pd.concat([da_df, pp_df], axis=1)
eda_df


# %% EDA
"""
## <a name="eda"></a> Preliminary data analysis


"""

##
a = da_df['car_id'].astype(object).describe()

##
b = da_df.loc[:, ['delay_at_checkout_in_minutes', 'time_delta_with_previous_rental_in_minutes']].describe()

##
c = da_df.loc[:, ['checkin_type', 'state']].describe()


"""
super nice
"""

"""
### some graphs

"""



# %% ML










