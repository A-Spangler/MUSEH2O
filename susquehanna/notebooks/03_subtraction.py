import os
import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import csv
import logging
from itertools import chain

logging.basicConfig(level=logging.INFO)
plt.rcParams["figure.figsize"] = [12, 8]
# sys.path.append('..')

#TODO: is release a yearly or daily logged varb?

sys.path.append(os.path.abspath(".."))
from susquehanna_model import SusquehannaModel
import rbf_functions


rbfs = [
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]

reference_sets = {}
for entry in rbfs:
    name = entry.__name__
    reference_sets[name] = pd.read_csv(
        os.path.join("./refsets", f"{name}_refset_with_variables.csv")
    )


for filename in os.listdir("../data1999"):
    if filename.startswith("w"):
        globals()[f"{filename[:-4]}"] = np.loadtxt(f"../data1999/{filename}")
    elif filename == "salinity_min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
           os.path.join("../data1999", filename)
        )
    elif filename == "min_flow_req.txt":
        globals()[f"{filename[:-4]}"] = np.loadtxt(
            os.path.join("../data1999", filename)
        )

entry = rbfs[0]
reference_set = reference_sets[entry.__name__]

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Specify the first RBF entry to use
entry = rbfs[0]
name = entry.__name__
reference_set = reference_sets[name]

numberOfRBF = 6  # numberOfInput + 2
n_inputs = 2  # (time, storage of Conowingo)
n_outputs = 4  # Atomic, Baltimore, Chester, Downstream:- (hydropower, environmental)
n_rbfs = n_inputs + 2
rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

# Initialize model
nobjs = 6
n_years = 1
susquehanna_river = SusquehannaModel(
    108.5, 505.0, 5, n_years, rbf
)  # l0, l0_MR, d0, years
susquehanna_river.set_log(True)

# Evaluate the model for each row in the reference set
for _, row in reference_set.iloc[:, 0:32].iterrows():
    susquehanna_river.evaluate(row)

# Retrieve outputs
level_CO, level_MR, ratom, rbalt, rches, renv = susquehanna_river.get_log()

#convert renv and min_flow to numpy arrays
renv = np.array(renv)
min_flow_req = np.array(min_flow_req)

# areas between min_flow_req and each release in renv
areas_ferc = []
days = 365

for release in renv:
    if release < min_flow_req:
        area = (min_flow_req - release)
    else:
        area = 0

    yr_area = 0
    for i in range(days):
        yr_area += area
    areas_ferc.append(yr_area)

print(areas_ferc)
