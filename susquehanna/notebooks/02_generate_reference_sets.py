#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pareto
import os
from itertools import chain
import rbf_functions
import sys

sys.path.append("..")

# # Load solutions for each RBF for all seeds:

#rbf_functions is a class of functions defined in the rbf directory
rbfs = [
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]

# rbf_functions previously executed, __name__solution.csv files produced
# the code pulls these solutions and organizes them into pareto_sets dictionary
# pareto_sets dictionary contains each rbf and the associated solutions or pareto set
# __name__solution.csv files are generated in main_susquehana.py

pareto_sets = {}
for entry in rbfs:
    sollist = []
    solutions = []
    name = entry.__name__
    output_dir = f"../output/{name}/"
    for filename in os.listdir(output_dir):
        if filename.endswith("solution.csv"): #loads in __name__solution.csv from ../output/name/ dir
            sollist.append(filename[:-4])
            df_temp = pd.read_csv(f"{output_dir}{filename}", header=0) 
            solutions.append(df_temp.values.tolist())

    pareto_sets[name] = list(chain.from_iterable(solutions))
''''
pareto_sets.keys()
print("Amount of solutions for each RBF:")
for rbf in pareto_sets:
    print(rbf, len(pareto_sets[rbf]))
print(f"Loaded into list 'solutions': {', '.join(sollist)}")
'''

rbfs = [
    rbf_functions.squared_exponential_rbf,
    rbf_functions.original_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]

# iterates over each RBF with different seeds
# loads RBF solutions and decision variables
# organizes into pareto_sets dictionary
pareto_sets = {}
for entry in rbfs:
    sets_per_seed = []
    name = entry.__name__
    output_dir = f"../output/{name}/"

    for seed in np.arange(10, 110, 10):
        solutions = pd.read_csv(os.path.join(output_dir, f"{seed}_solution.csv"))
        variables = pd.read_csv(
            os.path.join(output_dir, f"{seed}_variables.csv"), header=None
        )

        combined = pd.concat([variables, solutions], axis=1)
        sets_per_seed.append(combined)

    pareto_sets[name] = pd.concat(sets_per_seed)

# #sort to find nondominated solutions, create reference set
# also includes the associated decision variables
reference_sets = {}
for rbf in pareto_sets:
    data = pareto_sets[rbf] #list

    #print(rbf, len(data))
    nondominated = pareto.eps_sort(
        [data.values],
        [32, 33, 34, 35, 36, 37], # objectives indicies by collumn number
        [0.5, 0.05, 0.05, 0.05, 0.001, 0.05], # epsilons
        maximize=[32, 33, 34, 35, 37],
    )
    reference_sets[rbf] = nondominated
    df_nondom = pd.DataFrame(nondominated, columns=data.columns)
    #print(rbf, len(nondominated))
    df_nondom.to_csv(
        f"./refsets/{rbf}_refset_with_variables.csv", index=False, header=True
    )

# Create ref set without variables
# rbf is a string with rbf names
# pareto_sets is a dictionary containing rbf name and a list of solutions
# Doing a nondominated sort on a Pandas Data Frame requires itertuples(False)
reference_sets = {}
for rbf in pareto_sets:
    print(rbf, len(pareto_sets[rbf]))

    # Convert the list of solutions to a DataFrame
    df = pd.DataFrame(pareto_sets[rbf], columns=[
        "hydropower",
        "atomicpowerplant",
        "baltimore",
        "chester",
        "environment",
        "recreation"
    ])
    nondominated = pareto.eps_sort(
        [list(df.itertuples(index=False))],  # Use itertuples(False) to get rows without the index
        [0, 1, 2, 3, 4, 5],
        [0.5, 0.05, 0.05, 0.05, 0.001, 0.05],
        maximize=[0, 1, 2, 3, 5]
    )
    reference_sets[rbf] = nondominated
    df_nondom = pd.DataFrame(
        nondominated,
        columns=[
            "hydropower",
            "atomicpowerplant",
            "baltimore",
            "chester",
            "environment",
            "recreation",
        ],
    )
    print(rbf, len(df_nondom))
    df_nondom.to_csv(f"../notebooks/refsets/{rbf}_refset.csv", index=False, header=True)


# # Find decision variables that belong to the generated refset:

d_refvar = {}
d_refsol = {}
for entry in rbfs:
    name = entry.__name__
    df_sol = pd.DataFrame(
        columns=[
            "hydropower",
            "atomicpowerplant",
            "baltimore",
            "chester",
            "environment",
            "recreation",
        ]
    )
    df_var = pd.DataFrame(
        columns=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ]
    )
    output_dir = f"../output/{name}"

    for filename in os.listdir(output_dir):
        # load
        if filename.endswith("solution.csv"):
            df_ts = pd.read_csv(f"{output_dir}/{filename}", header=0)
            df_sol = pd.concat([df_sol, pd.DataFrame(df_sol)])
        elif filename.endswith("variables.csv"):
            df_tv = pd.read_csv(f"{output_dir}/{filename}", header=None)
            df_var = pd.concat([df_var, pd.DataFrame(df_tv)])

    sol = df_sol.values.tolist()
    var = df_var.values.tolist()
    ref = reference_sets[name]
    refvar = []
    refsol = []

    for idx, value in enumerate(sol):
        if value in ref:
            refvar.append(var[idx])
            refsol.append(sol[idx])
    d_refsol[name] = pd.DataFrame(
        refsol,
        columns=[
            "hydropower",
            "atomicpowerplant",
            "baltimore",
            "chester",
            "environment",
            "recreation",
        ],
    )
    d_refvar[name] = pd.DataFrame(refvar)
    d_refsol[name].to_csv(
        f"../notebooks/refsets/{name}_refset.csv", index=False, header=True
    )
    d_refvar[name].to_csv(
        f"../notebooks/refsets/{name}_refset_variables.csv", index=False, header=False
    )


for entry in rbfs:
    name = entry.__name__
    print(name)
    print(f"refset: {len(d_refsol[name])}")
    print(f"varset: {len(d_refvar[name])}")


# # Generate global reference set for all RBFs:

x = 0
for rbf in pareto_sets:
    x += len(pareto_sets[rbf])
    print(rbf, len(pareto_sets[rbf]))
print("total:", x)

pareto_set = {}
sollist = []
solutions = []
for entry in rbfs:
    name = entry.__name__
    output_dir = f"../output/{name}/"
    for filename in os.listdir(output_dir):
        if filename.endswith("solution.csv"):
            sollist.append(filename[:-4])
            df_temp = pd.read_csv(f"{output_dir}{filename}", header=0)
            solutions.append(df_temp.values.tolist())
pareto_set = list(chain.from_iterable(solutions))
len(pareto_set)


print(len(pareto_set))
nondominated = pareto.eps_sort(
    [pareto_set],
    [0, 1, 2, 3, 4, 5],
    [0.5, 0.05, 0.05, 0.05, 0.001, 0.05],
    maximize=[0, 1, 2, 3, 5],
)
df_nondom = pd.DataFrame(
    nondominated,
    columns=[
        "hydropower",
        "atomicpowerplant",
        "baltimore",
        "chester",
        "environment",
        "recreation",
    ],
)
print(len(nondominated))
df_nondom.to_csv(f"../notebooks/refsets/global_refset.csv", index=False, header=True)
