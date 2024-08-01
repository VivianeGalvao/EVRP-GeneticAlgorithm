# author: Viviane de Jesus Galvao
import pandas as pd

from problem import EVRP
from genetic_algorithm import AGPermutation

n_pop = 50
prob_crossover = 0.9
prob_mutation = 0.7

root = 'evrp-benchmark-set/'
files = [
    "E-n23-k3.evrp",
    "E-n51-k5.evrp"
]
seeds = list(range(0, 20))

results_files = []


for file in files:
    print(file)
    solutions = []
    routes = []
    constrains = []
    for seed in seeds:
        print(seed)
        instance = EVRP(root+file, seed=seed)
        max_evals = 25000 * int(instance.n_dimension + instance.n_stations)
        ag = AGPermutation(
            instance=instance,
            max_evals=max_evals,
            n_pop=n_pop,
            crossover_prob=prob_crossover,
            mutation_prob=prob_mutation,
            seed=seed
        )

        sol, value = ag.run()
        print(sol)
        print(value)
        print(instance.constraints(sol, verbose=True))
        print(instance.fun_obj(sol))
        ag.save_outputs(f'{file}_{seed}')
        solutions.append(value)
        routes.append(sol)
        constrains.append(instance.constraints(sol))

    results_files.append({'solutions': solutions, 'routes': routes, 'constraints': constrains})
    pd.DataFrame(
        {'solutions': solutions, 'routes': routes, 'constraints': constrains}
    ).to_csv(f'tabelas/{file}_{max_evals}.csv')

print(pd.DataFrame(results_files[0]))
print(pd.DataFrame(results_files[1]))
