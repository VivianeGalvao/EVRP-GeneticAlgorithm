# author: Viviane de Jesus Galvao

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AGPermutation():

    def __init__(
            self,
            instance,
            max_evals,
            n_pop,
            crossover_prob,
            mutation_prob,
            seed=42
        ) -> None:
        
        random.seed(seed)
        self.max_evals = max_evals
        self.n_pop = n_pop
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.problem = instance

    def run(self):
        print(f'MAX EVALS: {self.max_evals}')

        self.output_bests = []
        self.mean_pop = []
        self.new_pop_rate = 0.9
        self.equals_rate = 0

        self.generate_population()
        self.evaluate_individuals()

        self.best_value = min(self.fitness)
        self.best_solution = self.pop[self.fitness.index(self.best_value)]

        evals = self.problem.get_evals()
        self.steps = 0        
        while evals < self.max_evals:
            self.output_bests.append(self.best_value)
            self.mean_pop.append(np.mean(self.fitness))
            new_pop = []
            while len(new_pop) < (int(self.new_pop_rate*self.n_pop)):
                c1, c2 = self.crossover()
                if (c1 is not None) & (c2 is not None):
                    c3 = self.mutation(c1)
                    c4 = self.mutation(c2)
                    if c3:
                        new_pop.append(c3)
                    if c4:
                        new_pop.append(c4)
            
            if (self.steps+1)%10==0:
                c1 = self.mutation(self.best_solution)
                if c1:
                    new_pop.append(c1)                
            
            self.replace_population(new_pop)
            self.update_best_solution()
    
            self.steps+=1

            evals = self.problem.get_evals()

        return self.best_solution, self.best_value
    
    def plot_best_values(self, namefig):
        plt.yscale("log")
        plt.plot(self.output_bests, label='Valor Melhor Individuo')
        plt.plot(self.mean_pop, label='Média do fitness da população')
        plt.legend()
        plt.savefig(namefig)
        plt.close()

    def save_outputs(self, name_problem):
        df_mean = pd.DataFrame({'media_fitness': self.mean_pop})
        df_mean.to_csv(f'media_fitness_pop_{name_problem}.csv')

        df_best = pd.DataFrame({'valor_best': self.output_bests})
        df_best.to_csv(f'valor_best_geracoes_{name_problem}.csv')

    def verify_ind_equals(self):
        rate=0
        i = 0
        while i < self.n_pop-1:
            j = i+1
            flag = 0
            while j < self.n_pop:
                if self.pop[i] == self.pop[j]:
                    flag+=1
                j+=1
            if flag > 0:
                rate +=1
            i+=1

        self.equals_rate = rate/self.n_pop

        print(len(self.pop), rate/self.n_pop)

    def __permutation_crossover__(self, p1, p2):

        size = len(p1) if len(p1) < len(p2) else len(p2)
        i1 = random.randint(0, size)
        i2 = random.randint(0, size)

        if i1 < i2:
            i = i1
            j = i2
        else:
            i = i2
            j = i1

        new_solution = p1[i:j]
        k=j
        while (k < len(p2)):
            if self.problem.is_customer(p2[k]-1):
                if p2[k] not in new_solution:
                    new_solution.append(p2[k])
            else:
                new_solution.append(p2[k])
            k+=1
        
        k=0
        b_solution = []
        while (k < i):
            if self.problem.is_customer(p2[k]-1):
                if p2[k] not in new_solution:
                    b_solution.append(p2[k])
            else:
                b_solution.append(p2[k])
            k+=1

        new_solution = b_solution + new_solution

        if ~(self.problem.all_customers_constraint(new_solution)):
            p1_new = [x for x in p1 if (x not in new_solution) & (self.problem.is_customer(x-1))]
            p2_new = [x for x in p2 if (x not in new_solution) & (self.problem.is_customer(x-1))]
            p1xp2 = list(set(p1_new + p2_new))
            i=0
            while i < len(p1xp2):
                if p1xp2[i] in p2:
                    index = p2.index(p1xp2[i])  
                else:
                    index = p1.index(p1xp2[i])
                new_solution = new_solution[:index] + [p1xp2[i]] + new_solution[index:]
                i+=1
        
        if new_solution[-1] != self.problem.depot_station:
            new_solution.append(self.problem.depot_station)
        if new_solution[0] != self.problem.depot_station:
            new_solution = [self.problem.depot_station] + new_solution

        assert (self.problem.all_customers_constraint(new_solution))

        return new_solution

    def crossover(self):
        p1 = self.selection()
        p2 = self.selection()

        while self.pop[p1] == self.pop[p2]:
            p2 = self.selection()

        p1 = self.pop[p1]
        p2 = self.pop[p2]

        r = random.random()
        if r < self.crossover_prob:
            f1 = self.__permutation_crossover__(p1, p2)
            f2 = self.__permutation_crossover__(p2, p1)
            return f1, f2

        return None, None
        
    def __insertion_mutation__(self, candidate):
    
        assert (self.problem.all_customers_constraint(candidate))
        chosen = random.sample(list(range(len(candidate))), 2)

        i = chosen[0]
        j = chosen[1]

        if i < j:
            new_candidate = candidate[:i+1] + [candidate[j]] + candidate[i+1:j] + candidate[j+1:]
        else:
            new_candidate = candidate[:j+1] + [candidate[i]] + candidate[j+1:i] + candidate[i+1:]

        if new_candidate[-1] != self.problem.depot_station:
            new_candidate.append(self.problem.depot_station)
        if new_candidate[0] != self.problem.depot_station:
            new_candidate = [self.problem.depot_station] + new_candidate

        assert (self.problem.all_customers_constraint(new_candidate))
        return new_candidate
    
    def __mixture_mutation(self, candidate):
        return None
    
    def __inversion_mutation(self, candidate):
        return None

    def __change_mutation__(self, candidate):
        new_candidate = candidate.copy()

        chosen = random.sample(list(range(len(candidate))), 2)

        i = chosen[0]
        j = chosen[1]

        aux = new_candidate[i]
        new_candidate[i] = new_candidate[j]
        new_candidate[j] = aux        

        if new_candidate[-1] != self.problem.depot_station:
            new_candidate.append(self.problem.depot_station)
        if new_candidate[0] != self.problem.depot_station:
            new_candidate = [self.problem.depot_station] + new_candidate

        assert (self.problem.all_customers_constraint(new_candidate))
        return new_candidate
    
    def mutation(self, c):
        c1 = c.copy()
        flag=0

        r = random.random()
        if r < (self.mutation_prob):
            flag=1
            c1 = self.__insertion_mutation__(c1)
            c1 = self.__change_mutation__(c1)
            return c1

        return None if flag==0 else c1
    
    def __fitness_selection__(self):
        return None
    
    def __classification_selection__(self):
        return None
    
    def __exp_selection__(self):
        return None
    
    def __tournament_selection__(self, k=3):
        chosen = random.sample(list(range(len(self.pop))), k)
        values = [self.fitness[i] for i in chosen]
        return chosen[np.argmin(values)]
    
    def selection(self):
        return self.__tournament_selection__()
    
    def replace_population(self, new_pop):
        
        new_fitness = [self.problem.fitness(x) for x in new_pop]

        df_new_pop = pd.DataFrame(
            {
                'ind': list(range(len(new_pop))),
                'fitness': new_fitness, 
                'solution': new_pop
                
            }
        )

        df_actual_pop = pd.DataFrame(
            {
                'ind': list(range(len(self.pop))),
                'fitness': self.fitness,
                'solution': self.pop
                
            }
        )

        df_actual_pop = df_actual_pop[~(df_actual_pop['solution'].isin(new_pop))]
        #v02
        df_new_pop['size'] = df_new_pop['solution'].apply(lambda x: len(x))
        df_actual_pop['size'] = df_actual_pop['solution'].apply(lambda x: len(x))
        #v02
        df_new_pop = df_new_pop.sort_values(['fitness', 'size'])
        df_actual_pop = df_actual_pop.sort_values(['fitness', 'size'])

        number_old = int(round((1.0-self.new_pop_rate)*self.n_pop+0.5, 1))
        number_new = (int(self.new_pop_rate*self.n_pop))           

        actual_ind = df_actual_pop['ind'].values[:number_old]
        new_ind = df_new_pop['ind'].values[:number_new]

        self.pop = [self.pop[i] for i in actual_ind] + [new_pop[i] for i in new_ind]
        self.fitness = [self.fitness[i] for i in actual_ind] + [new_fitness[i] for i in new_ind]

        assert len(self.pop) == self.n_pop
    
    def update_best_solution(self):
        for i in range(len(self.pop)):
            if self.fitness[i] < self.best_value:
                self.best_solution = self.pop[i]
                self.best_value = self.fitness[i]
                self.steps=0

        assert (self.problem.all_customers_constraint(self.best_solution))
    
    def evaluate_individuals(self):
        self.fitness = [self.problem.fitness(x) for x in self.pop]
    
    def generate_population(self):
        self.pop = [self.problem.get_solution() for i in range(self.n_pop)]
