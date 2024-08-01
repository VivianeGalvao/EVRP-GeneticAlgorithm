# author: Viviane de Jesus Galvao

import math
import random

from read import read_problem


def euclidean_distance(p1, p2):

    x = p1[0] - p2[0]
    y = p1[1] - p2[1]

    return math.sqrt(x**2 + y**2)


def calc_distances(nodes: dict):
    dists = []
    n_nodes = len(nodes)

    for i in range(n_nodes):
        aux = []
        for j in range(n_nodes):
            aux.append(euclidean_distance(nodes[i+1], nodes[j+1]))

        dists.append(aux)

    return dists


class EVRP():

    def __init__(self, filename: str, seed=42) -> None:

        desc = read_problem(filename)
        self.evals = 0
        self.n_vehicles = desc['vehicles']
        self.n_dimension = desc['dimension']
        self.n_stations = desc['stations']
        self.max_capacity = desc['capacity']
        self.max_energy = desc['energy_capacity']
        self.energy_consumption = desc['energy_consumption']
        self.demand_nodes = list(desc['demand_section'].values()) + [0]*int(self.n_stations)
        self.charging_stations = list(desc['stations_coord_section'].values())
        self.depot_station = desc['depot_section']
        self.dist_matrix = calc_distances(desc['node_coord_section'])

        self.set_seed(seed)

    def set_seed(self, seed):
        random.seed(seed)

    def custumer_demand_constraint(self, current_capacity, y):
    
        capacity_temp = current_capacity - self.__get_customer_demand__(y)
        if capacity_temp < 0:
            return True
        return False
        
    def energy_consumption_constraint(self, current_energy, x, y):
        capacity_temp = current_energy - self.__get_energy_consumption__(x, y)
        if capacity_temp < 0:
            return True
        
        return False
        
    def same_node_constraint(self, x, y):
        return self.dist_matrix[x][y] == 0

    def all_customers_constraint(self, solution):
        customers = 0

        for sol in solution:
            if self.is_customer(sol-1):
                customers += 1
        return customers == int(self.n_dimension-1)

    def repeated_customers_constraint(self, solution):
        customers = 0

        for sol in solution:
            if self.is_customer(sol-1):
                customers += 1

        return customers > int(self.n_dimension-1)
        
    def constraints(self, solution: list, verbose=False):

        energy_temp = self.max_energy
        capacity_temp = self.max_capacity
        
        size = len(solution)
        p = 0
        i = 0
        while i < (size-1):
            x = solution[i] - 1
            y = solution[i+1] -1

            capacity_temp -= self.__get_customer_demand__(y)
            energy_temp -= self.__get_energy_consumption__(x, y)

            if capacity_temp < 0:
                if verbose:
                    print(f'Error: capacity bellow 0 at customer {x+1}')
                p+=(self.max_capacity)**3

            if energy_temp < 0:
                if verbose:
                    print(f'Error: energy bellow 0 from {x+1} to {y+1}')
                p+=(self.max_energy)**3
            
            if self.same_node_constraint(x, y):
                if verbose:
                    print(f'Error: x:{x+1} == y:{y+1}')
                p+= 100

            if not self.all_customers_constraint(solution):
                if verbose:
                    print('Error: customers not covered in solution')
                customers = 0
                for sol in solution:
                    if self.is_customer(sol-1):
                        customers += 1
                k = (self.n_dimension-1) - customers
                p+= 1000**k

            if self.repeated_customers_constraint(solution):
                if verbose:
                    print('Error: customers not covered in solution')
                customers = 0
                for sol in solution:
                    if self.is_customer(sol-1):
                        customers += 1
                k = customers - (self.n_dimension-1)
                p+= 1000**k
            
            if y+1 == self.depot_station:
                capacity_temp = self.max_capacity

            if (self.charging_stations[y]==1) | (y+1 == self.depot_station):
                energy_temp = self.max_energy

            i+=1

        return p
    
    def fun_obj(self, solution: list):

        tour_length = 0.0
        n_size = len(solution)
        
        i=0
        while i < (n_size-1):
            x = solution[i]-1
            y = solution[i+1]-1
            tour_length += self.dist_matrix[x][y]
            i+=1

        self.evals += 1

        return tour_length
    
    def fitness(self, solution:list):

        p = self.constraints(solution)
        f = self.fun_obj(solution)
        k = 1

        return f+k*p
    
    def __get_energy_consumption__(self, x: int, y: int):
        
        return self.energy_consumption*self.dist_matrix[x][y]
    
    def __get_customer_demand__(self, customer: int):

        if self.charging_stations[customer]:
            return 0
        
        return self.demand_nodes[customer]
    
    def get_evals(self):
        return self.evals

    def get_problem_size(self):
        return int(self.n_dimension + self.n_stations*2*self.n_dimension + 2*self.n_dimension)
    
    def is_customer(self, x: int):
        return ~(self.charging_stations[x]) & (x+1 != self.depot_station)
    
    def get_solution(self):
        customers = [i+1 for i in range(len(self.charging_stations)) if self.is_customer(i)]
        possibles = [i+1 for i in range(len(self.charging_stations)) if self.charging_stations[i]] + [self.depot_station]

        solution = [self.depot_station] 
        random.shuffle(customers)
        solution = solution + customers
        sol_size = random.randint(len(possibles), 2*len(possibles))
        i = 0
        while i < sol_size:
            n = random.randint(1, len(solution)-1)
            station = random.sample(possibles, 1)
            solution = solution[:n] + station + solution[n:]         
            i+=1

        solution.append(self.depot_station)

        return solution
    
    def remove_charging_stations(self, solution):
        new_solution = solution.copy()
        ids_chargers = [solution[i] for i in range(len(solution)) if self.charging_stations[solution[i]-1]]

        if len(ids_chargers) == 0:
            return new_solution
        id = random.choice(ids_chargers)

        new_solution.remove(id)

        return new_solution
    
    def add_charging_stations(self, solution):
        new_solution = solution.copy()
        stations = [i+1 for i in range(len(self.charging_stations)) if self.charging_stations[i]]
        n_add = random.randint(0, len(stations))
        if n_add == 0:
            return solution
        station = random.sample(stations, n_add)

        for s in station:
            id_add = random.randint(1, len(solution))
            new_solution = new_solution[:id_add] + [s] + new_solution[id_add:]

        return new_solution
    
    def remove_depots(self, solution):
        new_solution = solution.copy()
        id_depots = [i for i in range(len(solution)) if solution[i] == self.depot_station]

        id_depots.remove(0)
        id_depots.remove(len(solution)-1)

        if len(id_depots) == 0:
            return solution

        n = random.choice(id_depots)
        new_solution = new_solution[:n] + new_solution[n+1:]
        
        assert self.all_customers_constraint(solution)
        assert self.all_customers_constraint(new_solution)

        return new_solution

    def add_depot(self, solution):
        new_solution = solution.copy()

        id_add = random.randint(1, len(solution))
        new_solution = new_solution[:id_add] + [self.depot_station] + new_solution[id_add:]

        return new_solution
