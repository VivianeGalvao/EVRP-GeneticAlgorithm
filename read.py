# author: Viviane de Jesus Galvao

def read_node_coords(lines, id_start, n_nodes):

    output = {}
    for i in range(id_start+1, id_start+n_nodes+1, 1):
        aux = lines[i].split(' ')
        output[int(aux[0])] = (float(aux[1]), float(aux[2]))
    
    return output


def read_demand_sections(lines, id_start, dimension):
    output = {}
    for i in range(id_start+1, id_start+dimension+1, 1):
        aux = lines[i].split(' ')
        output[int(aux[0])] = float(aux[1].replace('\n', ''))

    return output


def read_stations_coord(lines, id_start, n_nodes, n_stations):
    output = {}
    for i in range(1, n_nodes+1, 1):
        output[i] = 0

    for i in range(id_start+1, id_start+n_stations+1, 1):
        output[int(lines[i])] = 1

    return output
        

def read_problem(filename):

    output = {}
    tags = [
        'VEHICLES:',
        'DIMENSION:', 
        'STATIONS:',
        'ENERGY_CAPACITY:',
        'ENERGY_CONSUMPTION:',
        'CAPACITY:'
    ]

    with open(filename, "r") as f:
        lines = f.readlines()
        n_lines = len(lines)
        i = 0
        while i < n_lines:
            line = lines[i]
            for tag in tags:
                if tag in line:
                    output[tag.lower().replace(':', '')] = float(line.split(':')[-1].replace('\n', '').strip())
                    break

            if 'NODE_COORD_SECTION' in line:
                n_nodes = int(output['dimension']) + int(output['stations'])
                output['node_coord_section'] = read_node_coords(lines, i, n_nodes)
                i = i + n_nodes -1
            if 'DEMAND_SECTION' in line:
                output['demand_section'] = read_demand_sections(
                    lines, i, int(output['dimension'])
                )
                i = i + int(output['dimension']) -1
            if 'STATIONS_COORD_SECTION' in line:
                n_nodes = int(output['dimension']) + int(output['stations'])
                output['stations_coord_section'] = read_stations_coord(
                    lines, i, n_nodes, int(output['stations'])
                )
                i = i + int(output['stations']) -1
            if 'DEPOT_SECTION' in line:
                i+=1
                output['depot_section'] = int(lines[i])
            
            i+=1

    return output
