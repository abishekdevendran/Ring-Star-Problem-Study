import os
import numpy as np
from collections import defaultdict
from pygad import GA
import matplotlib.pyplot as plt


def tspLibrary(dir: str):
    """Reads the tsp files in the directory and returns a list[list[floats]]"""
    with open(dir) as f:
        lines = f.readlines()
    data = [x.strip().split() for x in lines]
    # cleanup the data where count of elements is not equal to 3
    data = [x for x in data if len(x) == 3]
    # skip the lines where the first element is not a number
    while not data[0][0].strip().isnumeric():
        data = data[1:]
    data = [[float(x) for x in y] for y in data]
    # drop the first element of all the lists
    data = [x[1:] for x in data]
    return data


def loadAllTSPs(directory: str):
    """Loads all tsp files in the directory"""
    files = os.listdir(directory)
    files = [x for x in files if x.endswith('.tsp')]
    data = []
    for file in files:
        try:
            data.append(tspLibrary(directory + file))
        except Exception as e:
            # print(f"Error in reading file {file}")
            # print(e)
            pass
    return data


def calcDist_EUC(nodes: list[tuple]):
    """
    Calculates the Euclidean distance matrix between a set of nodes.

    Args:
                    nodes: A list of Node objects.

    Returns:
                    A NumPy array representing the distance matrix.
    """

    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i][j] = np.sqrt(
                (nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)

    return distance_matrix


def TSP_NN(distance_matrix: np.ndarray, hub_indices: list[int]) -> list[int]:
    """
    Solves the TSP to form an approximate ring connecting the selected hubs using the nearest neighbor algorithm.

    Args:
        distance_matrix: A NumPy array representing distances between each pair of nodes.
        hub_indices: A list of indices corresponding to the selected hub nodes.

    Returns:
        A list of hub indices (in the original node list) in the order they should be visited to form the approximate ring.
    """

    # Start with the first hub
    current_hub = hub_indices[0]

    # Initialize the list of hubs in the order they will be visited
    ordered_hubs = [current_hub]
    remaining_hubs = hub_indices[1:]

    while remaining_hubs:
        next_hub = remaining_hubs[np.argmin(
            distance_matrix[current_hub, remaining_hubs])]
        ordered_hubs.append(next_hub)
        remaining_hubs.remove(next_hub)
        current_hub = next_hub

    return ordered_hubs


def minimum_spanning_tree(distance_matrix: np.ndarray, hub_indices: list[int]) -> list[tuple]:
    """
    Calculates the minimum spanning tree of a graph using Prim's algorithm.

    Args:
        distance_matrix: A NumPy array representing distances between each pair of nodes.
        hub_indices: A list of indices corresponding to the selected hub nodes.

    Returns:
        A list of tuples representing the edges of the minimum spanning tree.
    """
    # find MST for nodes in hub_indices
    num_nodes = len(hub_indices)
    mst = []
    visited = [False] * num_nodes
    visited[0] = True

    while len(mst) < num_nodes - 1:
        min_edge = None
        min_distance = float('inf')

        for i in range(num_nodes):
            if visited[i]:
                for j in range(num_nodes):
                    if not visited[j] and distance_matrix[hub_indices[i]][hub_indices[j]] < min_distance:
                        min_edge = (hub_indices[i], hub_indices[j])
                        min_distance = distance_matrix[hub_indices[i]
                                                       ][hub_indices[j]]

        mst.append(min_edge)
        visited[hub_indices.index(min_edge[1])] = True

    return mst


def MST_viz(nodes: list[tuple], distance_matrix: np.ndarray, hub_indices: list[int], mst: list[tuple]):
    # label all points in red or blue based on hub_indices
    for i, point in enumerate(nodes):
        if i in hub_indices:
            plt.scatter(point[0], point[1], color='red')
        else:
            plt.scatter(point[0], point[1], color='blue')

    # draw edges in mst
    for edge in mst:
        plt.plot([nodes[edge[0]][0], nodes[edge[1]][0]],
                 [nodes[edge[0]][1], nodes[edge[1]][1]], color='green')

    # title the plot as "MST for Christofides"
    plt.title("MST for Christofides")
    plt.grid()
    plt.show()


def TSP_Christofides(nodes: list[tuple], distance_matrix: np.ndarray, hub_indices: list[int], mst_viz=False) -> list[int]:
    """
    Solves the TSP to form an approximate ring connecting the selected hubs using the Christofides algorithm.

    Args:
        distance_matrix: A NumPy array representing distances between each pair of nodes.
        hub_indices: A list of indices corresponding to the selected hub nodes.

    Returns:
        A list of hub indices (in the original node list) in the order they should be visited to form the approximate ring.
    """
    # calculate the minimum spanning tree
    mst = minimum_spanning_tree(distance_matrix, hub_indices)
    if mst_viz:
        # print("MST: ", mst)
        MST_viz(nodes, distance_matrix, hub_indices, mst)

    # find the vertices with odd degree in the minimum spanning tree
    degree = defaultdict(int)
    for edge in mst:
        degree[edge[0]] += 1
        degree[edge[1]] += 1

    odd_vertices = [v for v in degree if degree[v] % 2 == 1]

    # form the subgraph induced by the odd vertices
    subgraph = np.zeros((len(odd_vertices), len(odd_vertices)))
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            subgraph[i][j] = distance_matrix[u][v]

    # solve the minimum weight matching problem on the subgraph
    matching = []
    while len(matching) < len(odd_vertices) // 2:
        min_edge = np.unravel_index(
            np.argmin(subgraph), subgraph.shape)
        matching.append((odd_vertices[min_edge[0]],
                         odd_vertices[min_edge[1]]))
        subgraph[min_edge[0]] = float('inf')
        subgraph[:, min_edge[1]] = float('inf')

    # add the matching edges to the minimum spanning tree
    for edge in matching:
        mst.append(edge)

    # find the Eulerian tour of the augmented graph
    eulerian_tour = []
    for edge in mst:
        eulerian_tour.append(edge[0])
        eulerian_tour.append(edge[1])

    # remove duplicate vertices
    visited = set()
    eulerian_tour = [v for v in eulerian_tour if not (
        v in visited or visited.add(v))]
    eulerian_tour.append(eulerian_tour[0])

    # construct the final tour by skipping already visited vertices
    final_tour = []
    for v in eulerian_tour:
        if v in final_tour:
            continue
        final_tour.append(v)

    return final_tour


def nodeAlloc_NN(nodes: list[tuple], hub_indices: list[int], distance_matrix: np.ndarray) -> dict:
    """
    Assigns non-hub nodes to their nearest hub using a nearest neighbor approach (hubs represented by indices).

    Args:
                    nodes: A list of all Node objects.
                    hub_indices:  A list of indices corresponding to hub nodes.
                    distance_matrix: A NumPy array representing distances between each pair of nodes.

    Returns:
                    A dictionary of assignments: {hub_index: [list of assigned non-hub nodes]}
    """
    assignments = {}
    # print("Hub Indices: ", hub_indices)
    for node_index, node in enumerate(nodes):
        if node_index in hub_indices:
            continue  # Skip if the node itself is a hub
        distances_to_hubs = [distance_matrix[node_index]
                             [hub_index] for hub_index in hub_indices]
        nearest_hub_index = hub_indices[np.argmin(distances_to_hubs)]
        if nearest_hub_index not in assignments:
            assignments[nearest_hub_index] = [node_index]
        else:
            assignments[nearest_hub_index].append(node_index)

    return assignments


def GA_cost(data: list[tuple], mask: list[int], distance_matrix: np.ndarray, HUB_EDGE_COST=2, NON_HUB_EDGE_COST=1, tsp_heuristic: str = "NN", print_ordered_hubs=False) -> float:
    """
    Calculates the fitness of a given mask, which represents a potential solution to the problem.

    Args:
                    mask: A list of integers representing the order in which the nodes will be visited.

    Returns:
                    The fitness of the mask.
    """
    # Create a list of hub indices and non-hub indices based on the mask
    hub_indices = [i for i, x in enumerate(mask) if x == 1]
    # non_hub_indices = [i for i, x in enumerate(mask) if x == 0]

    # Allocate non-hub nodes to the nearest hub
    assignments = nodeAlloc_NN(
        data, hub_indices, distance_matrix)

    # print("Assignments: ", assignments)

    # Solve the TSP to form an approximate ring connecting the selected hubs
    ordered_hubs = TSP_NN(
        distance_matrix, hub_indices) if tsp_heuristic == "NN" else TSP_Christofides(data, distance_matrix, hub_indices)
    if print_ordered_hubs:
        print("Ordered Hubs: ", ordered_hubs)

    # Calculate the total cost of the solution
    total_cost: float = 0
    hub_cost: float = 0
    non_hub_cost: float = 0
    # Add the cost of the edges between hubs
    for i in range(len(ordered_hubs)):
        hub_cost += distance_matrix[ordered_hubs[i]
                                      ][ordered_hubs[(i+1) % len(ordered_hubs)]] * HUB_EDGE_COST

    # Add the cost of the edges between non-hub nodes and their assigned hubs
    for hub_index, assigned_nodes in assignments.items():
        for node_index in assigned_nodes:
            non_hub_cost += distance_matrix[hub_index][node_index] * \
                NON_HUB_EDGE_COST
            
    total_cost = hub_cost + non_hub_cost
    # print("Hub Cost: ", hub_cost)
    # print("Non-Hub Cost: ", non_hub_cost)
    # print("Total Cost: ", total_cost)
    return total_cost


def create_GA_fitness(data: list[tuple], dist_matrix: np.ndarray, HUB_EDGE_COST=100, NON_HUB_EDGE_COST=1,tsp_heuristic="NN"):
    """
    Calculates the fitness of a given solution, which represents a potential solution to the problem.

    Args:
        data: A list of tuples representing the coordinates of each node.
        solution: A list of 0/1 values representing the selection of hub nodes.
        dist_matrix: A NumPy array representing distances between each pair of nodes.

    Returns:
        A function that calculates the fitness of the solution.
    """
    def GA_fitness(ga_instance: GA, solution: list[int], solution_idx: int) -> float:
        # Since the problem is a minimization problem, we need to return the cost as a negative value
        return 100/GA_cost(data, solution, dist_matrix, HUB_EDGE_COST, NON_HUB_EDGE_COST, tsp_heuristic)

    return GA_fitness


def ansViz(data: list[tuple], mask, dist_matrix: np.ndarray, tsp_heuristic: str = "NN"):
    tmp1 = TSP_NN(
        dist_matrix, [i for i, x in enumerate(mask) if x == 1]) if tsp_heuristic == "NN" else TSP_Christofides(data, dist_matrix, [i for i, x in enumerate(mask) if x == 1], mst_viz=True)

    # label each point by its index
    for i, point in enumerate(data):
        if mask[i] == 1:
            plt.scatter(point[0], point[1], color='red')
        else:
            plt.scatter(point[0], point[1], color='blue')
        # plt.text(point[0], point[1], str(i))

    # draw the ring
    for i in range(len(tmp1)):
        plt.plot([data[tmp1[i]][0], data[tmp1[(i+1) % len(tmp1)]][0]],
                 [data[tmp1[i]][1], data[tmp1[(i+1) % len(tmp1)]][1]], color='green')

    # tmp2 = nodeAlloc_NN(
    #     data, [i for i, x in enumerate(mask) if x == 1], dist_matrix)

    # for hub_index, assigned_nodes in tmp2.items():
    #     for node_index in assigned_nodes:
    #         plt.plot([data[hub_index][0], data[node_index][0]], [
    #                  data[hub_index][1], data[node_index][1]], color='black')

    # title the plot as "Final ANswer visualization"
    plt.title("Final Answer visualization")
    plt.grid()
    plt.show()


def clustering(data: list[tuple], dist_matrix: np.ndarray, number_of_clusters: int = 5, tsp_heuristic: str = "NN", HUB_EDGE_COST=100, NON_HUB_EDGE_COST=1, show_mst=False):
    """
    Clusters the nodes in the graph into a given number of clusters using the K-Means algorithm.

    Args:
        data: A list of tuples representing the coordinates of each node.
        number_of_clusters: The number of clusters to form.

    Returns:
        A list of lists, where each inner list contains the indices of the nodes in a cluster.
    """
    # K means clustering with non-empty clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(data)
    # find node indices closest to each cluster center
    cluster_centers = kmeans.cluster_centers_
    clusters = [[] for _ in range(number_of_clusters)]
    for i, node in enumerate(data):
        clusters[kmeans.labels_[i]].append(i)
    # for each cluster, find the node closest to the cluster center
    hubs = []
    for cluster in clusters:
        min_dist = float('inf')
        hub = None
        for node in cluster:
            dist = np.linalg.norm(
                np.array(data[node]) - cluster_centers[clusters.index(cluster)])
            if dist < min_dist:
                min_dist = dist
                hub = node
        hubs.append(hub)
    # print("Hubs: ", hubs)


    # find order of hubs using NN
    ordered_hubs = TSP_NN(dist_matrix, hubs) if tsp_heuristic == "NN" else TSP_Christofides(
        data, dist_matrix, hubs, show_mst)
    
    # alloc non-hub nodes to nearest hub
    assignments = nodeAlloc_NN(data, ordered_hubs, dist_matrix)

    # find cost
    total_cost:float = 0
    hub_cost:float = 0
    non_hub_cost:float = 0
    # hub costs
    for i in range(len(ordered_hubs)):
        hub_cost += dist_matrix[ordered_hubs[i]
                                  ][ordered_hubs[(i+1) % len(ordered_hubs)]] * HUB_EDGE_COST
    # distance of each node to its hub
    for hub_index, assigned_nodes in assignments.items():
        for node_index in assigned_nodes:
            non_hub_cost += dist_matrix[hub_index][node_index] * NON_HUB_EDGE_COST

    total_cost = hub_cost + non_hub_cost
    # print("Hub Cost: ", hub_cost)
    # print("Non-Hub Cost: ", non_hub_cost)
    # print("Total Cost: ", total_cost)

    return (clusters, ordered_hubs, total_cost)


def create_clustering_fitness(data: list[tuple], dist_matrix: np.ndarray, tsp_heuristic="NN"):
    """
    Calculates the fitness of a given solution, which represents a potential solution to the problem.

    Args:
        data: A list of tuples representing the coordinates of each node.
        solution: A list of 0/1 values representing the selection of hub nodes.
        dist_matrix: A NumPy array representing distances between each pair of nodes.

    Returns:
        A function that calculates the fitness of the solution.
    """
    def clustering_fitness(ga_instance: GA, solution: list[int], solution_idx: int) -> float:
        # print("Testing solution: ", solution)
        # Since the problem is a minimization problem, we need to return the cost as a negative value
        return 100/clustering(data, dist_matrix, solution[0], tsp_heuristic, 2, 1)[2]

    return clustering_fitness
