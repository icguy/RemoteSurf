import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils
import cProfile

def extract_full(graph, max_num):
    triplets = []
    for k in graph:
        neigh = list(graph[k])
        num_n = len(neigh)
        for i in range(num_n):
            for j in range(i + 1, num_n):
                n1 = neigh[i]
                n2 = neigh[j]
                if k in graph[n1] and k in graph[n2]:
                    triplets.append((k, n1, n2))
    num = 3
    cliques = triplets
    all_levels = [triplets]
    while num < max_num:
        cliques = add_level(graph, cliques)
        all_levels.append(list(cliques))
        num += 1
    return all_levels

def add_level(graph, cliques):
    newcliques = set()
    for c in cliques:
        elem = c[0]
        for n in graph[elem]:
            if connected_to_all(graph, n, c):
                newclique = list(c)
                newclique.append(n)
                newcliques.add(tuple(sorted(newclique)))
    return newcliques

def connected_to_all(graph, node, clique):
    for n in clique:
        if node not in graph[n]:
            return False
    return True

if __name__ == '__main__':
    a = (1, 2, 3)
    b = (4, 5, 6)
    c = [aa - bb for aa, bb in zip(a, b)]
    print c