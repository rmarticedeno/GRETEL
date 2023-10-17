import networkx as nx
import numpy as np

from src.n_dataset.manipulators.base import BaseManipulator


class Diameter(BaseManipulator):
    def graph_info(self, instance):
        graph = instance._build_nx()

        diameter = 0
        if nx.is_connected(graph):
            diameter = nx.diameter(graph)
        else:
            components = nx.connected_components(graph)
            largest_component = max(components, key=len)
            subgraph = graph.subgraph(largest_component)
            diameter = nx.diameter(subgraph)
        
        feature_map = {
            "diameter": list(diameter)
        }
        return feature_map
    