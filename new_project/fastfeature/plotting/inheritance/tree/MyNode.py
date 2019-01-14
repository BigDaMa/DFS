from graphviz import Digraph
from typing import List
from fastfeature.candidates.CandidateFeature import CandidateFeature
import matplotlib.pyplot as plt
import networkx as nx
import time

class MyNode(object):
    def __init__(self, name: int, candidate: CandidateFeature, score: float, children=None):
        self.name = name
        self.candidate = candidate
        self.score = score
        self.children: List[MyNode] = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, MyNode)
        self.children.append(node)

    def get_color(self,a,b):
        if a > b:
            color = 'red'
        elif a < b:
            color = 'green'
        else:
            color = 'blue'
        return color


    def get_color2(self,a,b):
        if a >= b:
            color = '#ff0000'#'red'
        else:
            color = '#00ff00'#'green'
        return color

    def get_color3(self,a,b):
        return 'black'


    def color_tuple_to_hex_str(self, color_tuple):
        my_str = '#'
        for i in range(3):
            my_str += hex(int(color_tuple[i] * 255.0))[-2:]
        assert len(my_str) == 7, 'color length is wrong'
        return my_str

    def color_tuple_to_str(self, color_tuple):
        my_str = ''
        for i in range(3):
            my_str += str(color_tuple[i]) + ' '
        my_str = my_str[:-1]
        return my_str

    def get_color_new(self, score, starter_score, max_score):
        Blues = plt.get_cmap('Blues')
        if score >= starter_score:
            normed_value = (score - starter_score) / (max_score - starter_score)
            color_tuple = Blues(normed_value)
            str_hex = self.color_tuple_to_str(color_tuple)
            print(str_hex)
            return str_hex
        else:
            return '#ff0000'

    def get_color_tuple(self, score, starter_score, max_score):
        Blues = plt.get_cmap('Blues')
        if score >= starter_score:
            normed_value = (score - starter_score) / (max_score - starter_score)
            color_tuple = Blues(normed_value)
            return color_tuple
        else:
            return (0,0,0)




    def create_graph(self, graph=Digraph(comment='Candidate'), start_score=-1, max_score=2):
        node_text: str = ''
        if self.candidate == None:
            node_text = 'root'
        else:
            node_text = self.candidate.get_name()

        node_color = self.get_color_new(self.score, start_score, max_score)

        size = -1.0
        if self.score > start_score:
            normed_value = (self.score - start_score) / (max_score - start_score)
            size = str(100 * normed_value)
        else:
            size = str(1)

        graph.node(str(self.name), node_text, color=node_color, score=str(self.score), fixedsize='true', width=str(size), height=str(size))
        for child_i in range(len(self.children)):
            self.children[child_i].create_graph(graph, start_score=start_score, max_score=max_score)
            if node_text != 'root':
                graph.edge(str(self.name), str(self.children[child_i].name), color='#ffffff')
        return graph

    def create_graph_graphml(self, graph: nx.Graph, start_score=-1, max_score=2):
        node_text: str = ''
        if self.candidate == None:
            node_text = 'root'
        else:
            node_text = self.candidate.get_name()

        print("Node:" + str(node_text) + ": " +  str(self.name))

        size = -1.0
        if self.score > start_score:
            normed_value = (self.score - start_score) / (max_score - start_score)
            size = str(100 * normed_value)
        else:
            size = str(1)

        nodeid = str(self.name)
        graph.add_node(nodeid)
        graph.node[nodeid]['label'] = node_text
        graph.node[nodeid]['size'] = size
        graph.node[nodeid]['score'] = str(self.score)
        graph.node[nodeid]['r'] = int(self.get_color_tuple(self.score, start_score, max_score)[0] * 256)
        graph.node[nodeid]['g'] = int(self.get_color_tuple(self.score, start_score, max_score)[1] * 256)
        graph.node[nodeid]['b'] = int(self.get_color_tuple(self.score, start_score, max_score)[2] * 256)

        for child_i in range(len(self.children)):
            self.children[child_i].create_graph_graphml(graph, start_score=start_score, max_score=max_score)
            if node_text != 'root':
                edge = graph.add_edge(str(self.name), str(self.children[child_i].name))
                print("edge: " + str(self.name) + " -> " + str(self.children[child_i].name))
        return graph