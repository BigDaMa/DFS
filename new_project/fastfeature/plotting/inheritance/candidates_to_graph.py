import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fastfeature.candidates.RawFeature import RawFeature
from fastfeature.candidates.CandidateFeature import CandidateFeature


def get_color_tuple(score, starter_score, max_score):
    Blues = plt.get_cmap('Blues')
    if score >= starter_score:
        normed_value = (score - starter_score) / (max_score - starter_score)
        color_tuple = Blues(normed_value)
        return color_tuple
    else:
        return (0, 0, 0)


def generate_edges_for_candidate(graph, candidate, candidate_name_to_node):
    id = candidate_name_to_node[candidate.get_name()]
    if not isinstance(candidate, RawFeature):
        for p in range(len(candidate.parents)):
            if candidate.parents[p].get_name() in candidate_name_to_node:
                if not graph.has_edge(str(id), candidate_name_to_node[candidate.parents[p].get_name()]):
                    graph.add_edge(str(id), candidate_name_to_node[candidate.parents[p].get_name()])
                generate_edges_for_candidate(graph, candidate.parents[p], candidate_name_to_node)


def get_parent_graph(candidate_id, parent: CandidateFeature, candidate_name_to_node, graph):
    if parent.get_name() in candidate_name_to_node:
        if not graph.has_edge(str(candidate_id), candidate_name_to_node[parent.get_name()]):
            graph.add_edge(str(candidate_id), candidate_name_to_node[parent.get_name()])
    else:
        if not isinstance(parent, RawFeature):
            for p in parent.parents:
                get_parent_graph(candidate_id, p, candidate_name_to_node, graph)


def generate_edges_for_connected_graph(graph, candidate, candidate_name_to_node):
    id = candidate_name_to_node[candidate.get_name()]
    if not isinstance(candidate, RawFeature):
        for p in range(len(candidate.parents)):
            if candidate.parents[p].get_name() in candidate_name_to_node:
                if not graph.has_edge(str(id), candidate_name_to_node[candidate.parents[p].get_name()]):
                    graph.add_edge(str(id), candidate_name_to_node[candidate.parents[p].get_name()])
                generate_edges_for_candidate(graph, candidate.parents[p], candidate_name_to_node)
            else:
                get_parent_graph(id, candidate.parents[p], candidate_name_to_node, graph)


def candidates_to_graph(candidates, candidate_scores, start_score, graph_file='/tmp/node_new.graphml', connected=False):

    max_score = np.max(candidate_scores)

    graph = nx.Graph()

    candidate_name_to_node = {}
    for i in range(len(candidates)):
        id = str(i)
        candidate = candidates[i]
        score = candidate_scores[i]
        candidate_name_to_node[candidate.get_name()] = id

        size = -1.0
        if score > start_score:
            normed_value = (score - start_score) / (max_score - start_score)
            size = str(100 * normed_value)
        else:
            size = str(1)

        graph.add_node(id)
        graph.node[id]['label'] = candidate.get_name()
        graph.node[id]['viewLabel'] = candidate.get_name()
        graph.node[id]['score'] = str(score)
        graph.node[id]['size'] = 5
        graph.node[id]['r'] = int(get_color_tuple(score, start_score, max_score)[0] * 256)
        graph.node[id]['g'] = int(get_color_tuple(score, start_score, max_score)[1] * 256)
        graph.node[id]['b'] = int(get_color_tuple(score, start_score, max_score)[2] * 256)



    for i in range(len(candidates)):
        if connected:
            generate_edges_for_connected_graph(graph, candidates[i], candidate_name_to_node)
        else:
            generate_edges_for_candidate(graph, candidates[i], candidate_name_to_node)



    for isolate in nx.isolates(graph):
        print(isolate)

    print("number isolates: " + str(len(list(nx.isolates(graph)))))

    nx.write_graphml(graph, graph_file)
    nx.write_gml(graph, "/tmp/test.gml")