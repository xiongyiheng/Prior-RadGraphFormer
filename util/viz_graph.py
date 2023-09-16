"""
A helper function library for viz the graph containing nodes and edges.
"""

import igraph as ig
from igraph import *
import json
import matplotlib.pyplot as plt
import torch
import numpy as np

EDGE_THRESHOLD = 0.6

def filter_bg(dict,mode="gt"):
    """
    filter out all BG tokens,labels and edges
    :param dict: Pre dict or GT dict
    :return: filtered out dict
    """
    if mode == "gt":        # rect['tokens'], rect['labels'], rect['edges']
        dict['tokens'] = dict['tokens'].cpu().detach().numpy().squeeze()
        dict['labels'] = dict['labels'].cpu().detach().numpy().squeeze()
        dict['edges'] = dict['edges'].cpu().detach().numpy().squeeze()
        #delete all bg for gt
        #dict['edges'] = np.delete(dict['edges'],np.where(dict['edges'][:,2]==0))
        dict['tokens'] = np.delete(dict['tokens'], np.where(dict['tokens'] == 0))
        dict['labels'] = np.delete(dict['labels'], np.where(dict['labels'] == 0))
        len = dict['edges'].shape[0]
        edge_ls = []
        #print(dict['edges'].shape)
        for i in range(len):
            #print(dict['pred_rels_score'][i,dict['pred_rels_class'][i]])
            if dict['edges'][i,2] != 0:
                edge = [0,0,0]
                edge[:3] = dict['edges'][i, :]
                edge_ls.append(edge)
        dict['edges'] = edge_ls

    if mode == "pre":   # out['pred_rels'], out['pred_rels_class'], out['pred_rels_score'],out['pred_tokens'], out['pred_labels']
        dict['pred_rels'] = np.asarray(dict['pred_rels']).squeeze()#.cpu().detach().numpy()
        dict['pred_rels_class'] = np.asarray(dict['pred_rels_class']).squeeze()#.cpu().detach().numpy()
        dict['pred_rels_score'] = np.asarray(dict['pred_rels_score']).squeeze()#.cpu().detach().numpy()
        dict['tokens'] = np.asarray(dict['pred_tokens']).squeeze()#.cpu().detach().numpy()
        dict['labels'] = np.asarray(dict['pred_labels']).squeeze()#.cpu().detach().numpy()

        # filter out edges with low scores & merge into 'edge'
        len = dict['pred_rels_score'].shape[0]
        edge_ls = []
        #print(dict['pred_rels_score'].shape)
        for i in range(len):
            print(dict['pred_rels_score'][i,dict['pred_rels_class'][i]])
            if dict['pred_rels_score'][i,dict['pred_rels_class'][i]] >= EDGE_THRESHOLD:
                edge_array = [0,0,0]
                edge_array[:2] = dict['pred_rels'][i, :]
                edge_array[2] = dict['pred_rels_class'][i]
                edge_ls.append(edge_array)
        dict['edges'] = edge_ls
        # id_over_thresh = np.where(dict['pred_rels_score'] > EDGE_THRESHOLD)
        # print(id_over_thresh)
        # print(dict['pred_rels_class'].shape)
        # print(dict['pred_rels_score'])
        # dict['pred_rels_class'] = dict['pred_rels_class'][id_over_thresh]
        # dict['pred_rels'] = dict['pred_rels'][id_over_thresh,:]
        # # remove all bg
        # edge_ls = np.array([])
        # for i in range(len(dict['pred_rels_class'])):
        #     edge_array = np.zeros([3])
        #     edge_array[:2] = dict['pred_rels'][i,:]
        #     edge_array[2] = dict['pred_rels'][i, :]
        #     edge_ls = np.append(edge_ls,edge_array)
        # dict['edges'] = edge_ls

        #print(dict)

    return dict

def map_cls_name(ls,mode,token_ls=None):
    """

    :param ls: a list of numbers that need to be converted to names
    :param mode: "tokens" "labels" or "edges"
    :return: a list of name with shape = ls
    """
    ls_name = []

    if mode == "tokens":
        with open('/home/guests/mlmi_kamilia/RadGraph Relationformer_matcher1/datasets/radgraph/OBS_ANAT_list.json', 'r') as f:
            map_dict = json.load(f)
            map_ls = map_dict["total"]
        for i in range(len(ls)):
            name = map_ls[ls[i]]
            ls_name.append(name)
    if mode == "labels": # can not be "labels" for baseline
        map_ls = ["OBS-DA","ANAT-DP","OBS-DP","OBS-U"]
        for i in range(len(ls)):
            name = map_ls[ls[i]]
            ls_name.append(name)
    if mode == "edges":
        map_dict = ["M","L","S"]
        #print(ls)
        for i in range(len(ls)):
            rl_name = map_dict[ls[i][-1]]
            head_name = token_ls[ls[i][0]]
            edge_name = token_ls[ls[i][1]]

            #map tokens
            ls_name.append([head_name,edge_name,rl_name])

    return ls_name

def draw_graph(dict,mode,num):
    """
    draw graph using Igraph
    :param dict: a dictionary contains ["tokens","labels","edges"]. It could be either GT or Pre
        :param tokens: list in shape 1*len(tokens). No background here
        :param labels: list in same shape as tokens
        :param edges:  list in shape 3*len(edges)
    :return:
    """
    dict = filter_bg(dict,mode)

    tokens = dict["tokens"]
    tokens_name = map_cls_name(tokens,"tokens")
    labels = dict["labels"]
    edges = dict["edges"]
    edges_name = map_cls_name(edges,"edges")

    ### draw a graph ###
    g = ig.Graph(n=len(tokens),directed=True)
    g.add_edges([(edge[0],edge[1]) for edge in edges])
    g.vs["label"] =tokens_name
    g.es["label"] = edges_name
    g.vs["color"] = "Coral"
    #g.vs["vertex_size"] = 15
    #g.vs["vertex_label_size"] = 1
    len_ls = []
    for i in range(len(g.vs["label"])):
        len_vs = len(g.vs["label"][i])*8.5
        len_ls.append(len_vs)

    ### layout and plot ###
    #fig, ax = plt.subplots()
    layout = g.layout_kamada_kawai()
    ig.plot(g,"/home/guests/mlmi_kamilia/RadGraph Relationformer_matcher1/viz/"+str(num)+"_viz_"+mode+".pdf", vertex_size=50,vertex_label_size=15,layout=layout,bbox=(800, 800),margin=(50,50,50,50))#,vertex_label_size=10,vertex_size=20)
    # ig.plot(
    #     g,
    #     target=ax,
    #     edge_label = edges_name,
    #     vertex_label = tokens_name,
    #     vertex_size = 0.8,
    #     vertex_label_size=10,
    #     edge_font = 5000,
    #     layout=layout,
    #     edge_color='#666',
    #     edge_align_label=True,
    #     bbox=(500000, 500000)
    #     # margin=(10, 10, 10, 10)
    # )
    #plt.show()
def main():
    # with open('D:/studium/MIML/radgraph/GraphGen/baseline/pred_dict_from_radgraph.json', 'r') as f:
    #     sample_dict = json.load(f)
    # draw_graph(sample_dict["/home/guests/mlmi_kamilia/RATCHET/out_folder/p18_p18026902_s55739083.txt"],gt)
    token_ls = [101, 39, 151, 107, 222, 186, 73, 87, 18, 68]
    edge_ls = [[0, 1, 1], [2, 1, 1], [3, 4, 1], [5, 6, 1], [7, 8, 0], [9, 8, 1], [2, 1, 1]]

    token_ls = map_cls_name(token_ls,"tokens",None)
    edge_ls = map_cls_name(edge_ls, "edges", token_ls)

    print(token_ls)
    print(edge_ls)
    #port igraph as ig
    #
    # g = ig.Graph(directed=True, n=4)
    # g.add_edges([(0, 1), (0, 2), (2, 3)])
    # g.es["weight"] = [1, 2, 3]
    #
    # # 1st run
    # g.es["label"] = ["orange", "magenta", "purple"]
    # g.vs["label"] = ["vA", "vB", "vC", "vD"]
    #
    # # 2nd run
    # # g.es["label"] = ["blue", "green", "yellow"]
    # # g.vs["label"] = ["v1", "v2", "v3", "v4"]
    #
    # ig.plot(g, "output.png", layout=g.layout("rt"), bbox=(300, 300), edge_width=g.es['weight'])

if __name__ == '__main__':
    main()