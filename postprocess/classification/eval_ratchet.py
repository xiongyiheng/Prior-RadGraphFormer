import json
import numpy as np

def map_cls_name(ls,mode,token_ls=None):
    """

    :param ls: a list of numbers that need to be converted to names
    :param mode: "tokens" "labels" or "edges"
    :return: a list of name with shape = ls
    """
    ls_name = []

    if mode == "tokens":
        with open('/home/guests/mlmi_kamilia/RadGraph Relationformer_matcher1/datasets/radgraph/OBS_ANAT_temp.json', 'r') as f:
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



def eval():
    "given id of study, return the cls list []"
    with open("/home/guests/mlmi_kamilia/Radgraphformer_emb/datasets/radgraph/dev_all.json", 'r') as f:
      data = json.load(f)

    tp = np.zeros(4)
    fp = np.zeros(4)
    fn = np.zeros(4)

    for id in data:
        gt_tokens = map_cls_name(data[id]["tokens"],"tokens",None)
        #print(gt_tokens)
        #break
        gt_edges = map_cls_name(data[id]["edges"], "edges", gt_tokens)

        #ls = [0,0,0,0]#["atelectasis", "edema", "pleural effusion", "lung opacity"]

        #find pred_cls in json
        with open("/home/guests/mlmi_kamilia/RATCHET/classification/extracted_cls_ratchet_AP.json", 'r') as f:
          pre = json.load(f)
        if id.replace('/','_') in pre.keys():
            pre_ls = pre[id.replace('/','_')]
            ### iterate PRE
            if pre_ls[2]:
                if ['effusion', 'pleural', 'L'] in gt_edges:
                    tp[2] += 1
                else:
                    fp[2] +=1

            if pre_ls[3]:
                if ['opacity', 'lung', 'L'] in gt_edges:
                    tp[3] += 1
                else:
                    fp[3] +=1

            if pre_ls[0]:
                if 'atelectasis' in gt_tokens:
                    tp[0] += 1
                    #print("yes")
                else:
                    fp[0] +=1
            if pre_ls[1]:
                if 'edema' in gt_tokens:
                    tp[1] += 1
                    #print("yes")
                else:
                    fp[1] +=1

            ### iterate GT
            if ['effusion', 'pleural', 'L'] in gt_edges:
                if pre_ls[2] == 0:
                    fn[2] += 1

            if ['opacity', 'lung', 'L'] in gt_edges:
                if pre_ls[3] == 0:
                    fn[3] += 1

            if 'edema' in gt_tokens:
                #print("yes")
                if pre_ls[1] == 0:
                    fn[1] += 1

            if 'atelectasis' in gt_tokens:
                if pre_ls[0] == 0:
                    fn[0] += 1

    return tp, fp, fn

tp, fp, fn = eval()

for i in range(4):
    precision = tp[i] / (tp[i] + fp[i] + 1e-6)
    recall = tp[i] / (tp[i] + fn[i] + 1e-6)
    print(2*precision*recall / (precision + recall + 1e-6))

    # 0.41666617669809775
    # 0.2918913951797972
    # 0.5265818035319086
    # 0.19298202724008617









