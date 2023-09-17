import json
from ast import literal_eval
with open('/home/jingsong/MIML/eval_nlp/radgraph_vanilla/all_out_mapped.json','r') as f:
    pred_graph = json.load(f)

def mapping_word_into_word_cluster(word,word_dict):

    if word in word_dict.keys():
        return word_dict[word]
    else:
        return word


def generate_text_based_on_rules(token_dict,edges_ls):
    """
    
    Args:
        token_dict: {"countour":"ANAT","normal":"OBS"} 
        edges_ls: [['contour', 'mediastinum', 'M'], ['contour', 'heart', 'M']]

    Returns:
        sen: list of strings
    """
    sen = []
    edges_ls_copy = edges_ls.copy()
    word_dict = {}
    for edge in edges_ls:
        head_label = token_dict[edge[0]]
        tail_label = token_dict[edge[1]]
        if head_label.startswith("ANAT") and tail_label.startswith("ANAT"):
            # in tht case we merge the two anatomies together as "head of the tail" for example "tube of lung"
            if edge[2] == 'S':
                sen.append(edge[0]+" is a suggestive of "+edge[1]+' .')
            if edge[2] == 'M':
                word_dict[edge[1]] = edge[0] + " of " + edge[1]
            idx = edges_ls_copy.index(edge)
            del edges_ls_copy[idx]

        if head_label.startswith("OBS") and tail_label.startswith("OBS"):
            if edge[2] == 'S':
                sen.append(edge[0]+" is a suggestive of "+edge[1]+' .')
            if edge[2] == 'M':
                word_dict[edge[1]] = edge[0] + " " + edge[1]   #"small effusion"
            idx = edges_ls_copy.index(edge)
            del edges_ls_copy[idx]

    for left_edge in edges_ls_copy:
        head_label = token_dict[left_edge[0]]
        tail_label = token_dict[left_edge[1]]
        if head_label.startswith("OBS") and tail_label.startswith("ANAT"):
            #generate a sentence to describe the relation between "OBS" and "ANAT" for each such edge
            if left_edge[2] == 'L':
                text_obs = mapping_word_into_word_cluster(left_edge[0],word_dict)
                text_anat = mapping_word_into_word_cluster(left_edge[1],word_dict)

                if text_obs in ["clear","normal","unchanged"]:

                    if head_label == "OBS-U":
                        ori_text = text_anat + " is possibly "+ text_obs + ". "
                    elif head_label == "OBS-DA":
                        ori_text = text_anat + " is not "+ text_obs + ". "
                    else:
                        ori_text = text_anat + " is "+ text_obs + ". "
                    sen.insert(0,ori_text.capitalize())
                else:
                    if head_label == "OBS-U":
                        sen.insert(0, "There is possibly " + text_obs + " located at " + text_anat + ". ")
                    elif head_label == "OBS-DA":
                        sen.insert(0, "There is no " + text_obs + " located at " + text_anat + ". ")
                    else:
                        sen.insert(0,"There is "+text_obs+" located at "+text_anat +". ")
            else:
                print("here")
                print(left_edge)

        else:
            print("there")
            print(left_edge)

    if len(sen) == 0:
        for key in word_dict:
            sen.insert(0,"There is some findings about "+word_dict[key]+" .")

    return sen

num = 0

for id in pred_graph:
    print(id)
    token_ls = literal_eval(pred_graph[id]["tokens"])
    #print(token_ls)
    labels_ls = literal_eval(pred_graph[id]["labels"])
    edges_ls = literal_eval(pred_graph[id]["edge"])
    #print(type(edges_ls))


    token_dict = {}
    for i in range(len(token_ls)):
        token_dict[token_ls[i]] = labels_ls[i]

    sen = generate_text_based_on_rules(token_dict,edges_ls)

    if len(sen) == 0:
        pass
    else:
        out_dir = '/home/jingsong/MIML/eval_nlp/radgraph_vanilla/reports/' + id.replace('/',"_")

        with open(out_dir, 'w') as f: # totally 463
            f.writelines(sen)
    #break

    # if num >20:
    #     break
    # num += 1
    # print(num)
    #break

