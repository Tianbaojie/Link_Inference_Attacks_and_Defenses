from torch_geometric.utils import contains_isolated_nodes, remove_isolated_nodes


def process_isolated_nodes(edge_index):
    if contains_isolated_nodes(edge_index):
        new_edge_index, _, mask = remove_isolated_nodes(edge_index)
        mapping = {}
        for i in range(edge_index.shape[1]):
            if edge_index[0, i] != new_edge_index[0, i]:
                mapping[new_edge_index[0, i].item()] = edge_index[0, i].item()
        return new_edge_index, mapping, mask
    else:
        return edge_index, None, [True]*(edge_index.shape[1])
def restore_isolated_ndoes(new_edge_index, mapping):
    for i in range(new_edge_index.shape[1]):
        if new_edge_index[0, i].item() in mapping:
            new_edge_index[0, i] = mapping[new_edge_index[0, i].item()]
        if new_edge_index[1, i].item() in mapping:
            new_edge_index[1, i] = mapping[new_edge_index[1, i].item()]
    return new_edge_index

def restore_isolated_ndoes_int(new_edge_index, mask):
    mapping={}
    i=0
    index=0
    while index < len(mask):
        if mask[index]:
            mapping[i]=index
            i+=1
        index+=1

    for i in range(new_edge_index.shape[1]):
        new_edge_index[0, i]=mapping[new_edge_index[0, i].item()]
        new_edge_index[1, i]=mapping[new_edge_index[1, i].item()]
    return new_edge_index
