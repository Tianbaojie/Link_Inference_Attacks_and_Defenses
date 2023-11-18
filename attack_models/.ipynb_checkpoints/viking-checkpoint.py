from attack_models.VIKING.perturbation_attack import *
from attack_models.VIKING.utils import *
def get_candidates(adj_matrix, ctype='combined', numcds=5000):
    #adj_matrix_sparses
    if ctype == 'addition':
        candidates = generate_candidates_addition(
            adj_matrix=adj_matrix, n_candidates=numcds)
    elif ctype == 'removal':
        candidates = generate_candidates_removal(adj_matrix=adj_matrix)
    elif ctype == 'combined':
        candidates1 = generate_candidates_addition(
            adj_matrix=adj_matrix, n_candidates=numcds)
        candidates2 = generate_candidates_removal(adj_matrix=adj_matrix)
        candidates = np.concatenate([candidates1, candidates2])
    return candidates

def get_attacked_graph_viking(adj_matrix,attack='our', n_flips=None, dim=None, window_size=None, L=None):
    candidates=get_candidates(adj_matrix)
    if attack is not None:
        if attack == 'rnd':
            flips = baseline_random_top_flips(candidates, n_flips, 0)
        elif attack == 'deg':
            flips = baseline_degree_top_flips(
                adj_matrix, candidates, n_flips, True)
        elif attack == 'our':
            flips = perturbation_top_flips(
                adj_matrix, candidates, n_flips, dim, window_size, L)
        elif attack == 'ori':
            flips = perturbation_top_flips(
                adj_matrix, candidates, n_flips, dim, window_size, mode='unsup')
        adj_matrix_flipped = flip_candidates(adj_matrix, flips)
        return adj_matrix_flipped
    return adj_matrix