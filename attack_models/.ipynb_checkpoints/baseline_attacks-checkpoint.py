
from scipy.sparse import csr_matrix

from deeprobust.graph.global_attack import (DICE, Metattack, MinMax, PGDAttack,
                                            Random)
from deeprobust.graph.utils import preprocess


def baseline_attacks(args, surrogate,adj, features, labels, n_perturbation,device,idx_train=None, idx_unlabeled=None):
    if args.attack_method == 'metattack':
        model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                          attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled,
                     n_perturbations=n_perturbation, ll_constraint=False)
    elif args.attack_method == 'dice':
        model = DICE()
        model.attack(adj, labels, n_perturbations=n_perturbation)
    elif args.attack_method == 'random':
        model = Random()
        model.attack(adj, n_perturbations=n_perturbation)
    elif args.attack_method == 'minmax':
        adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False)  # conver to tensor
        model = MinMax(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    elif args.attack_method == 'pgd':
        adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False) # conver to tensor
        model = PGDAttack(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    else:
        raise ValueError('Invalid name of the attack method!')
    return model


