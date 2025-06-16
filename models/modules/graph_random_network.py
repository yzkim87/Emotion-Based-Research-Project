import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix  # 강제 CSR

class GraphRandomNetwork(nn.Module):
    def __init__(self, top_k=20, rmax=1e-5, num_hops=3):
        super(GraphRandomNetwork, self).__init__()
        self.top_k = top_k
        self.rmax = rmax
        self.num_hops = num_hops

    def build_sparse_adj(self, edge_index, num_nodes):
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        data = np.ones(len(row))

        adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        # Row-normalize
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1

        # ✅ CSR-safe row-normalization
        diag_inv_deg = csr_matrix(np.diag(1.0 / deg))
        adj = diag_inv_deg.dot(adj)

        # print(">>> adj type:", type(adj))  # 디버그 확인

        return adj

    def gfpush(self, adj, s_idx):
        N = adj.shape[0]
        p = np.zeros(N)
        r = np.zeros(N)
        r[s_idx] = 1.0

        queue = [s_idx]

        while queue:
            u = queue.pop(0)
            if r[u] < self.rmax:
                continue

            push_val = r[u]
            p[u] += push_val * 0.5
            push_residual = push_val * 0.5

            for v in adj.indices[adj.indptr[u]:adj.indptr[u+1]]:
                w = adj[u, v]
                delta = push_residual * w
                r[v] += delta
                if r[v] >= self.rmax and v not in queue:
                    queue.append(v)

            r[u] = 0.0

        if self.top_k < N:
            idx = np.argpartition(-p, self.top_k)[:self.top_k]
            sparse_p = np.zeros_like(p)
            sparse_p[idx] = p[idx]
            p = sparse_p

        return p

    def forward(self, node_feats, edge_index):
        N, D = node_feats.shape
        adj = self.build_sparse_adj(edge_index, N)
        push_matrix = []
        for s_idx in range(N):
            p_row = self.gfpush(adj, s_idx)
            push_matrix.append(p_row)
        P = np.vstack(push_matrix)
        P = torch.tensor(P, dtype=node_feats.dtype, device=node_feats.device)
        latent_feats = P @ node_feats
        return latent_feats
