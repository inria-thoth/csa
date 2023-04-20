import torch
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('DenseEdge')
class DenseEdgeEncoder(torch.nn.Module):
    '''
    Creates dense edge features `batch.edge_dense` of size 
    `(n_batch, batch_nodes, batch_nodes, emb_dim)` from `batch.edge_attr`

    Fills missing edge features by adding a learnable vector of size `emb_dim`
    to every pair of disconnected nodes (i, j), and another one to the diagonal (i, i)

    `input_batch.edge_attr` should be of compatible last dimension `emb_dim`
    '''
    def __init__(self, emb_dim, ignore_rings=True):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=3, embedding_dim=emb_dim, padding_idx=0)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        self.ignore_rings = ignore_rings

    def forward(self, batch):
        '''
        Create a dense edge features matrix `batch.edge_dense`,
        E_ij = edge_attr[i, j] if (i, j) are neighbors
               embedding_1 if i=j
               embedding_2 else
        '''
        if not hasattr(batch, 'edge_dense'):
            batch.edge_dense = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
        A_dense = get_dense_edge_types(batch, ignore_rings=self.ignore_rings)
        batch.edge_dense += self.encoder(A_dense)

        return batch


def get_dense_edge_types(batch, ignore_rings=True):
    '''
    Returns a dense complementary adjacency matrix of `batch`,
    of size `(n_batch, batch_nodes, batch_nodes, 1)`
    Matrix A_ij = 0 if (i,j) are connected (either by a bond or a ring));
                  1 if i=j;
                  2 otherwise
    FIXME: Should differentiate padding and edges (both have same value as 
           pyG `to_dense_adj` returns 0 for both)
    '''
    edge_index = batch.edge_index
    if not ignore_rings:
        edge_index = torch.cat([edge_index, batch.ring_index], dim=1) 
    edge_attr = 2 * torch.ones_like(edge_index[0])
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=1)
    # to_dense_adj returns a float tensor even when edge_attr is int...
    A_dense = 2 - to_dense_adj(edge_index, batch=batch.batch, edge_attr=edge_attr).long()

    return A_dense
