import torch
import torch.nn as nn

"""
    Colored Self-Attention Layer  
"""
def combine_h_p(h, p, operation='sum'):
    if operation == 'concat':
        h = torch.cat((h, p), dim=-1)
    elif operation == 'sum':
        h = h + p
    elif operation == 'product':
        h = h * p
    return h


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, in_dim_edges, out_dim, num_heads,
                 use_bias=False, share_edge_features=True,
                 attn_dropout=0.0):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.share_edge_features = share_edge_features
        if not self.share_edge_features:
            self.E_att = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)
            self.E_value = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        
    def forward(self, h, e=None, e_att=None, e_value=None, attn_mask=None):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        n_batch, num_nodes, full_out_dim = V_h.shape

        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        Q_h = Q_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)

        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K_h = K_h * scaling

        # Compute node-node attention scores.
        # Leave trailing dimension to add the edge features
        scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)

        # Make sure attention scores for padding are 0.
        # Keep -1e24 instead of -infinity to avoid NaN errors in softmax
        attn_mask = attn_mask.view(-1, num_nodes, num_nodes, 1, 1)
        scores = scores - 1e24 * (~attn_mask)
        # scores = scores + torch.log(attn_mask)
        
        # Compute edge features if necessary
        E_att = e_att if self.share_edge_features else self.E_att(e)
        E_value = e_value if self.share_edge_features else self.E_value(e)
        # Match the shape of `scores`
        E_att = E_att.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)

        # Bias the attention matrix with the edge filters
        scores = scores + E_att
        # The head dimension is not needed anymore, merge it with the feature dimension
        scores = scores.reshape(-1, num_nodes, num_nodes, self.num_heads * self.out_dim)

        # L1-normalize across emitting nodes
        scores = nn.functional.softmax(scores, dim=2)

        # Dropout connections
        attn_mask = self.attn_dropout(attn_mask.float()).squeeze(-1)
        scores = scores * attn_mask

        # Modulate and sum the node messages hi <- Sum_j a(i,j) * Vj
        h = torch.einsum('bijk,bjk->bik', scores, V_h)

        # Add the edge messages
        h += (scores * E_value).sum(2)

        # Make sure return type is contiguous
        return h.contiguous()