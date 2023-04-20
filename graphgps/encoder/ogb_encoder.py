import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

@register_node_encoder('Atom1')
class Atom1Encoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(Atom1Encoder, self).__init__()
        
        self.atom_embedding = torch.nn.Embedding(full_atom_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.atom_embedding.weight.data)


    def forward(self, batch):
        batch.x = self.atom_embedding(batch.x[:, 0])

        return batch

@register_node_encoder('Bond1')
class Bond1Encoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(Bond1Encoder, self).__init__()
        
        self.bond_embedding = torch.nn.Embedding(full_bond_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.bond_embedding.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.bond_embedding(batch.edge_attr[:, 0])

        return batch   


@register_node_encoder('AtomCustom')
class AtomEncoderAvg(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        """"""
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])

        batch.x = encoded_features / (batch.x.shape[1]).sqrt()
        return batch



@register_edge_encoder('BondCustom')
class BondEncoderAvg(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """
    def __init__(self, emb_dim):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()
    
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        """"""
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        batch.edge_attr = bond_embedding / (batch.edge_attr.shape[1]).sqrt()
        return batch

