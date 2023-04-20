from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # Share edge attention biases and values between layers.
    cfg.dataset.edge_encoder_shared = False

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # Whether to add ring information per graph
    cfg.dataset.rings = False
    cfg.dataset.rings_max_length = 6
    cfg.dataset.rings_coalesce_edges = False

    # Whether to add shortest path information per graph
    cfg.dataset.spd = False
    cfg.dataset.spd_max_length = 6

    # Whether to complete dense edge features
    cfg.dataset.edge_encoder_dense = True
