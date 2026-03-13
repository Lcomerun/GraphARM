import torch
from torch_geometric.data import Data
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb
import os

from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking
from grapharm import GraphARM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")


# ---------------------------------------------------------------------------
# B-rep CAD dataset
# ---------------------------------------------------------------------------
# Replace the block below with your actual dataset loader.
# The expected PyG Data format for each sample is:
#
#   data.x          [N, 11]  node features:
#                              dim 0   : face_type  (int 0–6)
#                              dim 1–6 : face_BBox  (float, scaled ×3, [-3, 3])
#                              dim 7–9 : face_normal (float, [-1, 1])
#                              dim 10  : loop_count  (int ≥ 1)
#
#   data.edge_attr  [2E, 8]  edge features:
#                              dim 0   : edge_type  (int 0–3)
#                              dim 1   : convexity  (int 0–2)
#                              dim 2–7 : edge_BBox  (float, scaled ×3, [-3, 3])
#
#   data.edge_index [2, 2E]  undirected graph (both directions per B-rep edge)
#   data.num_faces  = N
#   data.num_brep_edges = E
#   data.uid        = sample identifier string
#
# ---------------------------------------------------------------------------

def make_synthetic_brep_graph(n_faces: int = 4,
                               n_edges: int = 5,
                               num_face_types: int = 7,
                               num_edge_types: int = 4) -> Data:
    '''
    Generates a random synthetic B-rep graph for testing / demonstration.
    '''
    # Node features [N, 11]
    face_type   = torch.randint(0, num_face_types, (n_faces, 1)).float()
    face_bbox   = torch.rand(n_faces, 6) * 6 - 3          # [-3, 3]
    face_normal = torch.nn.functional.normalize(
                      torch.randn(n_faces, 3), dim=-1)     # [-1, 1]
    loop_count  = torch.randint(1, 4, (n_faces, 1)).float()
    x = torch.cat([face_type, face_bbox, face_normal, loop_count], dim=1)  # [N, 11]

    # Build undirected edges (both directions) – random pairs
    edges = []
    for _ in range(n_edges):
        i = torch.randint(0, n_faces, ()).item()
        j = torch.randint(0, n_faces, ()).item()
        if i != j:
            edges.append((i, j))
            edges.append((j, i))
    if not edges:
        edges = [(0, 1), (1, 0)]
    src, dst = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    n_directed = edge_index.shape[1]

    # Edge features [2E, 8]
    edge_type  = torch.randint(0, num_edge_types, (n_directed, 1)).float()
    convexity  = torch.randint(0, 3,              (n_directed, 1)).float()
    edge_bbox  = torch.rand(n_directed, 6) * 6 - 3
    edge_attr  = torch.cat([edge_type, convexity, edge_bbox], dim=1)  # [2E, 8]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_faces      = n_faces
    data.num_brep_edges = n_edges
    data.uid            = "synthetic"
    return data


# Build a small synthetic dataset (replace with real data loading)
NUM_GRAPHS    = 10
NUM_FACE_TYPES = 7   # face_type ∈ {0,...,6}
NUM_EDGE_TYPES = 4   # edge_type ∈ {0,...,3}
NODE_FEATURE_DIM = 11
EDGE_FEATURE_DIM = 8

dataset = [make_synthetic_brep_graph(n_faces=torch.randint(3, 8, ()).item(),
                                      n_edges=torch.randint(3, 10, ()).item(),
                                      num_face_types=NUM_FACE_TYPES,
                                      num_edge_types=NUM_EDGE_TYPES)
           for _ in range(NUM_GRAPHS)]

# Use the first graph to initialise NodeMasking (it inspects feature shapes)
ref_graph = dataset[0]

diff_ord_net = DiffusionOrderingNetwork(
    node_feature_dim=NODE_FEATURE_DIM,
    num_node_types=NUM_FACE_TYPES,
    num_edge_types=NUM_EDGE_TYPES,
    num_layers=3,
    out_channels=1,
    device=device
)

masker = NodeMasking(ref_graph)

denoising_net = DenoisingNetwork(
    node_feature_dim=NODE_FEATURE_DIM,
    edge_feature_dim=EDGE_FEATURE_DIM,
    num_node_types=NUM_FACE_TYPES,
    num_edge_types=NUM_EDGE_TYPES,
    num_layers=7,
    device=device
)


wandb.init(
        project="GraphARM",
        group=f"v2.4.0",
        name=f"BRepCAD_GraphARM",
        config={
            "policy": "train",
            "n_epochs": 10000,
            "batch_size": 1,
            "lr": 1e-3,
            "node_feature_dim": NODE_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "num_face_types": NUM_FACE_TYPES,
            "num_edge_types": NUM_EDGE_TYPES,
        },
        # mode='disabled'
    )

torch.autograd.set_detect_anomaly(True)


grapharm = GraphARM(
    dataset=ref_graph,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

batch_size = 5
try:
    grapharm.load_model()
    print("Loaded model")
except:
    print("No model to load")
# train loop
for epoch in range(2000):
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_batch=dataset[2*epoch*batch_size:(2*epoch + 1)*batch_size],
        val_batch=dataset[(2*epoch + 1)*batch_size:batch_size*(2*epoch + 2)],
        M=4
    )
    grapharm.save_model()