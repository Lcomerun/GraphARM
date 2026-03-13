import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

def random_node_decay_ordering(datapoint):
    # create random list of nodes
    return torch.randperm(datapoint.x.shape[0]).tolist()

class NodeMasking:
    def __init__(self, dataset):
        self.dataset = dataset

        # Support both 1-D and multi-dim node features [N, D].
        # The first feature dimension (dim 0) is the discrete type used for
        # masking/classification; remaining dims are continuous attributes.
        x = dataset.x if dataset.x.dim() == 2 else dataset.x.unsqueeze(-1)
        self.node_feature_dim = x.shape[1]

        # Support both 1-D and multi-dim edge features [E, D].
        # dim 0 is the discrete edge type.
        if dataset.edge_attr.dim() == 2:
            edge_attr = dataset.edge_attr
        else:
            edge_attr = dataset.edge_attr.unsqueeze(-1)
        self.edge_feature_dim = edge_attr.shape[1]

        # Special token indices (based on the type dimension, dim 0).
        self.NODE_MASK = int(x[:, 0].unique().shape[0])
        self.EMPTY_EDGE = int(edge_attr[:, 0].unique().shape[0])
        self.EDGE_MASK = self.EMPTY_EDGE + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_2d(self, tensor):
        '''Ensure tensor is 2-D [N, D]; expands 1-D [N] → [N, 1].'''
        if tensor is None:
            return tensor
        return tensor if tensor.dim() == 2 else tensor.unsqueeze(-1)

    def _node_mask_features(self):
        '''Feature vector for a masked node: type=NODE_MASK, rest=0.'''
        feat = torch.zeros(self.node_feature_dim)
        feat[0] = self.NODE_MASK
        return feat

    def _edge_mask_features(self):
        '''Feature vector for a masked edge: type=EDGE_MASK, rest=0.'''
        feat = torch.zeros(self.edge_feature_dim)
        feat[0] = self.EDGE_MASK
        return feat

    def _empty_edge_features(self):
        '''Feature vector for an empty (absent) edge: type=EMPTY_EDGE, rest=0.'''
        feat = torch.zeros(self.edge_feature_dim)
        feat[0] = self.EMPTY_EDGE
        return feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def idxify(self, datapoint):
        '''
        Converts the discrete type dimensions (x[:,0] and edge_attr[:,0]) to
        contiguous 0-based indices.  Continuous feature dimensions are unchanged.
        '''
        datapoint = datapoint.clone()
        x = self._to_2d(datapoint.x).clone()
        ea = self._to_2d(datapoint.edge_attr).clone()

        # Re-index node type (dim 0)
        unique_node_types = {nt.item(): idx for idx, nt in enumerate(x[:, 0].unique())}
        x[:, 0] = torch.tensor([unique_node_types[nt.item()] for nt in x[:, 0]],
                                dtype=x.dtype)

        # Re-index edge type (dim 0)
        unique_edge_types = {et.item(): idx for idx, et in enumerate(ea[:, 0].unique())}
        ea[:, 0] = torch.tensor([unique_edge_types[et.item()] for et in ea[:, 0]],
                                 dtype=ea.dtype)

        datapoint.x = x
        datapoint.edge_attr = ea
        return datapoint

    def deidxify(self, datapoint):
        '''
        Converts contiguous type indices back to their original values.
        '''
        datapoint = datapoint.clone()
        x = self._to_2d(datapoint.x).clone()
        ea = self._to_2d(datapoint.edge_attr).clone()

        unique_node_types = {idx: nt.item() for idx, nt in enumerate(x[:, 0].unique())}
        unique_edge_types = {idx: et.item() for idx, et in enumerate(ea[:, 0].unique())}

        x[:, 0] = torch.tensor([unique_node_types.get(nt.item(), self.NODE_MASK)
                                 for nt in x[:, 0]], dtype=x.dtype)
        ea[:, 0] = torch.tensor([unique_edge_types.get(et.item(), self.EDGE_MASK)
                                  for et in ea[:, 0]], dtype=ea.dtype)

        datapoint.x = x
        datapoint.edge_attr = ea
        return datapoint

    def is_masked(self, datapoint, node=None):
        '''
        Returns whether node(s) are masked, based on the type dimension (dim 0).
        '''
        x = self._to_2d(datapoint.x)
        if node is None:
            return x[:, 0] == self.NODE_MASK
        return x[node, 0] == self.NODE_MASK

    def remove_node(self, datapoint, node):
        '''
        Removes a node from the graph together with all its edges.
        '''
        assert node < datapoint.x.shape[0], "Node does not exist"
        if datapoint.x.shape[0] == 1:
            return datapoint.clone()
        datapoint = datapoint.clone()

        # Remove node features
        datapoint.x = torch.cat([datapoint.x[:node], datapoint.x[node + 1:]])

        if datapoint.edge_index.shape[1] > 1:
            # Keep only edges that do not involve the removed node
            keep = torch.tensor([node not in ei for ei in datapoint.edge_index.T])
            ea = self._to_2d(datapoint.edge_attr)
            datapoint.edge_attr = ea[keep]
            datapoint.edge_index = datapoint.edge_index[:, keep]
            # Shift indices above the removed node
            datapoint.edge_index[datapoint.edge_index > node] -= 1
        return datapoint

    def add_masked_node(self, datapoint):
        '''
        Appends a fully-masked node and connects it (with masked edges) to all
        existing nodes.
        '''
        datapoint = datapoint.clone()
        n_nodes = datapoint.x.shape[0]
        x = self._to_2d(datapoint.x)
        ea = self._to_2d(datapoint.edge_attr)

        # New masked node
        new_node = self._node_mask_features().unsqueeze(0)
        datapoint.x = torch.cat([x, new_node], dim=0)

        # Masked edges for the new node (one edge from each existing node + self)
        new_edges_feat = self._edge_mask_features().unsqueeze(0).expand(n_nodes + 1, -1).clone()
        datapoint.edge_attr = torch.cat([ea, new_edges_feat], dim=0)

        new_ei = torch.tensor([(i, n_nodes) for i in range(n_nodes + 1)],
                               dtype=torch.long).transpose(1, 0)
        datapoint.edge_index = torch.cat([datapoint.edge_index, new_ei], dim=1)
        return datapoint

    def mask_node(self, datapoint, selected_node):
        '''
        Masks a node: replaces its features with the NODE_MASK sentinel and
        sets all incident edges to EDGE_MASK.
        '''
        datapoint = datapoint.clone()
        x = self._to_2d(datapoint.x).clone()
        ea = self._to_2d(datapoint.edge_attr).clone()

        x[selected_node] = self._node_mask_features()

        src_mask = datapoint.edge_index[0] == selected_node
        dst_mask = datapoint.edge_index[1] == selected_node
        ea[src_mask] = self._edge_mask_features()
        ea[dst_mask] = self._edge_mask_features()

        datapoint.x = x
        datapoint.edge_attr = ea
        return datapoint

    def _reorder_edge_attr_and_index(self, graph):
        '''
        Reorders edge_attr and edge_index into dense (i,j) order.
        '''
        graph = graph.clone()
        n = graph.x.shape[0]
        ea = self._to_2d(graph.edge_attr)

        full_ea = self._empty_edge_features().unsqueeze(0).expand(n * n, -1).clone()
        for ea_row, ei in zip(ea, graph.edge_index.T):
            full_ea[ei[0] * n + ei[1]] = ea_row

        graph.edge_attr = full_ea
        graph.edge_index = torch.stack(
            [torch.tensor([i, j]) for i in range(n) for j in range(n)], dim=1
        ).long()
        return graph

    def remove_empty_edges(self, graph):
        '''
        Removes EMPTY_EDGE entries from the graph.
        '''
        graph = graph.clone()
        ea = self._to_2d(graph.edge_attr)
        keep = ea[:, 0].squeeze() != self.EMPTY_EDGE
        graph.edge_index = graph.edge_index[:, keep]
        graph.edge_attr = ea[keep]
        return graph

    def demask_node(self, graph, selected_node, node_type, connections_types):
        '''
        Unmasks a node.

        Parameters
        ----------
        node_type : scalar tensor or 1-D feature vector
            If scalar (or single-element tensor), sets only the type dimension
            (dim 0) of the node feature.  If a full feature vector is supplied
            its length must equal node_feature_dim.
        connections_types : 1-D tensor [N]
            Edge-type indices for edges from each existing node to selected_node.
        '''
        assert connections_types.shape[0] == graph.x.shape[0], \
            "Number of connections must equal number of nodes"

        graph = graph.clone()
        x = self._to_2d(graph.x).clone()
        ea = self._to_2d(graph.edge_attr).clone()

        # Set node features FIRST so is_masked() returns the updated state
        # when we iterate over edges below (handles the self-loop case).
        nt = torch.as_tensor(node_type)
        if nt.dim() == 0 or nt.numel() == 1:
            x[selected_node, 0] = nt.squeeze().to(x.dtype)
        else:
            x[selected_node] = nt.to(x.dtype)
        graph.x = x  # propagate the update before the edge loop

        # Set edge features
        for i, connection in enumerate(connections_types):
            if not self.is_masked(graph, node=i):
                ct = torch.as_tensor(connection)
                mask1 = torch.logical_and(graph.edge_index[0] == i,
                                          graph.edge_index[1] == selected_node)
                mask2 = torch.logical_and(graph.edge_index[1] == i,
                                          graph.edge_index[0] == selected_node)
                if ct.dim() == 0 or ct.numel() == 1:
                    ea[mask1, 0] = ct.squeeze().to(ea.dtype)
                    ea[mask2, 0] = ct.squeeze().to(ea.dtype)
                else:
                    ea[mask1] = ct.to(ea.dtype)
                    ea[mask2] = ct.to(ea.dtype)

        graph.edge_attr = ea
        return graph

    def fully_connect(self, graph, keep_original_edges=True):
        '''
        Fully connects the graph; missing edges receive EMPTY_EDGE features.
        '''
        adjacency_matrix = to_dense_adj(graph.edge_index)[0]
        adjacency_matrix[adjacency_matrix == 0] = 1

        n = graph.x.shape[0]
        fully_connected = graph.clone()
        ea = self._to_2d(graph.edge_attr)

        empty_feat = self._empty_edge_features()
        full_ea = empty_feat.unsqueeze(0).expand(n * n, -1).clone()

        if keep_original_edges:
            for ea_row, ei in zip(ea, graph.edge_index.T):
                i, j = ei[0].item(), ei[1].item()
                full_ea[i * n + j] = ea_row
                full_ea[j * n + i] = ea_row  # Ensure symmetry

        fully_connected.edge_attr = full_ea
        fully_connected.edge_index = torch.nonzero(adjacency_matrix).T
        return fully_connected

    def generate_fully_masked(self, n_nodes):
        '''
        Generates a fully-masked graph (all nodes and edges masked).
        '''
        node_feat = self._node_mask_features()
        x = node_feat.unsqueeze(0).expand(n_nodes, -1).clone()

        edge_feat = self._edge_mask_features()
        ea = edge_feat.unsqueeze(0).expand(n_nodes * n_nodes, -1).clone()

        edge_index = torch.tensor(
            [(i, j) for i in range(n_nodes) for j in range(n_nodes)],
            dtype=torch.int64
        ).transpose(0, 1)

        return Data(x=x, edge_index=edge_index, edge_attr=ea)

    def get_denoised_nodes(self, graph):
        '''
        Returns a list of node indices that are not masked.
        '''
        return [node for node in range(graph.x.shape[0])
                if not self.is_masked(graph, node)]