'''
Test Suite for the NodeMasking class with B-rep CAD data format.

Node features: x [N, 11]
  dim 0   : face_type  (int 0-6)
  dim 1-6 : face_BBox  (float [-3, 3])
  dim 7-9 : face_normal (float [-1, 1])
  dim 10  : loop_count (int >=1)

Edge features: edge_attr [E, 8]
  dim 0   : edge_type  (int 0-3)
  dim 1   : convexity  (int 0-2)
  dim 2-7 : edge_BBox  (float [-3, 3])

Tests:
    - masking a single node
    - demasking a single node
    - adding a masked node
    - removing a node
    - idxify
    - deidxify
    - is_masked
    - remove_node
    - generate_fully_masked
    - test fully_connect
'''


import pytest
import torch
from torch_geometric.data import Data

from utils import NodeMasking

@pytest.fixture
def test_data():
    # 4-node B-rep graph with 11-dim node features and 8-dim edge features
    x = torch.tensor([
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.0, 1.0, 0.0, 1],
        [1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  1.0, 0.0, 0.0, 2],
        [2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.0, 0.0, 1.0, 1],
        [1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0, 0.0, 0.0, 3],
    ], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 0, 3], [1, 2, 0, 2, 3]], dtype=torch.long)
    edge_attr = torch.tensor([
        [0, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [2, 2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [1, 1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [2, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TestNodeMasking:
    @pytest.fixture(autouse=True)
    def setup(self, test_data):
        self.test_masker = NodeMasking(test_data)
        self.test_data = self.test_masker.idxify(test_data)
        self.test_data = self.test_masker.fully_connect(self.test_data)

    def test_mask_single_node(self):
        # Test masking one node at a time
        for node in range(self.test_data.x.shape[0]):
            masked_datapoint = self.test_masker.mask_node(self.test_data, node)
            self._assert_node_masked(masked_datapoint, node)
            self._assert_edges_masked(masked_datapoint, node)

    def test_demask_single_node(self):
        # Test demasking a single node
        for node in range(self.test_data.x.shape[0]):
            # Mask the node first
            masked_datapoint = self.test_masker.mask_node(self.test_data, node)

            # Retrieve original values for demasking
            demask_value = self.test_data.x[node].clone()   # full [11]-dim feature vector
            connection_types = []
            for i in range(self.test_data.x.shape[0]):
                edge_mask = (self.test_data.edge_index[0] == i) & (self.test_data.edge_index[1] == node)
                connection_types.append(
                    self.test_data.edge_attr[edge_mask, 0].item()  # edge_type (dim 0)
                )
            connection_types = torch.tensor(connection_types)

            # Demask the node
            demasked_datapoint = self.test_masker.demask_node(
                masked_datapoint, node, demask_value, connection_types
            )

            # Check that the node's full feature vector is restored
            assert torch.all(demasked_datapoint.x[node] == demask_value)

            # Check that the edges connected to this node are restored
            for i, edge in enumerate(demasked_datapoint.edge_index.T):
                if edge[0] == node:
                    expected = self.test_data.edge_attr[
                        (self.test_data.edge_index[0] == node) &
                        (self.test_data.edge_index[1] == edge[1]), 0
                    ][0]
                    assert demasked_datapoint.edge_attr[i, 0] == expected
                elif edge[1] == node:
                    expected = self.test_data.edge_attr[
                        (self.test_data.edge_index[1] == node) &
                        (self.test_data.edge_index[0] == edge[0]), 0
                    ][0]
                    assert demasked_datapoint.edge_attr[i, 0] == expected

    def test_remove_node(self):
        # Test removing a node
        for node in range(self.test_data.x.shape[0]):
            removed_datapoint = self.test_masker.remove_node(self.test_data, node)

            # Check if node is removed
            assert removed_datapoint.x.shape[0] == self.test_data.x.shape[0] - 1

            # Check that the remaining nodes are intact
            remaining_nodes = torch.arange(self.test_data.x.shape[0]) != node
            assert torch.all(removed_datapoint.x == self.test_data.x[remaining_nodes])

            # Check that all edge indices are valid (no stale references)
            assert torch.all(removed_datapoint.edge_index[0] < removed_datapoint.x.shape[0])
            assert torch.all(removed_datapoint.edge_index[1] < removed_datapoint.x.shape[0])

            # Check that edge_index and edge_attr dimensions are consistent
            assert removed_datapoint.edge_index.shape[1] == removed_datapoint.edge_attr.shape[0]

    def _assert_node_masked(self, masked_datapoint, node):
        assert masked_datapoint.x[node, 0] == self.test_masker.NODE_MASK

    def _assert_edges_masked(self, masked_datapoint, node):
        assert torch.all(
            masked_datapoint.edge_attr[masked_datapoint.edge_index[0] == node, 0]
            == self.test_masker.EDGE_MASK
        )
        assert torch.all(
            masked_datapoint.edge_attr[masked_datapoint.edge_index[1] == node, 0]
            == self.test_masker.EDGE_MASK
        )

    def test_add_masked_node(self):
        masked_datapoint = self.test_masker.add_masked_node(self.test_data)
        self._assert_new_node_added(masked_datapoint)
        self._assert_new_node_masked(masked_datapoint)
        self._assert_new_node_edges_masked(masked_datapoint)
        self._assert_new_node_connected(masked_datapoint)

    def _assert_new_node_added(self, masked_datapoint):
        assert masked_datapoint.x.shape[0] == self.test_data.x.shape[0] + 1

    def _assert_new_node_masked(self, masked_datapoint):
        assert masked_datapoint.x[-1, 0] == self.test_masker.NODE_MASK

    def _assert_new_node_edges_masked(self, masked_datapoint):
        assert masked_datapoint.edge_attr[-1, 0] == self.test_masker.EDGE_MASK

    def _assert_new_node_connected(self, masked_datapoint):
        last = masked_datapoint.x.shape[0] - 1
        assert torch.all(
            masked_datapoint.edge_attr[
                masked_datapoint.edge_index[0] == last, 0
            ] == self.test_masker.EDGE_MASK
        )

    def test_is_masked(self):
        for node in range(self.test_data.x.shape[0]):
            masked_datapoint = self.test_masker.mask_node(self.test_data, node)
            assert self.test_masker.is_masked(masked_datapoint, node)
            # Demask with zero node type and zero edge types
            demasked_datapoint = self.test_masker.demask_node(
                masked_datapoint, node,
                torch.zeros(1),
                torch.zeros(self.test_data.x.shape[0])
            )
            assert not self.test_masker.is_masked(demasked_datapoint, node)

    def test_generate_fully_masked(self):
        masked_datapoint = self.test_masker.generate_fully_masked(n_nodes=5)
        assert torch.all(masked_datapoint.x[:, 0] == self.test_masker.NODE_MASK)
        assert torch.all(masked_datapoint.edge_attr[:, 0] == self.test_masker.EDGE_MASK)

    def test_remove_empty_edges(self):
        # Mark first two edges as EMPTY
        test_data = self.test_data.clone()
        test_data.edge_attr[0:2, 0] = self.test_masker.EMPTY_EDGE
        removed_datapoint = self.test_masker.remove_empty_edges(test_data)
        assert removed_datapoint.edge_attr.shape[0] == removed_datapoint.edge_index.shape[1]
        assert torch.all(removed_datapoint.edge_attr[:, 0] != self.test_masker.EMPTY_EDGE)


class TestNodeReIndexing:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Use non-consecutive face_type values (0, 4, 18) to test re-indexing
        self.test_data = Data(
            x=torch.tensor([
                [0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 1],
                [4,  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 0.0, 0.0, 2],
                [18, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 1.0, 1],
            ], dtype=torch.float),
            edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long),
            edge_attr=torch.tensor([
                [1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [2, 1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [5, 2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [1, 0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ], dtype=torch.float),
        )
        self.reindexed_data = Data(
            x=torch.tensor([
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 1],
                [1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 0.0, 0.0, 2],
                [2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 1.0, 1],
            ], dtype=torch.float),
            edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long),
            edge_attr=torch.tensor([
                [0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [1, 1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [2, 2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0, 0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ], dtype=torch.float),
        )
        self.masker = NodeMasking(self.test_data)

    def test_idxify(self):
        reindexed_data = self.masker.idxify(self.test_data)
        # Only the type dimensions (dim 0) should be re-indexed
        assert torch.all(reindexed_data.x[:, 0] == self.reindexed_data.x[:, 0]), reindexed_data.x
        assert torch.all(reindexed_data.edge_index == self.reindexed_data.edge_index)

    def test_reindex(self):
        reindexed_data = self.masker.idxify(self.test_data)
        reindexed_data = self.masker.deidxify(reindexed_data)
        assert torch.all(reindexed_data.x[:, 0] == self.reindexed_data.x[:, 0]), reindexed_data.x
        assert torch.all(reindexed_data.edge_index == self.reindexed_data.edge_index)
        assert torch.all(reindexed_data.edge_attr[:, 0] == self.reindexed_data.edge_attr[:, 0]), \
            reindexed_data.edge_attr


def test_fully_connect(test_data):
    masker = NodeMasking(test_data)
    fully_connected_graph = masker.fully_connect(test_data)
    n = test_data.x.shape[0]
    assert fully_connected_graph.edge_index.shape[1] == n ** 2
    # Assert symmetry in the fully connected graph (edge type dim 0 is symmetric)
    for i, j in fully_connected_graph.edge_index.T:
        assert torch.all(
            fully_connected_graph.edge_attr[i * n + j]
            == fully_connected_graph.edge_attr[j * n + i]
        )