'''Test LoopGraphDataSet class'''
import unittest
import numpy as np
import sys
sys.path.append('../')
from AbFlex.base.dataset import LoopGraphDataSet  # noqa


class TestLoopContext(unittest.TestCase):

    def test_node_features(self):
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop_context')
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]

        self.assertEqual(graph.x.shape, (8, 22))
        # 6 framework residue node
        self.assertEqual(graph.x.sum(axis=0)[-1], 6)
        # correct sequence
        self.assertEqual(graph.x.sum(axis=0)[:21].tolist(),
                         [1, 0, 0, 1, 1, 1, 0, 0, 0, 1,
                          0, 1, 0, 2, 0, 0, 0, 0, 0, 0,
                          0])

    def test_edge_features_dimensions(self):
        # test correct shape of edge features

        # all encodings
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop_context',
                              edge_encoding=['intra/inter',
                                             'intraCONTEXT/intraLOOP/inter',
                                             'covalent',
                                             'rbf'])
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]
        self.assertEqual(graph.edge_attr.shape[1], 13)

        # single encodings
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop_context',
                              edge_encoding=['intra/inter'])
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]
        self.assertEqual(graph.edge_attr.shape[1], 1)

        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop_context',
                              edge_encoding=['intraCONTEXT/intraLOOP/inter'])
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]
        self.assertEqual(graph.edge_attr.shape[1], 3)

        # combinations
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop_context',
                              edge_encoding=['covalent',
                                             'rbf'])
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]
        self.assertEqual(graph.edge_attr.shape[1], 9)

    def test_edge_features(self):
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop_context',
                              edge_encoding=['intra/inter',
                                             'intraCONTEXT/intraLOOP/inter',
                                             'covalent',
                                             'rbf'])
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]

        # edges are bidirectional
        for i in range(ds[0].edge_index.max().item()):
            self.assertTrue(
                (ds[0].edge_attr[np.where(ds[0].edge_index[0] == i)] ==
                 ds[0].edge_attr[np.where(ds[0].edge_index[1] == i)]).all())

        # correct encoding
        # intra/inter
        self.assertEqual(graph.edge_attr[:2].sum(axis=0)[0], 2)
        self.assertEqual(graph.edge_attr[2:32].sum(axis=0)[0], 30)
        self.assertEqual(graph.edge_attr[32:].sum(axis=0)[0], 0)
        # intra/intra/inter
        self.assertEqual(
            graph.edge_attr[:2].sum(axis=0)[1:4].tolist(), [2., 0., 0.]
            )
        self.assertEqual(
            graph.edge_attr[2:32].sum(axis=0)[1:4].tolist(), [0., 30., 0.]
            )
        self.assertEqual(
            graph.edge_attr[32:].sum(axis=0)[1:4].tolist(), [0., 0., 20.]
            )
        # covalent
        self.assertEqual(graph.edge_attr.sum(axis=0)[4], 14)
        # all
        self.assertEqual(
            graph.edge_attr.sum(axis=0)[:5].tolist(), [32, 2, 30, 20, 14]
            )

    def test_rbf_encodings(self):
        # test correct rbf distance encoding
        distances = np.arange(0, 11, 1)
        ds = LoopGraphDataSet()
        distance_encodings = ds._rbf(distances, count=11)
        # diagonal should be 1
        distance_encodings = np.diag(distance_encodings)
        self.assertTrue((distance_encodings == np.ones(11)).all())

    def test_predict(self):
        # test correct prediction behavior
        ds = LoopGraphDataSet(predict=True)
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]
        assert graph.y is None


class TestLoop(unittest.TestCase):

    def test_node_features(self):
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop')
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]

        self.assertEqual(graph.x.shape, (2, 21))

    def test_edge_features(self):
        ds = LoopGraphDataSet(interaction_dist=10,
                              graph_mode='loop',
                              edge_encoding=['intra/inter',
                                             'intraCONTEXT/intraLOOP/inter',
                                             'covalent',
                                             'rbf'])
        ds.populate(input_file='example_dataset.csv')
        graph = ds[0]

        # correct number of edge features
        self.assertEqual(graph.edge_attr.shape[1], 13)

        # edges are bidirectional
        self.assertTrue((graph.edge_attr[0] == graph.edge_attr[1]).all())

        # correct encoding
        # all
        self.assertEqual(
            graph.edge_attr.sum(axis=0)[:5].tolist(), [2, 2, 0, 0, 2]
            )
