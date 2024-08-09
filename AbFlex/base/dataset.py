from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import torch as th
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

from AbFlex.base import utils


class LoopGraphDataSet(Dataset):

    def __init__(self,
                 interaction_dist: float = 10,
                 context_inclusion_dist: float = 10,
                 graph_mode: str = 'loop_context',
                 typing_mode='res_type',
                 edge_encoding=['covalent', 'rbf'],
                 aa_map_mode='extended',
                 cache_frames: bool = False,
                 predict=False,
                 **kwargs,
                 ):

        self.predict = predict
        self.type_map = utils.get_type_map()
        self.protloop_defs = pd.DataFrame()
        self.labels = []
        self.interaction_dist = interaction_dist
        self.context_inclusion_dist = context_inclusion_dist
        self.typing_mode = typing_mode
        self.graph_mode = graph_mode
        self.cache = {}
        self.cache_frames = cache_frames
        self.edge_encoding = edge_encoding
        self.graph_generation_function_dict = {  # options for graph generation
            'loop_context': self._get_loop_context_graph,
            'loop': self._get_loop_graph,
        }

        if aa_map_mode == 'standard':
            self.aa_map = utils.standard_aa_map
        elif aa_map_mode == 'extended':
            self.aa_map = utils.extended_aa_map

    def _get_loop_context_graph(self,
                                prot_df: pd.DataFrame,
                                loop_chain: str,
                                loop_resi: list):
        """Return graph composed of nodes in loops and surrounding context
        nodes

        Args:
            prot_df (pd.DataFrame): Protein dataframe
            loop_chain (str): Chain of loop
            loop_resi (list): Residue numbers of loop
        """
        # obtain coordinates of the loop and its context
        loop_nodes = prot_df.loc[
            (prot_df["chain_id"] == loop_chain) &
            (prot_df["residue_number"].isin(
                np.arange(loop_resi[0], loop_resi[1]+1)
            ))
        ]
        loop_coords = loop_nodes.loc[
            :, ['x_coord', 'y_coord', 'z_coord']].to_numpy(
            dtype=np.float64)
        context_nodes = prot_df.loc[
            ~((prot_df["chain_id"] == loop_chain) &
              (prot_df["residue_number"].isin(
                np.arange(loop_resi[0], loop_resi[1]+1)
              ))), :]
        context_coords = context_nodes.loc[
            :, ['x_coord', 'y_coord', 'z_coord']].to_numpy(
                dtype=np.float64)

        # ID NODES #
        # dist between loop and points in context
        dist_loop_context = cdist(loop_coords, context_coords)

        # identifying nodes within distance of cdr
        _, neighbours_context = np.where(
            dist_loop_context < self.context_inclusion_dist
        )
        context_node_idx = sorted(np.unique(neighbours_context))
        context_nodes = context_nodes.iloc[context_node_idx, :]
        context_coords = context_coords[context_node_idx, :]

        # FEATURE VECTOR #
        # features: one-hot encoded lmg atom/res type
        loop_node_features = self._get_node_features(loop_nodes)
        context_node_features = self._get_node_features(context_nodes)

        # is the node in the cdr or fw?
        loop_node_bool = np.zeros((len(loop_coords), 1))
        context_node_bool = np.ones((len(context_coords), 1))

        # features: node xyz coordinates [cols 0-2], one-hot encoded res types
        # [cols 3-13], label of whether loop/context [col 14]
        feats_loop = np.concatenate(
            [loop_coords, loop_node_features, loop_node_bool], axis=-1)
        feats_context = np.concatenate(
            [context_coords, context_node_features, context_node_bool],
            axis=-1)

        feats = np.concatenate([feats_loop, feats_context], axis=-2)

        # EDGES #
        edge_indices, edge_attr_full = self._edge_encoding(
            loop_nodes,
            loop_coords,
            context_nodes=context_nodes,
            context_coords=context_coords,
            encodings=self.edge_encoding)

        return feats, edge_indices, edge_attr_full

    def _get_loop_graph(self, prot_df: pd.DataFrame, loop_chain: str,
                        loop_resi: list):
        """Return graph composed of nodes in loop

        Args:
            prot_df (pd.DataFrame): Protein dataframe
            loop_chain (str): Chain of loop
            loop_resi (list): Residue numbers of loop
        """
        # obtain coordinates of the cdr and framework
        loop_nodes = prot_df.loc[
            (prot_df["chain_id"] == loop_chain) &
            (prot_df["residue_number"].isin(
                np.arange(loop_resi[0], loop_resi[1]+1)
            ))
        ]
        loop_coords = loop_nodes.loc[
            :, ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)

        # FEATURE VECTOR #
        # features: one-hot encoded lmg atom/res type
        loop_node_features = self._get_node_features(loop_nodes)

        # features: node xyz coordinates [cols 0-2],
        # one-hot encoded res types [cols 3-13]
        feats = np.concatenate([loop_coords, loop_node_features], axis=-1)

        # EDGES #
        edge_indices, edge_attr_full = self._edge_encoding(
            loop_nodes, loop_coords, encodings=self.edge_encoding
        )

        return feats, edge_indices, edge_attr_full

    def _get_node_features(self, df: pd.DataFrame):
        mode = self.typing_mode

        if mode == 'lmg':
            types = df['lmg_types']
            types = types.apply(lambda x: self.type_map[x])
            types = np.array(types)
            types = utils.get_one_hot(
                types, nb_classes=max(self.type_map.values()) + 1
            )
            return types
        elif mode == 'res_type':
            types = df['residue_name']
            types = types.apply(
                lambda x: self.aa_map[x] if x in self.aa_map.keys() else 20)
            types = types.astype(np.int64)
            types = np.array(types)
            types = utils.get_one_hot(
                types, nb_classes=max(self.aa_map.values()) + 1
            )
            return types
        else:
            raise NotImplementedError(mode)

    def _edge_attr_intra_inter(self, intra_dst_loop, intra_dst_context,
                               inter_dst_context):
        '''one-hot encoding intra/inter'''
        edge_attr = np.concatenate([
            np.ones(len(intra_dst_loop)),
            np.ones(len(intra_dst_context)),
            np.zeros(len(inter_dst_context)),
            np.zeros(len(inter_dst_context))
        ])
        return np.expand_dims(edge_attr, 1)

    def _edge_attr_intraLOOP_intraCONTEXT_inter(self,
                                                intra_dst_loop,
                                                intra_dst_context,
                                                inter_dst_context):
        '''3 class one-hot encoding for intra_loop, intra_context,
        inter_context_loop
        '''
        edge_attr = np.zeros(
            (len(intra_dst_loop)+len(intra_dst_context) +
             len(inter_dst_context)+len(inter_dst_context), 3))
        start = len(intra_dst_loop)
        end = len(intra_dst_loop)+len(intra_dst_context)
        edge_attr[:start, 0] = 1
        edge_attr[start:end, 1] = 1
        edge_attr[end:, 2] = 1
        return edge_attr

    def _edge_attr_covalent(self, loop_nodes, context_nodes, intra_src_loop,
                            intra_dst_loop, intra_src_context,
                            intra_dst_context, inter_src_loop,
                            inter_dst_context):
        '''one-hot encoding for covalent bonds'''
        # 1 intra_loop
        interval = np.abs(
            loop_nodes.iloc[intra_src_loop].residue_number.values -
            loop_nodes.iloc[intra_dst_loop].residue_number.values
        )
        idx = np.argwhere(interval == 1)
        intra_loop_covalent = np.zeros_like(intra_src_loop)
        intra_loop_covalent[idx] = 1

        if self.graph_mode == 'loop_context':
            # 2 intra_context
            interval = np.abs(
                context_nodes.iloc[intra_src_context].residue_number.values -
                context_nodes.iloc[intra_dst_context].residue_number.values
                )
            idx = np.argwhere(interval == 1)
            intra_context_covalent = np.zeros_like(intra_src_context)
            intra_context_covalent[idx] = 1

            # 3 inter
            interval = np.abs(
                loop_nodes.iloc[inter_src_loop].residue_number.values -
                context_nodes.iloc[inter_dst_context].residue_number.values
                )
            idx = np.argwhere(interval == 1)
            inter_covalent = np.zeros_like(inter_src_loop)
            inter_covalent[idx] = 1

            edge_attr = np.concatenate([
                intra_loop_covalent,
                intra_context_covalent,
                inter_covalent,
                inter_covalent
                ])

        elif self.graph_mode == 'loop':
            edge_attr = intra_loop_covalent

        else:
            raise NotImplementedError(self.graph_mode)

        return np.expand_dims(edge_attr, 1)

    def _rbf(self, distances, count=8):
        '''Encodes a distance array in a radial basis function representation
        with n=count basis functions
        '''
        min, max = 0., 10
        mu = np.linspace(min, max, count)
        sigma = (max - min) / count
        distances = distances.reshape(-1, 1)
        return np.exp(-((distances - mu) / sigma)**2)

    def _edge_encoding(self, loop_nodes, loops_coords, context_nodes=None,
                       context_coords=None, encodings=None):
        '''Generate edge indices and edge attributes'''
        # IDx EGDES #
        # 1 intra_loop: interactions between loop nodes
        dist_intra_loop = cdist(loops_coords, loops_coords)
        intra_src_loop, intra_dst_loop = np.where(
            dist_intra_loop < self.interaction_dist
        )

        if self.graph_mode == 'loop_context':
            # 2 intra_context: interactions between context nodes
            dist_intra_context = cdist(context_coords, context_coords)
            intra_src_context, intra_dst_context = np.where(
                dist_intra_context < self.interaction_dist
            )

            # 3 inter: interactions between loop & context
            dist_inter = cdist(loops_coords, context_coords)
            inter_src_loop, inter_dst_context = np.where(
                dist_inter < self.interaction_dist
            )

            context_offset = len(loop_nodes)

            # edge source and destination nodes
            edge_src = np.concatenate([
                [i for i in intra_src_loop],
                [i + context_offset for i in intra_src_context],
                [i for i in inter_src_loop],
                # make inter edges bidriectional
                [i + context_offset for i in inter_dst_context]
            ])
            edge_dst = np.concatenate([
                [i for i in intra_dst_loop],
                [i + context_offset for i in intra_dst_context],
                [i + context_offset for i in inter_dst_context],
                [i for i in inter_src_loop]  # make inter edges bidirectional
            ])

            dist = np.concatenate(
                [dist_intra_loop[intra_src_loop, intra_dst_loop],
                 dist_intra_context[intra_src_context, intra_dst_context],
                 dist_inter[inter_src_loop, inter_dst_context],
                 dist_inter[inter_src_loop, inter_dst_context]])

        elif self.graph_mode == 'loop':
            intra_src_context, intra_dst_context = [], []
            inter_src_loop, inter_dst_context = [], []

            edge_src = intra_src_loop
            edge_dst = intra_dst_loop

            dist = dist_intra_loop[intra_src_loop, intra_dst_loop]

        else:
            raise NotImplementedError(self.graph_mode)

        edge_indices = np.vstack([edge_src, edge_dst]).astype(np.float64)

        # EDGE ATTRIBUTES #
        edge_attrs = []
        if 'intra/inter' in encodings:
            edge_attrs.append(self._edge_attr_intra_inter(
                intra_dst_loop, intra_dst_context, inter_dst_context
            ))
        if 'intraCONTEXT/intraLOOP/inter' in encodings:
            edge_attrs.append(self._edge_attr_intraLOOP_intraCONTEXT_inter(
                intra_dst_loop, intra_dst_context, inter_dst_context
            ))
        if 'covalent' in encodings:
            edge_attrs.append(self._edge_attr_covalent(
                loop_nodes, context_nodes, intra_src_loop, intra_dst_loop,
                intra_src_context, intra_dst_context, inter_src_loop,
                inter_dst_context
            ))
        if 'rbf' in encodings:
            edge_attrs.append(self._rbf(dist))

        edge_attrs = np.concatenate(edge_attrs, 1)
        return edge_indices, edge_attrs

    def _generate_graph_dict(self, prot_df: pd.DataFrame, loop_chain: str,
                             loop_resi: list):
        """Generate dictionary of nodes, edge indices and edge attributes
        in graph
        """
        graph_dict = {}
        graph_func = self.graph_generation_function_dict[self.graph_mode]
        nodes, edge_ind, edge_attr = graph_func(prot_df, loop_chain, loop_resi)

        graph_dict['nodes'] = nodes
        graph_dict['edge_ind'] = edge_ind
        graph_dict['edge_attr'] = edge_attr

        return graph_dict

    def _parse_graph(self, graph_dict: dict, label: np.ndarray,
                     protloop_def: dict):
        """Generate parsed graph object in correct format. This is intended to
        be overwritten by child classes depending on requirements of the
        downstream models. Base implementation parses the data into a pytorch
        geometric data instance.

        Args:
            graph_dict: output of _generate_graph_dict, contains nodes
                        (coordinates & features), edge indices and edge
                        attributes of graph
        """
        # remove self loops
        edge_index, edge_attr = remove_self_loops(
                edge_index=th.from_numpy(graph_dict['edge_ind']).long(),
                edge_attr=th.from_numpy(graph_dict['edge_attr'])
            )

        if self.predict:
            graph = Data(
                x=th.from_numpy(graph_dict['nodes'][:, 3:]),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=th.from_numpy(graph_dict['nodes'][:, :3]),
                pdb_file=str(protloop_def['pdb'])
            )

        else:
            graph = Data(
                x=th.from_numpy(graph_dict['nodes'][:, 3:]),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=th.from_numpy(graph_dict['nodes'][:, :3]),
                y=th.tensor(label),
                pdb_file=str(protloop_def['pdb'])
            )

        return graph

    def __len__(self):
        """ Get number of entries in dataset """
        return len(self.labels)

    def populate(self, input_file: Path, overwrite: bool = False):
        """Extract information from input files and save in protloop_defs list
        of lists

        Args:
            input_file: cvs containing columns: labels, pdb [path to file],
            ab_chains, chain of selected loop, resi of selected loop
        """

        protloop_defs = pd.read_csv(input_file)
        if self.predict:
            # empty list of labels at inference
            labels = [None for _ in range(len(protloop_defs))]
        else:
            labels = protloop_defs["labels"].to_list()

        # either overwrite or add to self.protloop_defs and self.labels
        if overwrite:
            self.protloop_defs = protloop_defs
            self.labels = labels
        else:
            if len(self.protloop_defs) > 0:
                self.protloop_defs += protloop_defs
            else:
                self.protloop_defs = protloop_defs
            if len(self.labels) != 0:
                self.labels = self.labels + labels
            else:
                self.labels = labels

    def __getitem__(self, idx: int, force_recalc: bool = False):
        """ Generate graph for complex in dataset

        Args:
            idx: index
        """
        # read in pdbs #
        # obtain label and ppi complex info (path to pdb file, chains in ab)
        label = self.labels[idx]
        protloop_def = self.protloop_defs.iloc[idx]

        # generate path to typed parquet file
        typed_pdb = Path(str(protloop_def['pdb']).replace('.pdb', '.parquet'))

        # check if typed file in cache
        if (self.cache_frames and
                str(typed_pdb) in self.cache and not
                force_recalc):
            pdb_df = self.cache[str(typed_pdb)].copy()

        # check if typed file exists
        elif typed_pdb.exists() and not force_recalc:
            pdb_df = pd.read_parquet(typed_pdb)

        # if not create and save typed files
        else:
            pdb_df = utils.parse_pdb_to_parquet(
                protloop_def['pdb'], typed_pdb, lmg_typed=False, ca=True
            )

        if self.cache_frames:
            if not str(typed_pdb) in self.cache:
                self.cache[str(typed_pdb)] = pdb_df.copy()

        # generate graphs #
        # get selected chains
        protein_chains = []
        for ch in protloop_def['ab_chains']:
            protein_chains.append(ch)

        prot_df = pdb_df[pdb_df['chain_id'].isin(protein_chains)]

        # generate graph
        graph_dict = self._generate_graph_dict(
            prot_df,
            loop_chain=protloop_def['chain'],
            loop_resi=[protloop_def['resi_start'], protloop_def['resi_end']]
        )
        graph = self._parse_graph(
            graph_dict, label, protloop_def=protloop_def
        )

        return graph
