from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import torch
from torch import nn
from torch.nn import Linear, BCEWithLogitsLoss, ModuleList
from torch.nn.functional import relu
from torch import sigmoid
from torch import optim
import pytorch_lightning as pl
from torch_geometric.utils import dropout_edge
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from AbFlex.base.dataset import LoopGraphDataSet
from AbFlex.models.graphnorm.graphnorm import GraphNorm
from AbFlex.models.egnn.egnn import E_GCL, EGNN


def to_np(x):
    return x.cpu().detach().numpy()


class flexEGNN(pl.LightningModule):
    def __init__(
        self,
        loader_config: dict,
        dataset_config: dict,
        trainer_config: dict,
        num_edge_features: int = 9,
        embedding_in_nf: int = 128, # num features of embedding in
        embedding_out_nf: int = 128, # num features of embedding out
        egnn_layer_hidden_nfs: list = [128,128,128], # number and dimension of hidden layers
        num_classes: int=1, # dimension of output
        opt: str='adam',
        loss: str='bce_logits',
        scheduler: str=None,
        lr: float=10e-3,
        dropout: float=0.0,
        balanced_loss: bool = False,
        attention: bool = False,
        residual: bool = True,
        normalize: bool = False,
        tanh: bool = False,
        update_coords: bool = True,
        weight_decay: float = 0,
        norm: str = None,
        norm_nodes: str = None,
        pooling: str = 'max',
        pool_first: bool = True, # pool before or after passing to last linear layer
        reload_best_model: bool = False,
        save_dir: str = None,
        **kwargs,
    ):
        super(flexEGNN, self).__init__()

        # load parameters from yaml config file
        self.loader_config = loader_config
        self.dataset_config = dataset_config
        self.trainer_config = trainer_config
        self.update_coords = update_coords
        self.reload_best_model = reload_best_model
        self.save_dir = save_dir

        try: self.masking_bool = self.dataset_config['masking']
        except: self.masking_bool = False

        # this is for the inheriting models
        self.embedding_out_nf = embedding_out_nf
        self.num_classes = num_classes

        # pooling
        self.pool_first = pool_first
        if pooling == 'max':
            self.pooling_fn = global_max_pool
        elif pooling == 'mean':
            self.pooling_fn = global_mean_pool
        else:
            raise NotImplementedError('Pooling function not implemented')

        num_node_features = 22 if dataset_config['graph_generation_mode'] == 'loop_context' else 21
        self.embedding_in = Linear(num_node_features, embedding_in_nf)
        self.embedding_out = Linear(embedding_in_nf, embedding_out_nf)

        if pool_first:
            self.post_pool_linear = Linear(embedding_out_nf, num_classes)
        else:
            self.pre_pool_linear = Linear(embedding_out_nf, num_classes)
        self.dropout = dropout

        egnn_layers = []
        for hidden_nf in egnn_layer_hidden_nfs:
            layer = E_GCL(
                embedding_in_nf, 
                hidden_nf, 
                embedding_in_nf, 
                edges_in_d=num_edge_features,
                act_fn=nn.SiLU(),
                attention=attention,
                residual=residual,  # if True, in and out nf need to be equal
                normalize=normalize,
                coords_agg='mean',
                tanh=tanh,
                norm_nodes=norm_nodes,
            )
            egnn_layers.append(layer)
        self.egnn_layers = ModuleList(egnn_layers)

        # setup of training environment
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        if loss == 'bce_logits':
            self.loss_fn = BCEWithLogitsLoss()
        else:
            raise NotImplementedError
        if balanced_loss:
            raise NotImplementedError

        self.test_set_predictions = []

        self.norm_nodes = norm_nodes
        if norm_nodes:
            self.graphnorm = GraphNorm(embedding_out_nf)

        # empty lists for summary statistics
        self.preds = []
        self.targets = []
        self.val_roc_aucs = []
        self.val_prc_aucs = []
        self.val_epochs = []


    def forward(self, graph):
        nodes = graph.x.float() # node features
        edge_ind = graph.edge_index
        coords = graph.pos.float() # coords
        edge_attr = graph.edge_attr.float()

        nodes = self.embedding_in(nodes)

        for egnn_layer in self.egnn_layers:
            edge_ind_post_dropout, edge_mask = dropout_edge(edge_ind, p=self.dropout, training=self.training) # randomly drops edges from adjacency matrix; default dropout probability (p) set to 0
            edge_attr_post_dropout = edge_attr[edge_mask]

            # update node features (and, if update_coords=True, coordinates)
            if self.update_coords:
                nodes, coords, _ = egnn_layer(nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout, batch=graph.batch)
            else:
                nodes, _, _ = egnn_layer(nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout, batch=graph.batch)

        if self.norm_nodes:
            nodes = self.graphnorm(relu(self.embedding_out(nodes)), graph.batch)

        if self.pool_first:
            graph_vector = self.pooling_fn(nodes, graph.batch)
            out = self.post_pool_linear(graph_vector)
        else:
            nodes = self.pre_pool_linear(nodes)
            out = self.pooling_fn(nodes, graph.batch)

        return out


    def training_step(self, batch, batch_idx):
        y = batch.y # true label

        batch_idx = sorted(batch.batch.unique())
        batch_repl = {}

        for i, b in enumerate(batch_idx):
            batch_repl[b.item()] = i
        for i in range(len(batch.batch)):
            b = batch.batch[i].item()
            batch.batch[i] = batch_repl[b]

        pred = self.forward(batch)

        if y.shape != pred.shape:
            try:
                y = y.view_as(pred)
            except:
                pass

        loss = self.loss_fn(pred.float(), y.float())
        self.log('Loss/train', loss, on_step=False, on_epoch=True, batch_size=self.loader_config['batch_size'])
        pred = sigmoid(pred) # sigmoid activation function for prediction

        self.preds.append(pred.detach())
        self.targets.append(y.detach())

        return {'loss': loss, 'pred': pred, 'y': y}
    

    def on_train_epoch_end(self):
        roc, pr_auc = self.epoch_metrics(self.preds, self.targets)
        self.log('roc_auc/train', roc, on_step=False, on_epoch=True)
        self.log('pr_auc/train', pr_auc, on_step=False, on_epoch=True)

        self.preds = []
        self.targets = []


    def on_train_end(self):
        self.log_best_model_val_metrics()
        self.log('model/n_trainable_params', sum(p.numel() for p in self.parameters() if p.requires_grad), on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        y = batch.y # true label
        
        batch_idx = sorted(batch.batch.unique())
        batch_repl = {}
        for i, b in enumerate(batch_idx):
            batch_repl[b.item()] = i
        for i in range(len(batch.batch)):
            b = batch.batch[i].item()
            batch.batch[i] = batch_repl[b]

        y = torch.masked_select(y, y != -1)
        pred = self.forward(batch)

        if y.shape != pred.shape:
            y = y.view_as(pred)
        loss = self.loss_fn(pred.float(), y.float())
        self.log('Loss/val', loss, on_step=False, on_epoch=True, batch_size=self.loader_config['batch_size'])
        pred = sigmoid(pred)

        self.preds.append(pred.detach())
        self.targets.append(y.detach())

        return {'loss': loss, 'pred': pred, 'y': y}


    def on_validation_epoch_end(self):
        roc, pr_auc = self.epoch_metrics(self.preds, self.targets)
        self.log('roc_auc/val', roc, on_step=False, on_epoch=True)
        self.log('pr_auc/val', pr_auc, on_step=False, on_epoch=True)

        self.val_roc_aucs.append(roc)
        self.val_prc_aucs.append(pr_auc)
        self.val_epochs.append(self.current_epoch)

        self.preds = []
        self.targets = []


    def test_step(self, batch, batch_idx):
        y = batch.y # true label

        batch_idx = sorted(batch.batch.unique())
        batch_repl = {}
        for i, b in enumerate(batch_idx):
            batch_repl[b.item()] = i
        for i in range(len(batch.batch)):
            b = batch.batch[i].item()
            batch.batch[i] = batch_repl[b]

        pred = self.forward(batch)

        if y.shape != pred.shape:
            y = y.view_as(pred)
        loss = self.loss_fn(pred.float(), y.float())
        pred = sigmoid(pred)

        self.preds.append(pred.detach())
        self.targets.append(y.detach())

        # save predicted output
        output_preds = []
        labels = to_np(y).flatten()
        for ind, score in enumerate(to_np(pred)):
            if self.masking_bool:
                output_preds.append((batch.pdb_file[ind], batch.masked_cdrh3_res[ind], score, labels[ind]))
            else:
                output_preds.append((batch.pdb_file[ind], score, labels[ind]))
        self.test_set_predictions += output_preds

        return {'loss': loss, 'pred': pred, 'y': y}
    

    def on_test_epoch_end(self):
        print(self.preds, self.targets)
        roc, pr_auc = self.epoch_metrics(self.preds, self.targets)
        self.log(f'roc_auc/test', roc, on_step=False, on_epoch=True)
        self.log(f'pr_auc/test', pr_auc, on_step=False, on_epoch=True)

        self.preds = []
        self.targets = []


    def predict_step(self, batch):
        pred = self.forward(batch)
        pred = sigmoid(pred)
        return to_np(pred)


    def epoch_metrics(self, predictions, targets):
        preds = to_np(torch.cat(predictions, dim=0))
        targets = to_np(torch.cat(targets, dim=0))
    
        try:
            roc = roc_auc_score(y_true=targets, y_score=preds)
            precision, recall, thresholds = precision_recall_curve(y_true=targets, y_score=preds)
            pr_auc = auc(recall, precision)

        except:
            roc = None
            pr_auc = None
        
        return roc, pr_auc


    def log_best_model_val_metrics(self):
        """Log best validation epoch and metrics"""
        best_ind = np.argmax(self.val_prc_aucs)
        self.logger.experiment.log({
            'epoch_best': self.val_epochs[best_ind],
            'pr_auc/val_best': self.val_prc_aucs[best_ind],
            'roc_auc/val_best': self.val_roc_aucs[best_ind],
        })


    def save_test_predictions(self, filename: Path):
        """
        Save test predictions to csv file
        
        Args:
            filename (Path): path to output file
        """
        with open(filename, 'w') as outf:
            outf.write('pdb,pred_score,true_label\n')
            for p in self.test_set_predictions:
                outf.write(f'{p[0]},{p[1]},{p[2]}\n')

                
    def configure_optimizers(self):
        if self.opt == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        if not self.scheduler:
            return self.optimizer
        else:
            # define scheduler
            if self.scheduler == 'CosineAnnealingWarmRestarts':
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    self.trainer_config['max_epochs'],
                    eta_min=1e-4)
            elif self.scheduler == 'CosineAnnealing':
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    self.trainer_config['max_epochs'])
            else: 
                raise NotImplementedError
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler,
            }
    

    def test_dataloader(self):
        if self.dataset_config['input_files']['test'] is None:
            return None

        ds = LoopGraphDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            edge_encoding=self.dataset_config['edge_encoding'],
            aa_map_mode=self.dataset_config['aa_map_mode'],
            cache_frames=self.dataset_config['cache_frames'],
            masking=self.masking_bool
        )

        for f in self.dataset_config['input_files']['test']:
            ds.populate(f, overwrite=False)
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'], 
            shuffle=False,
            num_workers=self.loader_config['num_workers'])
        return loader

    
    def train_dataloader(self):
        if self.dataset_config['input_files']['train'] is None:
            return None
        
        ds = LoopGraphDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            edge_encoding=self.dataset_config['edge_encoding'],
            aa_map_mode=self.dataset_config['aa_map_mode'],
            cache_frames=self.dataset_config['cache_frames'],
            masking=self.masking_bool,
        )

        for f in self.dataset_config['input_files']['train']:
            ds.populate(f, overwrite=False)

        if self.loader_config['balanced_sampling']:
            raise NotImplementedError('Balanced sampling not implemented.')
        else:
            sampler = None
            shuffle = False

        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'],  
            shuffle=shuffle, 
            num_workers=self.loader_config['num_workers'],
            sampler=sampler)
        
        return loader


    def val_dataloader(self):
        if self.dataset_config['input_files']['val'] is None:
            return None
                
        ds = LoopGraphDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            edge_encoding=self.dataset_config['edge_encoding'],
            aa_map_mode=self.dataset_config['aa_map_mode'],
            cache_frames=self.dataset_config['cache_frames'],
            masking=self.masking_bool,
        )

        for f in self.dataset_config['input_files']['val']:
            ds.populate(f, overwrite=False)
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'], 
            shuffle=False,
            num_workers=self.loader_config['num_workers'])
        return loader


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
