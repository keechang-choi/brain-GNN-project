from typing import Union, List, Dict

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.utils import softmax
import pandas as pd
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np
import os


from torch_geometric.nn import (GraphConv, SAGPooling, global_mean_pool,
                                JumpingKnowledge)



class GlobalAtt_gate(GlobalAttention):
    def __init__(self, h):
        super().__init__(h)

    def get_att(self, x):
        """ x for GCN result"""
        batch = torch.zeros(x.size(0), dtype = torch.long)
        size = batch[-1].item() + 1
        gate = self.gate_nn(x).view(-1,1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = softmax(gate, batch, num_nodes=size)
        return gate

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features = 149, num_classes = 3):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.att = GlobalAtt_gate(Linear(hidden_channels, 1))
        #self.att = GlobalAttention(Linear(hidden_channels, 1))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.att(x, batch)
        x = self.lin1(x)
        x = x.relu()

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
    def get_att(self,x,edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        out = self.att.get_att(x)
        return out

class TimeDiffClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #self.hparams['num_node_features'] ,
        self.model = GCN(hidden_channels=self.hparams['hidden_channels'], num_classes=self.hparams['num_classes'])

    def edge2mat(self, edge_index, edge_attr, n=148):
        efn = edge_attr.shape[-1]
        mat = torch.zeros([n, n, efn])
        for eidx, ef in zip(edge_index.T, edge_attr):
            mat[eidx[0], eidx[1]] = ef
        return mat

    def mat2edge(self, mat):
        n = mat.shape[0]
        efn = mat.shape[-1]
        edge_index = []
        edge_attr = []
        for i in range(n):
            for j in range(n):
                if mat[i, j] != torch.zeros(efn):
                    edge_index.append([i, j])
                    edge_attr.append(mat[i, j])
        edge_index = torch.tensor(edge_index).T
        edge_attr = torch.tensor(edge_attr).unsqueeze(-1)
        #edge_attr.shape
        return (edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr, batch):
        y_pred = self.model(x, edge_index, edge_attr, batch)
        return y_pred

    def general_step(self, batch,batch_idx, mode):
        batch = batch.to(self.device)
        y = batch.y
        y_pred = self.model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
        loss = F.cross_entropy(y_pred, y)

        preds = y_pred.argmax(axis = 1)
        n_correct = (y == preds).sum()
        return loss, n_correct
    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss, acc = self.general_end(outputs, "test")
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    def generate_data(self):
        s = '../data/DT_File_MAP.xlsx'
        # df = pd.read_excel(osp.join(*(s.split('/'))))
        df = pd.read_excel(self.hparams['s_node'])
        s = '../data/[UNC]ADNI-network/dataTS.csv'
        # df_ts = pd.read_csv(osp.join(*(s.split('/'))))
        df_ts = pd.read_csv(self.hparams['s_date'])

        df_m = df[["Subject", "VISCODE"]].merge(df_ts, how='inner', left_on="Subject", right_on="subject")
        df_m['date'] = pd.to_datetime(df_m['EXAMDATE.x'], format='%m/%d/%Y')
        df_ms = df_m.sort_values(["PTID", "date"])[["Subject", "PTID", "date", "VISCODE_x", "VISCODE_y", "DX_bl"]]
        df_msl = df_ms[1:].reset_index(drop=True)
        df_diff = df_msl.merge(df_ms, on=None, left_index=True, right_index=True, how="right", suffixes=('_l', '_e'))
        df_diff = df_diff[df_diff['PTID_l'] == df_diff['PTID_e']]
        df_diff = df_diff.reset_index(drop=True)
        df_diff['interval_m'] = df_diff['VISCODE_x_l'] - df_diff['VISCODE_x_e']
        df_diff['interval_d'] = (df_diff['date_l'] - df_diff['date_e']).dt.days
        labels = {}
        if (self.hparams['num_classes'] == 4):

            labels['CN'] = 0
            labels['EMCI'] = 1
            labels['LMCI'] = 2
            labels['AD'] = 3
        elif self.hparams['num_classes'] == 3:
            labels['CN'] = 0
            labels['EMCI'] = 1
            labels['LMCI'] = 2
            labels['AD'] = 2
        elif self.hparams['num_classes'] == 2:
            labels['CN'] = 0
            labels['AD'] = 1
        s = '../sparsification/data'
        # filepath_r = osp.join(*(s.split('/')))
        filepath_r = self.hparams['s_sparsified']
        data_dict = {}
        for i in range(df.shape[0]):
            label = df.loc[i]["LABEL"]
            if (label not in labels.keys()):
                continue
            y = labels[label]
            # filename = osp.join(filepath_r, "%s_fdt_network_matrix_df.csv" % df.loc[i]["Subject"])
            filename = filepath_r + "%s_fdt_network_matrix_df.csv" % df.loc[i]["Subject"]
            sbj = df.loc[i]['Subject']
            edge_table = pd.read_csv(filename, sep="\t")
            edge_table = edge_table[["src", "trg", "nij"]]
            edge_index = torch.tensor(edge_table[["src", "trg"]].values).transpose(0, 1)
            edge_attr = torch.tensor(edge_table[["nij"]].values, dtype=torch.float)
            x = torch.tensor(df[df.index == i][["Node %d" % (i) for i in range(1, 149)]].values,
                             dtype=torch.float).transpose(0, 1)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            data_dict[sbj] = data

        datalist = []
        for i in range(df_diff.shape[0]):
            label = df_diff.loc[i]['DX_bl_e']
            if (label not in labels.keys()):
                continue
            y = labels[label]
            print('data processing %d for label %d' % (i, y))
            sbj_l = df_diff.loc[i]['Subject_l']
            sbj_e = df_diff.loc[i]['Subject_e']
            d_l = data_dict[sbj_l]
            d_e = data_dict[sbj_e]

            x_l = d_l.x
            edge_index_l = d_l.edge_index
            edge_attr_l = d_l.edge_attr

            x_e = d_e.x
            edge_index_e = d_e.edge_index
            edge_attr_e = d_e.edge_attr

            mat_l = self.edge2mat(edge_index_l, edge_attr_l)
            mat_e = self.edge2mat(edge_index_e, edge_attr_e)
            # print(mat_l.shape, mat_e.shape)
            # print(edge_attr_e.shape)
            mat = (mat_l + mat_e) / 2.0

            edge_index, edge_attr = self.mat2edge(mat)
            x = x_l - x_e
            x = torch.cat((x, torch.eye(148)), dim=1)
            # print(x.size())
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            datalist.append(data)
        return datalist

    def prepare_data(self):
        file_dir = self.hparams['s_data']
        file_loc = os.path.join(file_dir, 'data_%d'%(self.hparams['num_classes']))
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        if not os.path.exists(file_loc):
            datalist = self.generate_data()
            torch.save(datalist,file_loc)
        else:
            datalist = torch.load(os.path.join(file_loc))
        N = len(datalist)
        torch.manual_seed(12345)


        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(datalist,
                                                                                [int(N*0.8), int(N*0.1), N-int(N*0.8)-int(N*0.1)])
        self.dataset = {}
        self.dataset['train'],  self.dataset['val'],  self.dataset['test'] = train_dataset, val_dataset, test_dataset
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of validation graphs: {len(val_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.hparams["batch_size"], shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        return optim

    def getTestAcc(self, loader=None):
        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            batch = batch.to(self.device)
            y = batch.y
            #y = generate_y(y)
            y_pred = self.forward(x=batch.x, edge_attr=batch.edge_attr,
                                    edge_index=batch.edge_index, batch=batch.batch)

            score = y_pred
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

    

class SAGPool(torch.nn.Module):
    def __init__(self, num_layers, hidden, num_node_features, num_classes, ratio=0.8):
        super(SAGPool, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden, aggr='mean')
        #self.conv1 = GCNConv(num_node_features, hidden, add_self_loops=False)
        
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            #GCNConv(hidden, hidden, add_self_loops=False)
            
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [SAGPooling(hidden, ratio) for i in range((num_layers-1) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x=x, edge_index = edge_index))
        #x = F.relu(self.conv1(x=x, edge_index = edge_index, edge_weight = edge_attr))
        
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x=x, edge_index = edge_index))
            #x = F.relu(conv(x=x, edge_index = edge_index, edge_weight = edge_attr))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,batch=batch)
                #x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index,edge_attr = edge_attr,batch=batch)
                
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    def get_att(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x=x, edge_index = edge_index))
        #x = F.relu(self.conv1(x=x, edge_index = edge_index, edge_weight = edge_attr))
        
        xs = [global_mean_pool(x, batch)]
        ps = []
        ss = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x=x, edge_index = edge_index))
            #x = F.relu(conv(x=x, edge_index = edge_index, edge_weight = edge_attr))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, perm, score = pool(x, edge_index,batch=batch)
                #x, edge_index, edge_attr, batch, perm, score =  pool(x, edge_index,edge_attr = edge_attr, batch=batch)
                
                ps.append(perm)
                ss.append(score)
                
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        #return F.log_softmax(x, dim=-1)
        return ps, ss

class SAGPool_g(torch.nn.Module):
    def __init__(self, num_layers, hidden, num_node_features, num_classes, ratio=0.8):
        super(SAGPool, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden, aggr='mean')
        #self.conv1 = GCNConv(num_node_features, hidden, add_self_loops=False)
        
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            #GCNConv(hidden, hidden, add_self_loops=False)
            
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [SAGPooling(hidden, ratio) for i in range((num_layers-1) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x=x, edge_index = edge_index))
        #x = F.relu(self.conv1(x=x, edge_index = edge_index, edge_weight = edge_attr))
        
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x=x, edge_index = edge_index))
            #x = F.relu(conv(x=x, edge_index = edge_index, edge_weight = edge_attr))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,batch=batch)
                #x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index,edge_attr = edge_attr,batch=batch)
                
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    
class TimeDiffClassifier_sagpooling(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = SAGPool(num_layers = self.hparams['num_layers'], 
                             hidden = self.hparams['hidden_channels'], 
                             num_classes=self.hparams['num_classes'],
                            num_node_features = self.hparams['num_node_features'],
                            ratio = self.hparams['ratio'])

    def edge2mat(self, edge_index, edge_attr, n=148):
        efn = edge_attr.shape[-1]
        mat = torch.zeros([n, n, efn])
        for eidx, ef in zip(edge_index.T, edge_attr):
            mat[eidx[0], eidx[1]] = ef
        return mat

    def mat2edge(self, mat):
        n = mat.shape[0]
        efn = mat.shape[-1]
        edge_index = []
        edge_attr = []
        for i in range(n):
            for j in range(n):
                if mat[i, j] != torch.zeros(efn):
                    edge_index.append([i, j])
                    edge_attr.append(mat[i, j])
        edge_index = torch.tensor(edge_index).T
        edge_attr = torch.tensor(edge_attr).unsqueeze(-1)
        #edge_attr.shape
        return (edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr, batch):
        y_pred = self.model(x, edge_index, edge_attr, batch)
        return y_pred

    def general_step(self, batch,batch_idx, mode):
        batch = batch.to(self.device)
        y = batch.y
        y_pred = self.model(batch)
        loss = F.cross_entropy(y_pred, y)

        preds = y_pred.argmax(axis = 1)
        n_correct = (y == preds).sum()
        return loss, n_correct
    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss, acc = self.general_end(outputs, "test")
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    def generate_data(self):
        s = '../data/DT_File_MAP.xlsx'
        # df = pd.read_excel(osp.join(*(s.split('/'))))
        df = pd.read_excel(self.hparams['s_node'])
        s = '../data/[UNC]ADNI-network/dataTS.csv'
        # df_ts = pd.read_csv(osp.join(*(s.split('/'))))
        df_ts = pd.read_csv(self.hparams['s_date'])

        df_m = df[["Subject", "VISCODE"]].merge(df_ts, how='inner', left_on="Subject", right_on="subject")
        df_m['date'] = pd.to_datetime(df_m['EXAMDATE.x'], format='%m/%d/%Y')
        df_ms = df_m.sort_values(["PTID", "date"])[["Subject", "PTID", "date", "VISCODE_x", "VISCODE_y", "DX_bl"]]
        df_msl = df_ms[1:].reset_index(drop=True)
        df_diff = df_msl.merge(df_ms, on=None, left_index=True, right_index=True, how="right", suffixes=('_l', '_e'))
        df_diff = df_diff[df_diff['PTID_l'] == df_diff['PTID_e']]
        df_diff = df_diff.reset_index(drop=True)
        df_diff['interval_m'] = df_diff['VISCODE_x_l'] - df_diff['VISCODE_x_e']
        df_diff['interval_d'] = (df_diff['date_l'] - df_diff['date_e']).dt.days
        df_ms_base = df_ms.groupby("PTID").first()
        df_ms_base = df_ms_base[['Subject']]
        df_ms_base.columns = ['Subject_b']
        df_diff = df_diff.merge(df_ms_base, left_on = 'PTID_l', right_on = 'PTID', how = 'inner')
        
        labels = {}
        if (self.hparams['num_classes'] == 4):

            labels['CN'] = 0
            labels['EMCI'] = 1
            labels['LMCI'] = 2
            labels['AD'] = 3
        elif self.hparams['num_classes'] == 3:
            labels['CN'] = 0
            labels['EMCI'] = 1
            labels['LMCI'] = 2
            labels['AD'] = 2
        elif self.hparams['num_classes'] == 2:
            labels['CN'] = 0
            labels['AD'] = 1
        s = '../sparsification/data'
        # filepath_r = osp.join(*(s.split('/')))
        filepath_r = self.hparams['s_sparsified']
        data_dict = {}
        for i in range(df.shape[0]):
            label = df.loc[i]["LABEL"]
            if (label not in labels.keys()):
                continue
            y = labels[label]
            # filename = osp.join(filepath_r, "%s_fdt_network_matrix_df.csv" % df.loc[i]["Subject"])
            filename = filepath_r + "%s_fdt_network_matrix_df.csv" % df.loc[i]["Subject"]
            sbj = df.loc[i]['Subject']
            edge_table = pd.read_csv(filename, sep="\t")
            edge_table = edge_table[["src", "trg", "nij"]]
            edge_index = torch.tensor(edge_table[["src", "trg"]].values).transpose(0, 1)
            edge_attr = torch.tensor(edge_table[["nij"]].values, dtype=torch.float)
            x = torch.tensor(df[df.index == i][["Node %d" % (i) for i in range(1, 149)]].values,
                             dtype=torch.float).transpose(0, 1)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            data_dict[sbj] = data

        datalist = []
        for i in range(df_diff.shape[0]):
            label = df_diff.loc[i]['DX_bl_e']
            if (label not in labels.keys()):
                continue
            y = labels[label]
            print('data processing %d for label %d' % (i, y))
            sbj_l = df_diff.loc[i]['Subject_l']
            sbj_e = df_diff.loc[i]['Subject_e']
            sbj_b = df_diff.loc[i]['Subject_b']
            d_l = data_dict[sbj_l]
            d_e = data_dict[sbj_e]
            d_b = data_dict[sbj_b]

            x_l = d_l.x
            edge_index_l = d_l.edge_index
            edge_attr_l = d_l.edge_attr

            x_e = d_e.x
            edge_index_e = d_e.edge_index
            edge_attr_e = d_e.edge_attr

            edge_index_b = d_b.edge_index
            edge_attr_b = d_b.edge_attr
            
            if(self.hparams['use_base_edge']):
                edge_index = edge_index_b
                edge_attr = edge_attr_b
            else:
                mat_l = self.edge2mat(edge_index_l, edge_attr_l)
                mat_e = self.edge2mat(edge_index_e, edge_attr_e)
                # print(mat_l.shape, mat_e.shape)
                # print(edge_attr_e.shape)
                mat = (mat_l + mat_e) / 2.0
                edge_index, edge_attr = self.mat2edge(mat)
                
            x = x_l - x_e
            x = torch.cat((x, torch.eye(148)), dim=1)
            # print(x.size())
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            datalist.append(data)
        return datalist

    def prepare_data(self):
        file_dir = self.hparams['s_data']
        file_loc = os.path.join(file_dir, 'data_%d'%(self.hparams['num_classes']))
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        if not os.path.exists(file_loc):
            datalist = self.generate_data()
            torch.save(datalist,file_loc)
        else:
            datalist = torch.load(os.path.join(file_loc))
        N = len(datalist)
        torch.manual_seed(12345)


        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(datalist,
                                                                                [int(N*0.6), int(N*0.2), N-int(N*0.6)-int(N*0.2)])
        self.dataset = {}
        self.dataset['train'],  self.dataset['val'],  self.dataset['test'] = train_dataset, val_dataset, test_dataset
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of validation graphs: {len(val_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.hparams["batch_size"], shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        return optim

    def getTestAcc(self, loader=None):
        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            batch = batch.to(self.device)
            y = batch.y
            #y = generate_y(y)
            y_pred = self.forward(x=batch.x, edge_attr=batch.edge_attr,
                                    edge_index=batch.edge_index, batch=batch.batch)

            score = y_pred
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

    
'''
if __name__ == '__main__' :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    hparams = {}
    hparams['batch_size'] = 32
    hparams['num_classes'] = 2
    hparams['hidden_channels'] = 64
    #hparams['num_node_features'] = 149
    hparams['batch_size'] = 32
    hparams['lr'] = 1.0e-03
    hparams['weight_decay'] = 5.0e-06


    s='./data/DT_File_MAP.xlsx'
    hparams['s_node'] = osp.join(*(s.split('/')))
    s='./data/[UNC]ADNI-network/dataTS.csv'
    hparams['s_date'] = osp.join(*(s.split('/')))
    s = './sparsification/data'
    hparams['s_sparsified'] = osp.join(*(s.split('/'))) + osp.sep
    
    model = TimeDiffClassify(hparams,)
    model = model.to(device)
    #model.prepare_data()

    trainer = None

    epochs = 10

    logdir = './lightning_logs'
    network_logger = pl.loggers.TensorBoardLogger(
        save_dir=logdir,
        name='TimeDiffClassify_logs'
    )
    logger = network_logger

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
    )

    trainer = pl.Trainer(max_epochs=epochs, early_stop_callback=early_stopping)

    trainer.fit(model)

'''