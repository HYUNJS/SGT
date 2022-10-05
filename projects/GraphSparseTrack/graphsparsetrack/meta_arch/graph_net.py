import copy

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_sum
from torch_geometric.nn import MetaLayer
from torch_scatter.utils import broadcast
from typing import Optional, Tuple
import torch.nn.functional as F
from projects.GraphSparseTrack.graphsparsetrack.utils import flatten_boxes_dict, mask_boxes_dict
from projects.GraphSparseTrack.graphsparsetrack.meta_arch.graph_build import build_sparse_graph


def build_layers(input_dim, fc_dims, dropout_p, norm_layer, act_func, bias=True):
    layers = []
    for fc_dim in fc_dims:
        layers.append(nn.Linear(input_dim, fc_dim, bias=bias))
        layers.append(norm_layer(fc_dim))
        layers.append(act_func)
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        input_dim = fc_dim
    return layers

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm1d
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm
    elif norm_type == 'no':
        norm_layer = nn.Identity
    else:
        raise Exception("Unsupported Normalization Layer - {}".format(norm_type))
    return norm_layer

def get_act_func(act_type):
    if act_type == 'relu':
        act_func = nn.ReLU(inplace=True)
    elif act_type == 'elu':
        act_func = nn.ELU(inplace=True)
    elif act_type == 'leaky_relu':
        act_func = nn.LeakyReLU(inplace=True)
    elif act_type == 'sigmoid':
        act_func = nn.Sigmoid()
    elif act_type == 'tanh':
        act_func = nn.Tanh()
    elif act_type == 'no':
        act_func = nn.Identity()
    else:
        raise Exception("Unsupported Activation Function - {}".format(act_type))
    return act_func

@torch.jit.script
def scatter_weighted_sum(src: torch.Tensor, index: torch.Tensor, weight: torch.Tensor,
         dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    src = src * weight
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

@torch.jit.script
def scatter_weighted_mean(src: torch.Tensor, index: torch.Tensor, weight: torch.Tensor,
      dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None) -> torch.Tensor:

    out = scatter_weighted_sum(src, index, weight, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)
    count = broadcast(count, out, dim)
    if torch.is_floating_point(out):
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out

class EdgeGNN(nn.Module):
    ## reference mot_neural_solver project - https://github.com/dvl-tum/mot_neural_solver
    def __init__(self, cfg):
        super(EdgeGNN, self).__init__()

        cfg_node = cfg.NODE_MODEL
        cfg_edge = cfg.EDGE_MODEL
        self.n_iter = cfg.N_ITER  # number of message passing iteration
        self.deep_loss = cfg_edge.CLASSIFY.DEEP_LOSS
        self.exclude_init_loss = cfg_edge.CLASSIFY.EXCLUDE_INIT_LOSS
        self.node_encode_flag = cfg_node.ENCODE.FLAG
        self.edge_encoder = EdgeEncodeNet(cfg_edge.ENCODE) # edge feature extracter (encoding edge input data into high dim feature)
        if self.node_encode_flag:
            self.node_encoder = NodeEncodeNet(cfg_node.ENCODE) # node feature encoder (remap node feature)
        self.MPNet = self.build_MPNet(cfg_node.UPDATE, cfg_edge.UPDATE) # message passing network

    def build_MPNet(self, node_model_param, edge_model_param):
        ## define node aggregation function
        node_agg_fn = node_model_param.AGG_FUNC.lower()
        assert node_agg_fn in ('mean', 'max', 'sum', 'w_sum', 'w_mean'), "node_agg_fn can only be 'max', 'mean' or 'sum'."
        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size, _: scatter_mean(out, row, dim=0, dim_size=x_size)
        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size, _: scatter_max(out, row, dim=0, dim_size=x_size)[0]
        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size, _: scatter_add(out, row, dim=0, dim_size=x_size)
        elif node_agg_fn == 'w_sum':
            node_agg_fn = lambda out, row, x_size, weight: scatter_weighted_sum(out, row, weight, dim=0, dim_size=x_size)
        elif node_agg_fn == 'w_mean':
            node_agg_fn = lambda out, row, x_size, weight: scatter_weighted_mean(out, row, weight, dim=0, dim_size=x_size)

        ## define node & edge feature dims
        self.reattach_initial_edge = edge_model_param.REATTACH
        self.reattach_initial_node = node_model_param.REATTACH
        self.skip_initial_edge = edge_model_param.SKIP_CONN
        self.skip_initial_node = node_model_param.SKIP_CONN
        # assert not (self.reattach_initial_edge and self.skip_initial_edge), "Choose only one option for edge feature - Reattach vs Skip"
        # assert not (self.reattach_initial_node and self.skip_initial_node), "Choose only one option for node feature - Reattach vs Skip"
        edge_factor = 2 if self.reattach_initial_edge else 1
        node_factor = 2 if self.reattach_initial_node else 1
        node_feat_dim, edge_feat_dim = node_model_param.IN_DIM, edge_model_param.IN_DIM
        edge_model_in_dim = node_factor * 2 * node_feat_dim + edge_factor * edge_feat_dim
        node_model_in_dim = node_factor * node_feat_dim + edge_feat_dim
        if self.skip_initial_edge:
            edge_norm_layer = get_norm_layer(edge_model_param.NORM_LAYER)
            self.edge_norm_layers = nn.ModuleList([edge_norm_layer(edge_feat_dim)]*self.n_iter)
        if self.skip_initial_node:
            node_norm_layer = get_norm_layer(node_model_param.NORM_LAYER)
            self.node_norm_layers = nn.ModuleList([node_norm_layer(node_feat_dim)]*self.n_iter)

        ## define edge update network
        if edge_model_param.ADAPTIVE_FLAG:
            edge_update_model = EdgeAdaptiveUpdateNet(edge_model_param, node_factor*node_feat_dim, edge_factor*edge_feat_dim)
        else:
            edge_update_model = EdgeUpdateNet(edge_model_param, edge_model_in_dim)

        ## define node update network
        if node_model_param.ADAPTIVE_FLAG:
            node_update_model = NodeAdaptiveUpdateNet(node_model_param, node_model_in_dim, edge_feat_dim, node_agg_fn)
        else:
            node_update_model = NodeUpdateNet(node_model_param, node_model_in_dim, node_agg_fn)

        ## package edge and node update networks into Message Passing Network
        return MetaLayer(edge_model=edge_update_model, node_model=node_update_model)

    def forward(self, graph_batch, **kwargs):
        initial_node_feats = graph_batch.x
        edge_index = graph_batch.edge_index
        initial_edge_feats = graph_batch.edge_attr
        latent_node_feats_list, latent_edge_feats_list = [], []
        if self.node_encode_flag:
            initial_node_feats = self.node_encoder(initial_node_feats)
        initial_edge_feats = self.edge_encoder(initial_edge_feats)
        latent_node_feats, latent_edge_feats = initial_node_feats, initial_edge_feats

        ## iterate Message Passing in GNN
        for step in range(self.n_iter):
            ## stack intermediate node and edge features
            if self.training and self.deep_loss and not (self.exclude_init_loss and step==0):
                latent_node_feats_list.append(latent_node_feats)
                latent_edge_feats_list.append(latent_edge_feats)

            ## attach initial feature before update
            if self.reattach_initial_node:
                latent_node_feats = torch.cat([initial_node_feats, latent_node_feats], dim=1)
            if self.reattach_initial_edge:
                latent_edge_feats = torch.cat([initial_edge_feats, latent_edge_feats], dim=1)

            ## message passing network
            latent_node_feats, latent_edge_feats, _ = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            ## skip connection
            if self.skip_initial_node:
                latent_node_feats = initial_node_feats + latent_node_feats
                latent_node_feats = self.node_norm_layers[step](latent_node_feats)
            if self.skip_initial_edge:
                latent_edge_feats = initial_edge_feats + latent_edge_feats
                latent_edge_feats = self.edge_norm_layers[step](latent_edge_feats)

        ## stack last updated node and edge features
        latent_node_feats_list.append(latent_node_feats)
        latent_edge_feats_list.append(latent_edge_feats)

        return latent_node_feats_list, latent_edge_feats_list

class EdgeEncodeNet(nn.Module):
    def __init__(self, cfg):
        super(EdgeEncodeNet, self).__init__()
        input_dim = cfg.IN_DIM
        out_dim = cfg.OUT_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = get_norm_layer(cfg.NORM_LAYER)
        act_func = get_act_func(cfg.ACT_TYPE)
        dropout_p = cfg.DROPOUT_P
        layers = build_layers(input_dim, fc_dims+[out_dim], dropout_p, norm_layer, act_func)
        layers += [nn.Linear(out_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_attr):
        return self.mlp(edge_attr)

class NodeEncodeNet(nn.Module):
    def __init__(self, cfg):
        super(NodeEncodeNet, self).__init__()
        input_dim = cfg.IN_DIM
        out_dim = cfg.OUT_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = get_norm_layer(cfg.NORM_LAYER)
        act_func = get_act_func(cfg.ACT_TYPE)
        dropout_p = cfg.DROPOUT_P
        layers = build_layers(input_dim, fc_dims+[out_dim], dropout_p, norm_layer, act_func)
        layers += [nn.Linear(out_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_attr):
        return self.mlp(edge_attr)

class EdgeUpdateNet(nn.Module):
    def __init__(self, cfg, edge_model_in_dim):
        super(EdgeUpdateNet, self).__init__()
        out_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = get_norm_layer(cfg.NORM_LAYER)
        act_func = get_act_func(cfg.ACT_TYPE)
        dropout_p = cfg.DROPOUT_P
        layers = build_layers(edge_model_in_dim, fc_dims+[out_dim], dropout_p, norm_layer, act_func)
        self.edge_mlp = nn.Sequential(*layers)
        self.self_loop = cfg.SELF_LOOP

    def forward(self, src, tgt, edge_attr, u=None, batch=None):
        '''
        :param src: [E, F_x], where E is the number of edges.
        :param tgt: [E, F_x], where E is the number of edges.
        :param edge_attr: [E, F_e * edge_factor], where edge_factor = 2 if reattach else 1
        :param u: [B, F_u], where B is the number of graphs.
        :param batch: [E] with max entry B - 1.
        :return: updated edge feature [E, F_e]
        '''
        # out = torch.cat([src, tgt, edge_attr, u[batch]], dim=1)
        out = torch.cat([src, tgt, edge_attr], dim=1)
        return self.edge_mlp(out)

class EdgeAdaptiveUpdateNet(nn.Module):
    def __init__(self, cfg, node_in_dim, edge_in_dim):
        super(EdgeAdaptiveUpdateNet, self).__init__()
        out_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = get_norm_layer(cfg.NORM_LAYER)
        dropout_p = cfg.DROPOUT_P
        act_func = get_act_func(cfg.ACT_TYPE)
        self.self_loop = cfg.SELF_LOOP
        self.edge_in_dim = cfg.IN_DIM
        self.node2edge = nn.Sequential(*build_layers(node_in_dim*2, fc_dims+[out_dim], dropout_p, norm_layer, act_func))
        self.edge_mlp1 = nn.Sequential(*build_layers(edge_in_dim, fc_dims+[out_dim], dropout_p, norm_layer, act_func))
        # self.weight_mlp = nn.Sequential(*build_layers(edge_in_dim, [out_dim], 0, get_norm_layer('no'), nn.Sigmoid()))
        self.weight_mlp = nn.Sequential(*build_layers(edge_in_dim, [1], 0, get_norm_layer('no'), nn.Sigmoid()))
        self.final_mlp = nn.Sequential(*build_layers(out_dim, [out_dim], dropout_p, norm_layer, act_func))

    def forward(self, src, tgt, edge_attr, u=None, batch=None):
        '''
        :param src: [E, F_x], where E is the number of edges.
        :param tgt: [E, F_x], where E is the number of edges.
        :param edge_attr: [E, F_e * edge_factor], where edge_factor = 2 if reattach else 1
        :param u: [B, F_u], where B is the number of graphs.
        :param batch: [E] with max entry B - 1.
        :return: updated edge feature [E, F_e]
        '''
        node2edge_feat = self.node2edge(torch.cat([src, tgt], dim=1))
        agg_edge_feat = self.edge_mlp1(edge_attr)
        fuse_weight = self.weight_mlp(edge_attr)
        fused_feat = node2edge_feat * fuse_weight + agg_edge_feat * (1 - fuse_weight)
        return self.final_mlp(fused_feat)

class NodeUpdateNet(nn.Module):
    def __init__(self, cfg, node_model_in_dim, node_agg_fn):
        super(NodeUpdateNet, self).__init__()
        out_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = get_norm_layer(cfg.NORM_LAYER)
        act_func = get_act_func(cfg.ACT_TYPE)
        dropout_p = cfg.DROPOUT_P
        self.shared_weight_flag = cfg.SHARED_FLAG
        self.self_loop = cfg.SELF_LOOP
        if self.shared_weight_flag:
            flow_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
        else:
            flow_in_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
            flow_out_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
        if cfg.TMPNN_FLAG:
            node_layers = build_layers(out_dim*2, [out_dim], dropout_p, norm_layer, act_func)
        else:
            node_layers = build_layers(out_dim, [out_dim], dropout_p, norm_layer, act_func)
        if self.shared_weight_flag:
            self.flow_shared_mlp = nn.Sequential(*flow_layers)
        else:
            self.flow_trk2det_mlp = nn.Sequential(*flow_in_layers)
            self.flow_det2trk_mlp = nn.Sequential(*flow_out_layers)
        if self.self_loop:
            self.self_mlp = nn.Sequential(*build_layers(out_dim, [out_dim], dropout_p, norm_layer, act_func))
            # self.w_self = nn.Parameter(torch.rand(out_dim, out_dim))
            # self.b_self = nn.Parameter(torch.rand(out_dim))
            # self.norm_self = nn.LayerNorm(out_dim)
        self.node_mlp = nn.Sequential(*node_layers)
        self.node_agg_fn = node_agg_fn
        self.tmpnn_flag = cfg.TMPNN_FLAG

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes. (all batch), e.g. if batch_size=1 and #tracklet=150 and #detection=100, then N=250
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e] E is number of all edges (all batch + both direections) e.g. if batch_size=1 and #edge in the graph (one direction)=1472, then E=1472*2=2944
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        flow_det2trk_mask = row < col
        flow_det2trk_row, flow_det2trk_col = row[flow_det2trk_mask], col[flow_det2trk_mask] # row: det | col: trk
        flow_det2trk_input = torch.cat([x[flow_det2trk_col], edge_attr[flow_det2trk_mask]], dim=1)
        if self.shared_weight_flag:
            flow_det2trk = self.flow_shared_mlp(flow_det2trk_input)  # [# edges of det2trk direction, #hidden dim]
        else:
            flow_det2trk = self.flow_det2trk_mlp(flow_det2trk_input) # [# edges of det2trk direction, #hidden dim]

        flow_trk2det_mask = row > col
        flow_trk2det_row, flow_trk2det_col = row[flow_trk2det_mask], col[flow_trk2det_mask]
        flow_trk2det_input = torch.cat([x[flow_trk2det_col], edge_attr[flow_trk2det_mask]], dim=1)
        if self.shared_weight_flag:
            flow_trk2det = self.flow_shared_mlp(flow_trk2det_input)
        else:
            flow_trk2det = self.flow_trk2det_mlp(flow_trk2det_input)

        flow_det2trk = self.node_agg_fn(flow_det2trk, flow_det2trk_row, x.size(0), None) # [# nodes, # hiddem dim]
        flow_trk2det = self.node_agg_fn(flow_trk2det, flow_trk2det_row, x.size(0), None)
        if self.tmpnn_flag:
            flow_total = torch.cat([flow_trk2det, flow_det2trk], dim=1) # [# nodes, # hidden dim * 2]
        else:
            flow_total = flow_trk2det + flow_det2trk
        updated_node_feats = self.node_mlp(flow_total)
        if self.self_loop:
            self_node_feats = self.self_mlp(x)
            updated_node_feats = updated_node_feats + self_node_feats

        return updated_node_feats

class NodeAdaptiveUpdateNet(nn.Module):
    def __init__(self, cfg, node_model_in_dim, edge_feat_dim, node_agg_fn):
        super(NodeAdaptiveUpdateNet, self).__init__()
        out_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = get_norm_layer(cfg.NORM_LAYER)
        act_func = get_act_func(cfg.ACT_TYPE)
        dropout_p = cfg.DROPOUT_P

        flow_in_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
        flow_out_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
        node_layers = build_layers(out_dim, [out_dim], dropout_p, norm_layer, act_func)
        self.flow_trk2det_mlp = nn.Sequential(*flow_in_layers)
        self.flow_det2trk_mlp = nn.Sequential(*flow_out_layers)
        self.edge2weight = nn.Sequential(*build_layers(edge_feat_dim, [1], 0, get_norm_layer('no'), get_act_func('sigmoid')))
        self.node_agg_weight = nn.Sequential(*build_layers(out_dim, [1], 0, get_norm_layer('no'), get_act_func('sigmoid')))
        self.node_mlp = nn.Sequential(*node_layers)
        self.node_agg_fn = node_agg_fn
        self.tmpnn_flag = cfg.TMPNN_FLAG
        self.self_loop = cfg.SELF_LOOP
        assert not self.tmpnn_flag, "Adaptive node update is not yet supporting TMPNN version"

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes. (all batch), e.g. if batch_size=1 and #tracklet=150 and #detection=100, then N=250
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e] E is number of all edges (all batch + both direections) e.g. if batch_size=1 and #edge in the graph (one direction)=1472, then E=1472*2=2944
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        flow_det2trk_mask = row < col
        flow_det2trk_row, flow_det2trk_col = row[flow_det2trk_mask], col[flow_det2trk_mask] # row: det | col: trk
        flow_det2trk_input = torch.cat([x[flow_det2trk_col], edge_attr[flow_det2trk_mask]], dim=1)
        flow_det2trk = self.flow_det2trk_mlp(flow_det2trk_input) # [# edges of det2trk direction, #hidden dim]

        flow_trk2det_mask = row > col
        flow_trk2det_row, flow_trk2det_col = row[flow_trk2det_mask], col[flow_trk2det_mask]
        flow_trk2det_input = torch.cat([x[flow_trk2det_col], edge_attr[flow_trk2det_mask]], dim=1)
        flow_trk2det = self.flow_trk2det_mlp(flow_trk2det_input)

        edge_weight = self.edge2weight(edge_attr)
        W_det2trk, W_trk2det = edge_weight[flow_det2trk_mask], edge_weight[flow_trk2det_mask]
        norm_x = F.normalize(x, dim=1)
        node_pair_sim = (norm_x[row[flow_det2trk_mask]] * norm_x[col[flow_det2trk_mask]]).sum(dim=1).unsqueeze(1)
        flow_det2trk = self.node_agg_fn(flow_det2trk, flow_det2trk_row, x.size(0), W_det2trk) # [# nodes, # hiddem dim]
        flow_trk2det = self.node_agg_fn(flow_trk2det, flow_trk2det_row, x.size(0), W_trk2det)
        node_det2trk = self.node_agg_fn(x[flow_det2trk_col], flow_det2trk_row, x.size(0), node_pair_sim)
        node_trk2det = self.node_agg_fn(x[flow_trk2det_col], flow_trk2det_row, x.size(0), node_pair_sim)

        flow_total = flow_det2trk + flow_trk2det
        node_total = node_det2trk + node_trk2det
        node_weight = self.node_agg_weight(flow_total)
        fused_feat = flow_total + node_weight * node_total
        # return self.node_mlp(torch.cat([fused_feat, x], dim=1))
        return self.node_mlp(fused_feat)

class CrossFrameInteractionGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg_graph = cfg.MODEL.TRACKER.GNN.GRAPH
        self.gnn = EdgeGNN(cfg.MODEL.TRACKER.GNN)
        self.num_proposals = cfg.MODEL.TRACKER.GNN.TOPK_DET
        self.topk = cfg_graph.TOPK
        self.graph_attr = cfg_graph.ATTR
        self.mask_out_zero_iou_flag = cfg_graph.MASK_OUT_ZERO_IOU_FLAG
        self.edge_attr = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.ENCODE.EDGE_ATTR
        self.edge_attr_legacy = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.ENCODE.EDGE_ATTR_LEGACY
        self.directional_edge_attr = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.ENCODE.DIRECTIONAL_EDGE_ATTR
        self.cut_edge_grad = cfg.MODEL.TRACKER.CUT_EDGE_GRAD
        self.cos_sim_by_node_feat = cfg.MODEL.TRACKER.NODE_PRE_ENCODE.FLAG and cfg.MODEL.TRACKER.NODE_PRE_ENCODE.COS_SIM
        self.output_size = cfg.MODEL.CENTERNET.OUTPUT_SIZE
        self.prev_det_proposal_flag = cfg.MODEL.TRACKER.GNN.GRAPH.PREV_DET_PROPOSAL_FLAG

    def separate_t1_t2_node_feats(self, t1_box_nums, t2_box_nums, updated_features):
        '''
        :param t1_box_nums: list of number of t1 boxes - length is batch size
        :param t2_box_nums: list of number of t2 boxes - length is batch size
        :param updated_features: node features of t1 and t2 nodes - concat([t2, t1])
        :return: t1 node features & t2 node features
        '''
        new_t1_feats, new_t2_feats = [], []
        curr_idx = 0
        for b_idx in range(len(t1_box_nums)):
            t1_box_num, t2_box_num = t1_box_nums[b_idx], t2_box_nums[b_idx]
            new_t2_feats.append(updated_features.narrow(0, curr_idx, t2_box_num))
            new_t1_feats.append(updated_features.narrow(0, curr_idx + t2_box_num, t1_box_num))
            curr_idx = curr_idx + t1_box_num + t2_box_num

        new_t1_feats = torch.cat(new_t1_feats)
        new_t2_feats = torch.cat(new_t2_feats)
        return new_t1_feats, new_t2_feats

    def build_graph(self, data_dict, history=None):
        '''
        concat missed tracklets (history) if they are given
        :param data_dict: Dict
        :param history: Dict
        :return: data in dict format to build a graph
        '''
        scores = data_dict['tensor']['scores']
        fmap_feats = data_dict['tensor']['fmap_feats']
        node_feats = data_dict['tensor']['node_feats']
        boxes_dict = copy.deepcopy(data_dict['tensor']['boxes_dict'])
        box_nums = copy.deepcopy(data_dict['box_nums'])

        if history is not None:  ## inference time, add history tracklets
            if self.prev_det_proposal_flag and history['boxes_dict']['xs'].dim() == 3:
                flatten_history_boxes = flatten_boxes_dict(history['boxes_dict'])
            else:
                flatten_history_boxes = history['boxes_dict']
            for k in boxes_dict.keys():
                boxes_dict[k] = torch.cat([boxes_dict[k], flatten_history_boxes[k]], dim=0)

            if self.prev_det_proposal_flag and history['fmap_feats'].dim() == 3:
                fmap_feats = torch.cat([fmap_feats, history['fmap_feats'].flatten(0, 1)], dim=0)
                node_feats = torch.cat([node_feats, history['node_feats'].flatten(0, 1)], dim=0)
            else:
                fmap_feats = torch.cat([fmap_feats, history['fmap_feats']], dim=0)
                node_feats = torch.cat([node_feats, history['node_feats']], dim=0)
            scores = torch.cat([scores, history['scores']], dim=1)
            box_nums[0] += history['tids'].size(1)

        data_for_graph = {'fmap_feats': fmap_feats, 'node_feats': node_feats, 'boxes_dict': boxes_dict,
                          'box_nums': box_nums, 'scores': scores}
        return data_for_graph

    def forward(self, t1_info, t2_info, history=None, extrp_result=None):
        assert t1_info is not None and t2_info is not None

        N = len(t1_info['box_nums'])
        images_whwh = self.output_size[::-1] + self.output_size[::-1]
        images_whwh = [torch.tensor(images_whwh, dtype=torch.float32, device=t1_info['tensor']['fmap_feats'].device)] * N

        ## build graph
        t1_info_for_graph = self.build_graph(t1_info, history)
        # t2_info_for_graph = self.build_graph(t2_info)
        t2_info_for_graph = self.build_graph(t2_info, extrp_result)
        t1_box_nums, t2_box_nums = t1_info_for_graph['box_nums'], t2_info_for_graph['box_nums']
        if t2_box_nums[0] > 100:
            t2_box_nums

        graph_data, edge_batch_masks, edge_batch_idx_offsets = build_sparse_graph(t1_info_for_graph, t2_info_for_graph,
                    images_whwh, k=self.topk, mask_out_zero_iou_flag=self.mask_out_zero_iou_flag, edge_attr=self.edge_attr,
                    edge_attr_legacy=self.edge_attr_legacy, graph_attr=self.graph_attr,  cut_edge_grad=self.cut_edge_grad,
                    directional_edge_attr=self.directional_edge_attr, cos_sim_by_node_feat=self.cos_sim_by_node_feat)

        ## run GNN to update node and edge features
        updated_features = self.gnn(graph_data) # return: Tuple([node_feats] * num_deep_loss, [edge_feats] * num_deep_loss)
        node_features_list, edge_features_list = updated_features[0], updated_features[1]

        ## separate t1 and t2 nodes per each iteration
        new_t1_feats_list, new_t2_feats_list = [], []
        num_deep_loss = len(node_features_list)
        for i in range(num_deep_loss):
            new_t1_feats, new_t2_feats = self.separate_t1_t2_node_feats(t1_box_nums, t2_box_nums, node_features_list[i])
            new_t1_feats_list.append(new_t1_feats)
            new_t2_feats_list.append(new_t2_feats)

        edge_dict = {'features': edge_features_list, 'index': graph_data.edge_index, 'batch_masks': edge_batch_masks,
                     'edge_batch_idx_offsets': edge_batch_idx_offsets}
        return new_t1_feats_list, new_t2_feats_list, edge_dict