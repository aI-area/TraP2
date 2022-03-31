""" explain.py

    Implementation of the explainer.
"""

import math
import time
import os
import copy
import scipy as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
import numpy as np
import tensorboardX.utils
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from functools import partial

from utils.pytorchtools import EarlyStopping

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils
import utils.math_utils as math_utils

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class Explainer:
    def __init__(
            self,
            model,
            adj,
            feat,
            label,
            pred,
            train_idx,
            args,
            writer=None,
            print_training=True,
            graph_mode=False,
            graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat

        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods \
            = graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops,
                                        use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training

        # difference between actual prediction and limited prediction
        self.fidelity_val = 0
        self.contrastivity = 0

    def generate_masked_adj(self, explainer, strategy, node_idx):
        """
        archieve vector combining adjacency and feature

        :param explainer: specified explainer module
        :param strategy: strategy to pick weight inside g function across all
        label class
        :param node_idx: node index to explain
        :return: a matrix that cofficients of each node pair are represented
        """

        if strategy == "max_prob_class":
            if len(self.label.shape) > 1:
                idx = self.label[0][
                    node_idx]
            else:
                idx = self.label[node_idx]

            masked_adj_feat = explainer.g_model.lr.weight[idx]
        elif strategy == "avg_prob_class":
            masked_adj_feat \
                = np.sum(np.fabs(np.array(explainer.g_model.lr.weight.data)),
                                          axis=0)
        else:
            print("Undifined Strategy !")
            exit()

        # assemble 1 dimension vector into 2 dimension

        if self.args.gpu:
            masked_adj_feat = masked_adj_feat.cpu()

        reshaped_masked_adj_feat = np.array(masked_adj_feat.data).reshape(
            explainer.x.shape[1],
            explainer.x.shape[2])
        coef_adj = np.sum(np.fabs(reshaped_masked_adj_feat), axis=1)

        # possible updated solution
        # =====
        masked_adj = self.construct_average_matrix(coef_adj)
        return (masked_adj * np.squeeze(np.array(explainer.adj.data)))\
            .cpu().detach().numpy()

    def obtain_fidelity(self, adj, masked_adj, x, node_idx, node_idx_new, \
                                dataset):

        masked_adj=1/(1+np.exp(-masked_adj))

        if dataset in ["syn1", "syn2", "syn4"]:
            topn = 6
        elif dataset == "syn5":
            topn = 12
        else:
            print("Undefined nb. of nodes inside ground structure dataset !")
            exit()

        if len(masked_adj.shape) == 2:
            topn_nodes = math_utils.largest_indices(np.triu(masked_adj), topn)
            below_topn_nodes = copy.deepcopy(np.triu(masked_adj))
        else:
            topn_nodes = math_utils.largest_indices(masked_adj, topn)
            below_topn_nodes = copy.deepcopy(masked_adj)

        below_topn_nodes[topn_nodes] = 0
        below_topn_values = below_topn_nodes[np.nonzero(below_topn_nodes)]
        avg_below_topn_values = np.average(below_topn_values)

        # calculation of contrastivity
        if len(masked_adj.shape) == 2:
            min_topn = masked_adj[topn_nodes[0][-1], topn_nodes[1][-1]]
        else:
            min_topn = masked_adj[topn_nodes[0][-1]]
        self.contrastivity += float(min_topn/avg_below_topn_values)

        topn_mat = np.zeros(masked_adj.shape)
        topn_mat[topn_nodes] = 1.0

        # transform upper triangular matrix into symmetric one
        topn_mat = topn_mat + topn_mat.T

        with torch.no_grad():
            topn_mat = topn_mat[np.newaxis,:, :]
            if self.args.gpu:
                topn_mat = torch.Tensor(topn_mat).cuda()
                x = x.cuda()
            else:
                topn_mat = torch.Tensor(topn_mat)

            ypred, _ = self.model(x, topn_mat)
            ypred = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx_new, :])

            gt_label_node = self.label if self.graph_mode \
                else self.label[0][node_idx]
            logit = ypred[gt_label_node]

            # ground truth label is produced from entire graph
            gt_prd_ent = self.pred[0][node_idx]
            if self.args.gpu:
                gt_prd_ent = torch.Tensor(gt_prd_ent).cuda()
            else:
                gt_prd_ent = torch.Tensor(gt_prd_ent)

            gt_prd = nn.Softmax(dim=0)(gt_prd_ent)[gt_label_node]
            res = math.fabs(float(gt_prd - logit))
            self.fidelity_val += res
        return res

    def saliency_map(self, input_grads):
        node_saliency_map = []
        for n in range(input_grads.shape[0]): # nth node
            node_grads = input_grads[n,:]
            node_saliency = torch.norm(F.relu(node_grads)).item()
            node_saliency_map.append(node_saliency)
        return node_saliency_map

    def grad_cam(self, final_conv_acts, final_conv_grads):
        node_heat_map = []
        # mean gradient for each feature (512x1)
        alphas = torch.mean(final_conv_grads[0], axis=0)
        for n in range(final_conv_acts.shape[1]): # nth node
            node_heat = F.relu(alphas @ final_conv_acts[0][
                n]).item()
            node_heat_map.append(node_heat)
        return node_heat_map

    def construct_average_matrix(self, matrix):
        new_one = np.zeros([np.array(matrix).shape[0],
                            np.array(matrix).shape[0]])
        for i in range(0, new_one.shape[0]):
            for j in range(0, new_one.shape[0]):
                new_one[i][j] = (matrix[i].sum() + matrix[j].sum())/2
        return torch.Tensor(new_one)

    # Main method
    def explain(
            self, node_idx, graph_idx=0, graph_mode=False,
            unconstrained=False, model="exp", explainer="LG-IME",
            dataset="unknown"
    ):
        """Explain a single node
        """
        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            print("node label: ", self.label[graph_idx][node_idx])
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors \
                = self.extract_neighborhood(
                node_idx, graph_idx
            )
            print("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted label: ", pred_label[node_idx_new])

        # LGLIMExplainer, ExplainModule
        if explainer == "LG-IME":
            explainer = LGLIMExplainer(
                adj=adj,
                x=x,
                model=self.model,
                label=label,
                args=self.args,
                writer=self.writer,
                graph_idx=self.graph_idx,
                node_idx=node_idx_new,
                num_classes=self.pred.shape[2],
                graph_mode=self.graph_mode,
                dataset=dataset
            )
        elif explainer == "GNNExplainer":
            explainer = ExplainModule(
                adj=adj,
                x=x,
                model=self.model,
                label=label,
                args=self.args,
                writer=self.writer,
                graph_idx=self.graph_idx,
                graph_mode=self.graph_mode,
            )
        print("Expanation model: " + str(explainer.__class__.__name__))

        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new,
                                        pred_label[node_idx_new])[0]
            )[graph_idx]

            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()

        elif model == "grad_node":
            explainer.zero_grad()
            node_grad \
                = torch.abs(F.relu(
                explainer.adj_feat_grad(node_idx_new,
                                        pred_label
                                        [node_idx_new])[1]))[graph_idx]

            node_grad = torch.sum(node_grad, 1)
            actual_adj = np.squeeze(adj.detach().numpy())
            masked_adj = actual_adj * node_grad.cpu().detach().numpy()

        elif model == "Grad-CAM":
            explainer.zero_grad()
            acts, grads = explainer.conv_grad(node_idx_new, pred_label[
                node_idx_new])

            grad_cam_weights = self.grad_cam(acts, grads)[:]
            actual_adj = np.squeeze(adj.detach().numpy())
            scaled_grad_cam_weights =  actual_adj * grad_cam_weights
            masked_adj = scaled_grad_cam_weights

        elif model == "greedy":
            masked_adj = explainer.greedy_search(node_idx_new, pred_label[
                node_idx_new])
        elif model == "random":
            masked_adj = explainer.random_search(node_idx_new, pred_label[
                node_idx_new])
        else:
            explainer.train()
            begin_time = time.time()

            if explainer.__class__.__name__ == "LGLIMExplainer":
                if self.args.fix_nearest_neig:
                    explainer.fix_hop1_nodes(node_idx_new)

                explainer.cal_perturbation_coef(node_idx_new)

                explainer.masked_adj = explainer._masked_adj()
                # transform from 1 dimension to 2 dimension (nb. of nodes *
                # dimension of feature)
                feat_mask = explainer\
                    .split_array(explainer.inv_mapped_feat_stack)

                print("finished perturbed instances in ", time.time() -
                      begin_time)
            begin_time = time.time()

            early_stopping = EarlyStopping(10, verbose=True)
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                # prediction produced from dumped model
                if explainer.__class__.__name__ == "LGLIMExplainer":
                    pred_label, adj_atts, ypred \
                        = explainer(node_idx_new, unconstrained=unconstrained,
                                    final_feat=feat_mask)
                else:
                    ypred, adj_atts = explainer(node_idx_new,
                                                unconstrained=unconstrained)
                if explainer.__class__.__name__ == "LGLIMExplainer":
                    loss = explainer.loss(ypred, pred_label, node_idx_new,
                                          epoch, self.args.loss_target)
                else:
                    loss = explainer.loss(ypred, pred_label,
                                          node_idx_new, epoch)
                loss.backward()
                explainer.optimizer.step()
                mask_density = explainer.mask_density()
                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                    )
                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )

                    if epoch == 0:
                        if self.model.att:
                            # explain node
                            print("adj att size: ", adj_atts.size())
                            adj_att = torch.sum(adj_atts[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float().cuda()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0],
                                "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0]\
                                .cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )

                if model != "exp" and model != "att":
                    break

                if explainer.__class__.__name__ == "LGLIMExplainer":
                    early_stopping(loss, explainer)
                    if early_stopping.early_stop:
                        break

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                if explainer.__class__.__name__ == "LGLIMExplainer":
                    masked_adj = self.generate_masked_adj(explainer,
                                                          "max_prob_class",
                                                          node_idx)
                else:
                    masked_adj = (
                            explainer.masked_adj[0].cpu()
                            .detach().numpy() * sub_adj.squeeze()
                    )
            elif model == "att":
                masked_adj = (
                        explainer.masked_adj[0].cpu().detach().numpy()
                        * sub_adj.squeeze()
                )

        if not graph_mode:

            fidelity_val = self.obtain_fidelity(adj, masked_adj, x, node_idx,
                                            node_idx_new, dataset)
            print("Fidelity: " + str(fidelity_val))

        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", fname)
        return masked_adj

    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes
            from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, dataset=args.dataset)
            for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat,
                                       threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")
        return masked_adjs

    def obtain_top_n_acc(self, pred_group, real_group):
        count = 0
        count_top_k = 0
        for i in range(0, len(pred_group)):
            ground_truth_edg = np.where(real_group[i] == 1)

            top_k=ground_truth_edg[0].shape[0]
            count_top_k += top_k
            top_k_idx=pred_group[i].argsort()[::-1][0:top_k]
            merged_list = ground_truth_edg[0].tolist()
            merged_list += top_k_idx.tolist()
            from collections import Counter

            dct_counter = Counter(merged_list)
            for item in dct_counter.keys():
                if dct_counter[item] == 2:
                    count += 1

        print("accuracy of matched nodes within top " +
              str(count_top_k)+  ": " + str(float(count/count_top_k)))

    def explain_nodes_gnn_stats(self, node_indices, args,
                                graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model,
                         explainer=args.explainer, dataset=args.dataset)
            for node_idx in node_indices
        ]

        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        # =====
        for i, idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat,
                                       threshold_num=20)
            pred, real = self.make_pred_real(masked_adjs[i], new_idx)

            pred_all.append(pred)
            real_all.append(real)
            denoised_feat = np.array([G.nodes[node]["feat"]
                                      for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(
                self.writer,
                G,
                "graph/{}_{}_{}".format(self.args.dataset, model, i),
                identify_self=True,
            )
        # =====

        pred_group = pred_all
        real_group = real_all

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        # =====
        # auc for all edges
        auc_all = roc_auc_score(real_all, pred_all)
        print("total auc: " + str(auc_all))

        # dicrect show of prediction probability
        print(pred_all)
        print(real_all)

        print("Fidelity is: " + str(self.fidelity_val/len(masked_adjs)))
        print("Contrastivity is: " + str(self.contrastivity/len(masked_adjs)))

        self.obtain_top_n_acc(pred_group, real_group)
        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices, explainer, dataset):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            # traverse across all nodes inside the graph

            merged_masked_adj = np.zeros((self.adj.shape[1],
                                          self.adj.shape[1]))
            for node_idx in range(0, self.adj[graph_idx].shape[0]):

                # Each variable masked_adj is a matrix that only the
                # elements on the certain row make sense that the relevance
                # from all other nodes are marked in corresponding elements
                masked_adj = self.explain(node_idx=node_idx,
                                          graph_idx=graph_idx,
                                          graph_mode=True,
                                          explainer=explainer,
                                          dataset=dataset)
                merged_masked_adj[node_idx] = masked_adj[node_idx]
            masked_adj = merged_masked_adj
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs

    # deprecated
    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx],
                                       pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors \
                    = self.extract_neighborhood(idx, graph_idx)
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh",
            tensorboardX.utils.figure_to_image(fig), 0
        )

    # deprecated
    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        self.alpha = self.preds_grad

    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]

        if self.graph_mode:
            sub_label = None
        else:
            sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
            self, ref_feat, ref_adj, ref_node_idx, curr_feat,
            curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0],
                                           curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # grid graph
        elif self.args.dataset in ["syn3", "syn5"]:
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            start = start - 1
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 4] > 0:
                real[start + 1][start + 4] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 5] > 0:
                real[start + 2][start + 5] = 10
            if real[start + 3][start + 6]:
                real[start + 3][start + 6] = 10

            if real[start + 3][start + 4]:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 7] > 0:
                real[start + 4][start + 7] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start + 5][start + 8] > 0:
                real[start + 5][start + 8] = 10
            if real[start + 6][start + 7]:
                real[start + 6][start + 7] = 10
            if real[start + 7][start + 8]:
                real[start + 7][start + 8] = 10

            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


class ExplainModule(nn.Module):
    """
        self.x: equals to Explainer.feat that features for all nodes
               (a matrix),
        self.feat_mask: mask matrix for all features (a tensor),

        self.adj: original adjacency matrix,
        self.mask: mask matrix for adjacency relation "M" in Equation (5),
        self.mask_bias: bias supplied for mask matrix (self.mask),
        self.masked_adj: combine self.adj, self.mask and self.mask_bias to
        establish final masked adjacency,

        self.graph_mode: whether current task belongs to graph explanation,
        self.graph_idx:,

        self.writer: log writer of tensorboard,
        self.args: parameters fof model, which is used to establish prefix
        of log file.
    """
    def __init__(
            self,
            adj,
            x,
            model,
            label,
            args,
            graph_idx=0,
            writer=None,
            use_sigmoid=True,
            graph_mode=False,
    ):
        """

        :param adj:
        :param x:
        :param model:
        :param label:
        :param args:
        :param graph_idx:
        :param writer:
        :param use_sigmoid:
        :param graph_mode:
        """

        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1),
                                                  init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) \
                         - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer \
            = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    # TODO:
    # minimal unit is each feature in a node, instead of each node
    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        """
        generate masked feature matrix
        :param feat_dim: nb. of nodes * dimension of input feature
        :param init_strategy: strategy of mask operation
        :return: masked adjacency matrix (type: Parameter;
        shape: nb. of nodes * dimension of input feature)
        """
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes,
                            init_strategy="normal", const_val=1.0):
        """
        generate masked adjacency matrix

        used private variable inside this class:
        self.args.

        :param feat_dim: dimension of input feature
        :param init_strategy: strategy of mask operation
        :return: masked adjacency matrix (shape: 1*10)
        """
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        """
        Add non-linear activation on self.mask with optional self.mask_bias

        used private variable inside this class:
        self.mask_act,
        self.diag_mask,
        self.mask,
        self.mask_bias,
        self.args,
        self.adj.

        :return:
        """
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    # Todo
    # codes need to be changed mainly
    def forward(self, node_idx, unconstrained=False,
                mask_features=True, marginalize=False):
        """

        inputs:
        self.x,
        self.args,
        self.mask,
        self.use_sigmoid,
        self.feat_mask
        self.model: dumped GNN model
        self.graph_mode,
        self.graph_idx.

        outputs !!!!!:
        self.masked_adj !!!!!

        :param node_idx:
        :param unconstrained:
        :param mask_features:
        :param marginalize:
        :return:
        """
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) \
                if self.use_sigmoid else self.mask
            self.masked_adj = (
                    torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0)
                    * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att

    # Todo
    # codes need to be changed mainly
    def loss(self, pred, pred_label, node_idx, epoch):
        """

        self.graph_mode,
        self.label
        self.mask
        self.mask_act
        self.coeffs,
        self.feat_mask,
        self.use_sigmoid,
        self.masked_adj,
        self.graph_idx,
        self.adj
        self.writer

        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            gt_label_node = self.label \
                if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        feat_mask = (
            torch.sigmoid(self.feat_mask)
            if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj \
            if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                        * (pred_label_t @ L @ pred_label_t)
                        / self.adj.numel()
                        )

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + \
               feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss",
                                   feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss",
                                   mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    # Summarize the density degree in mask operation
    def mask_density(self):
        """
        self._masked_adj,
        self.adj

        :return:
        """
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def adj_feat_grad(self, node_idx, pred_label_node):
        """
        self.model
        self.adj
        self.x
        self.args
        self.label: unused variable
        self.graph_mode
        self.graph_idx

        :param node_idx:
        :param pred_label_node:
        :return:
        """
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])

        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()

        return self.adj.grad, self.x.grad

    def conv_grad(self, node_idx, pred_label_node):
        """
        self.model
        self.adj
        self.x
        self.args
        self.label: unused variable
        self.graph_mode
        self.graph_idx


        :param node_idx:
        :param pred_label_node:
        :return:
        """
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)

        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])

        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()

        return self.model.first_act, self.model.first_conv_grads


    def mask_single_node(self, adj, idx):
        for i in range(0, adj.shape[1]):
            adj[0][idx][i] = 0
            adj[0][i][idx] = 0
            pass
        return adj

    def random_search(self, node_idx, pred_label_node):
        """
        self.model
        self.adj
        self.x
        self.args
        self.label: unused variable
        self.graph_mode
        self.graph_idx


        :param node_idx:
        :param pred_label_node:
        :return:
        """

        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
        else:
            x, adj = self.x, self.adj

        ary_diff_prd = torch.Tensor(
            np.fabs(np.random.uniform(0, 1,
                size=(1, x.shape[1])))).cuda()
        ary_diff_prd = ary_diff_prd * adj
        return np.array(ary_diff_prd[0].cpu().data)

    def greedy_search(self, node_idx, pred_label_node):
        """
        self.model
        self.adj
        self.x
        self.args
        self.label: unused variable
        self.graph_mode
        self.graph_idx

        :param node_idx:
        :param pred_label_node:
        :return:
        """

        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
        else:
            x, adj = self.x, self.adj

        org_pred, _ = self.model(x, adj)
        org_pred = org_pred[self.graph_idx, node_idx, pred_label_node]

        diff_prd = []
        for i in range(0, adj.shape[1]):
            adj_backup = copy.deepcopy(adj)
            adj_backup = self.mask_single_node(adj_backup, i)
            ypred, _ = self.model(x, adj_backup)

            if self.graph_mode:
                logit = ypred[0]
            else:
                logit = ypred[self.graph_idx, node_idx, :]

            diff_prd.append(math.fabs(org_pred - logit[pred_label_node]))

        ary_diff_prd = torch.Tensor(np.array([diff_prd])).cuda()
        ary_diff_prd = ary_diff_prd * adj
        return np.array(ary_diff_prd[0].cpu().data)

    def log_mask(self, masked_adj, node_idx_new, epoch):
        """
        output distribution of variable "mask, feat_mask, masked_adj" into
        "IMAGES/mask/" + "mask/ feat_mask/ adj" in tensorboard

        used private variable inside this class:
        self.mask,
        self.writer,
        self.feat_mask,
        self.masked_adj,
        self.args,
        self.mask_bias

        :param epoch: nb. of epoch of explainer model
        :return: None, related figures are saved into file system and
        tensorboard
        """
        if self.graph_mode:
            pass
        else:
            gt_label_node = self.label[0][node_idx_new]
            for param in self.g_model.parameters():
                param = param.view(param.size(0), self.x.shape[1], self.x.shape[2])
                mask_visual = param.abs().sum(2)

        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(masked_adj, cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx_new,
                feat=self.x[0],
                threshold=0.2,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="mask/graph",
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            # only limited nodes (highly relevant nodes) that surrounds node
            # node_index are retained
            G = io_utils.denoise_graph(
                masked_adj, node_idx_new, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="mask/graph",
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(),
                       cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        """
        Compute gradient based on dumped GNN model, only for gradient-based
        explainer baseline

        used private variable inside this class:
        self.graph_mode,
        self.graph_idx,
        self.adj,
        self.writer,
        self.masked_adj,
        self.x,
        self.args.

        :param node_idx: index of the node that need to be explained
        :param pred_label: specified label value (1 * nb. of nodes)
        :param epoch: nb. of epoch of explainer model
        :param label: unused input parameter, ground truth label of each node
        :return: None, related figures are saved into file system and
        tensorboard
        """
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)
        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        """
        output graph into log/mask directory as PDF files and simultaneously
        save into tensorboard IMAGE/mask/graph (only last epoch in tensorboard)

        used private variable inside this class:
        self.masked_adj,
        self.writer,
        self.graph_mode,
        self.args,
        self.x.

        :param node_idx: index of the node that need to be explained
        :param epoch: nb. of epoch of explainer model
        :param name: prefix of saved path or title in tensorboard
        :param label: unused input parameter, ground truth label of each node
        :return:
        """

        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            # only limited nodes (highly relevant nodes) that surrounds node
            # node_index are retained
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )


class LogisticRegression(nn.Module):
    def __init__(self, dim_input, dim_output, type):
        self.type = type
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(dim_input, dim_output, bias=True)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        if self.type == "NN":
            x = self.sm(x)
        return x


class LGLIMExplainer(ExplainModule):
    # Todo
    def __init__(
            self,
            adj,
            x,
            model,
            label,
            args,
            graph_idx=0,
            node_idx=None,
            num_classes=0,
            writer=None,
            use_sigmoid=True,
            graph_mode=False,
            debug=False,
            dataset=None
    ):
        # No feature inside Syn1 dataset
        super(ExplainModule, self).__init__()

        # determined input data for debug mode
        if debug:
            self.mock_var()
        else:
            self.adj = adj
            self.x = x
            self.label = label

        self.args = args
        self.match_parameters(dataset)

        self.writer = writer
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.model = model
        # g model in LG-IME
        self.g_model = LogisticRegression(self.x.shape[1]*self.x.shape[2],
                                          num_classes, self.args.g_model)

        if torch.cuda.is_available() and args.gpu:
            self.g_model.cuda()

        # diagonal elements are all 0, other elements are all 1
        self.diag_mask = torch.ones(self.x.shape[1], self.x.shape[1]) - \
                         torch.eye(self.x.shape[1])
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.exp_feat = self.map_feat(self.x, "unchange")
        self.exp_adj = self.map_adj(self.adj, self.args.num_gc_layers,
                                    "unchange", node_idx)

        self.mask = self.perturb_adj(self.exp_adj, self.args.adj_mask_stra,
                                     self.sample_num)

        self.feat_mask = self.perturb_feat(self.exp_feat, 
                                           self.args.feature_mask_stra,
                                           self.sample_num)

        self.inv_mapped_feat_stack = self.inv_map_feat(self.feat_mask,
                                                       "unchange")

        self.inv_mapped_adj_stack = self.inv_map_adj(self.adj, self.mask,
                                                     self.exp_adj, "unchange")

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        self.kernel_fn = partial(kernel, kernel_width=25)

        # unused code
        self.mask_act = args.mask_act

        self.scheduler, self.optimizer \
            = train_utils.build_optimizer(args, self.g_model.parameters())

        # output related hyper-parameters
        print("==========")
        print("Mask rate for adjacency matrix is: " + str(self.mask_adj_rate))
        print("Mask rate for feature is: " + str(self.mask_feat_rate))
        print("Nb. of perturbed instances is: " + str(self.sample_num))
        print("Nb. of epoches is: " + str(args.num_epochs))
        print("Learning rate is: " + str(self.args.lr))
        print("L1 regularization is: " + str(self.args.l1))
        print("Whether the nearest neighbors are fixed: " + str(
            self.args.fix_nearest_neig))
        print("Target of classes for loss calculation: " +
              self.args.loss_target)
        print("Model type of g function: " + self.args.g_model)
        print("Strategy for feature mask: " + self.args.feature_mask_stra)
        print("Strategy for adj mask: " + self.args.adj_mask_stra)
        print("Rate of energy consumed by feature perturbation: " +
              str(self.args.feature_energy))
        print("Rate of energy consumed by topology perturbation: " +
              str(self.args.topology_energy))
        print("==========")

    def match_parameters(self, dataset):
        # useful parameters transplanted from class ExplainModule

        if dataset == "syn1":
            # probability of output 1 (1 indicates remaining this element)
            # i.e. more bigger more high probability to remain, more low
            # probability to mask
            self.mask_adj_rate = 0.5
            self.mask_feat_rate = 0.8
            self.sample_num = 400
            self.args.num_epochs = 1000
            self.args.lr = 1e-2
            self.args.l1 = 0
            self.args.fix_nearest_neig = False
            self.args.loss_target = "target_class" # target_class or all_class
            self.args.g_model = "NN" # LR or NN
            self.args.feature_mask_stra = "None" # remove_feat or None
            self.args.adj_mask_stra = "remove_edge"
            self.args.feature_energy = 0
            self.args.topology_energy = 1
        else:
            print("Unmatched dataset and corresponding parameters !")
            exit()

    def mock_var(self):
        """
        Mock output of this class and they can be fed for visulization module,
        all mocked values are stored as private variables inside current class

        :return: None
        """

        self.x = torch.Tensor([[[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])

        self.adj = torch.Tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]])

        self.label = torch.Tensor([[1, 1, 2, 3, 4, 1, 1, 2]])

        # No matter node or graph classification, the product of LG-IME is
        # unique matrix with shape 1 * n
        self.mask = nn.Parameter(
            torch.Tensor([[0, 0.1, 0.2, 0.22, 0.32, 0.42, 0.62, 0.42]]))

        # In particular practice, self.adj should be equals to rechability
        # matrix of self.adj instead of self.adj itself !!!!!
        self.masked_adj = self.adj * self.mask

        # Equals to self.g_model.lr.weight
        self.feat_mask = nn.Parameter(
            torch.Tensor([0, 0.1, 0.2, 0.22,
                          0.9, 0.1, 0.1, 0.22,
                          0.0, 0.1, 0.2, 0.9,
                          0.0, 0.1, 0.2, 0.9,
                          0.96, 0.12, 0.1, 0.32,
                          0.32, 0.42, 0.62, 0.42,
                          0.14, 0.52, 0.43, 0.36,
                          0.31, 0.21, 0.562, 0.752]))

    # Summarize the density degree in mask operation
    def mask_density(self):
        """
        self._masked_adj,
        self.adj

        :return:
        """

        mask_sum = 0
        masked_adj = self._masked_adj()
        for sig_masked_adj in masked_adj:

            if self.args.gpu:
                sig_masked_adj = sig_masked_adj.cpu()

            existent_nodes = self.adj == 1
            # masked entries in a reachability matrix (no matter whether
            # actually exist)
            may_masked_nodes = sig_masked_adj == 0
            # masked entries in a reachability matrix (only the ones
            # that actually exist but be masked by perturbation operation)
            masked_nodes = existent_nodes & may_masked_nodes
            mask_sum += np.count_nonzero(np.where(
                masked_nodes==True, 1, 0))
        mask_sum = mask_sum /len(masked_adj)
        adj_sum = np.count_nonzero(self.adj)
        return torch.Tensor([mask_sum / adj_sum])

    def map_feat(self, feat, strategy):
        """
        original feature representations are transformed into explainable
        feature domain.
        Note: It's a transformation from one matrix to another one matrix.

        :param feat: feature matrix
        :param strategy: strategy for feature transformation
        :return: mapped feature matrix (shape: nb. of nodes *
        dimension of feature; type: Tensor)
        """
        # with torch.no_grad():
        if strategy == "unchange":
            mapped_feat = feat
        else:
            print("Undefined Strategy !")
            exit()
        return mapped_feat

    # Todo
    # Only support one node explanation now
    def map_adj(self, adj, nb_hop, strategy, node_idx):
        """
        input adjacency matrix is transformed into reachability matrix,
        and the definition of reachability equals to the hop limitation inside
        dumped GNNs.
        Note: It's a transformation from one matrix to another one matrix.

        :param adj: adjacency matrix
        :param nb_hop: number of hop limitation
        :param strategy: strategy for feature transformation
        :param node_idx: index of node to be explained
        :return: mapped reachability matrix (shape: 1 * nb. of
        nodes; type: np.array)
        """
        # with torch.no_grad():
        if strategy == "unchange":
            reach_mat = graph_utils.neighborhoods(adj=adj, n_hops=nb_hop,
                                                  use_cuda=use_cuda)
        else:
            print("Undefined Strategy !")
            exit()

        if node_idx is not None:
            result = np.array([reach_mat[0][node_idx]])
        else:
            result = np.array([reach_mat[0]])
        return result

    def inv_map_feat(self, feat, strategy):
        """
        input feature representation is inversely transformed back to
        original feature domain

        :param feat: feature matrix
        :param strategy: strategy for feature transformation
        :return: mapped feature matrix (shape: nb. of
        instances * (nb. of nodes * dimension of feature), type: parameter)
        """
        # with torch.no_grad():
        if strategy == "unchange":
            mapped_feat = feat
        else:
            print("Undefined strategy !")
            exit()
        return mapped_feat

    def inv_map_adj(self, adj, mask, exp_adj, strategy):
        """
        input adjacency matrix is inversely transformed back to original
        adjacency matrix.
        It's quite important that perturbation in function "perturb_adj" on
        node A probably influence the other node B that is connected with A
        before perturbation operation. In our

        :param adj: adjacency matrix
        :param mask: masked entries in reachability matrix
        :param exp_adj: original and complete reachability matrix
        :param strategy: strategy for reverse operation
        :return: inversely mapped *feature* matrix (shape: nb. of
        instances * (1 * nb. of nodes * nb. of nodes), type: Tensor)
        """
        # with torch.no_grad():
        if strategy == "unchange":
            exp_adj = exp_adj[0]
            mask = np.array(mask.data)
            inv_adj_stack = []

            for sig_instance in mask:

                # actual entries in a reachability matrix
                existent_nodes = exp_adj == 1
                # masked entries in a reachability matrix (no matter whether
                # actually exist)
                may_masked_nodes = sig_instance == 0
                # masked entries in a reachability matrix (only the ones
                # that actually exist but be masked by perturbation operation)
                masked_nodes = existent_nodes & may_masked_nodes
                masked_node_ins \
                    = np.nonzero(np.where(masked_nodes==True, 1, 0))

                # iteration on each perturbed reachability matrix
                inv_map_adj = copy.deepcopy(adj)
                for masked_node_idx in masked_node_ins[0]:
                    inv_map_adj[0][masked_node_idx][:] = 0
                    for i in range(0, inv_map_adj.shape[1]):
                        inv_map_adj[0][i][masked_node_idx] = 0
                    # ensure diagonal elements remain "1" value
                    inv_map_adj[0][masked_node_idx][masked_node_idx] = 1

                inv_adj_stack.append(inv_map_adj)
        else:
            print("Undefined strategy !")
            exit()
        return inv_adj_stack

    def perturb_feat(self, feat, strategy, nb_instances):
        """
        perturb input feature matrix under the certain strategy
        The perturbation transformation is 1:n.

        :param feat: feature matrix
        :param strategy: strategy for masking feature matrix
        :param nb_instances: nb. of perturbed instances
        :return: sampled perturbation feature matrices (shape: nb. of
        instances * (nb. of nodes * dimension of feature), type: parameter)
        """
        # with torch.no_grad():
        if strategy == "remove_feat":
            feat_mask = torch.Tensor(
                np.random.binomial(1, self.mask_feat_rate,
                                   (nb_instances, feat.shape[1]*
                                    feat.shape[2])))

        elif strategy == "None":
            feat_mask \
                = torch.Tensor(np.ones((nb_instances,
                                                     feat.shape[
                                                         1]*feat.shape[2])))

        else:
            print("Undefined Strategy !")
            exit()
        return feat_mask

    def perturb_adj(self, adj, strategy, nb_instances):
        """
        perturb input adjacency matrix under the certain strategy and target
        node.
        Note: The perturbation transformation is from one node to
        nb_instances nodes.

        :param adj: adjacency/reachability tensor for target node
        :param strategy: strategy for masking adjacency matrix
        :param nb_instances: nb. of sampled instances
        :return: sampled perturbation adjacency/reachability tensors (
        shape: nb. of instances * nb. of nodes)
        """
        # with torch.no_grad():
        if strategy == "remove_edge":
            mask = np.random.binomial(1, self.mask_adj_rate,
                                      (nb_instances, adj.shape[1]))
            mask = torch.Tensor(mask)
        elif strategy == "None":
            mask \
                = torch.Tensor(np.ones((nb_instances,adj.shape[1])))
        else:
            print("Undefined Strategy !")
            exit()
        return mask

    def fix_hop1_nodes(self, node_idx):
        reach_mat = self.adj
        mask_mat = self.mask
        hop_adj = reach_mat[0][node_idx]
        for i in range(0, self.mask.shape[0]):
            fixed_idxs = np.where(hop_adj==1)
            mask_mat[i][fixed_idxs] = 1
        self.inv_mapped_adj_stack = self.inv_map_adj(self.adj, self.mask,
                                                     self.exp_adj, "unchange")

    def softmax(self, X):
        """
        calculate softmax output
        :param X: a vector
        :return: operated vector
        """
        return np.exp(X) / np.sum(np.exp(X))

    def cosine_distance(self, matrix1,matrix2):
        """
        more closer to 1, more relevant; otherwise, more irrelevant
        :param matrix1: one vector
        :param matrix2: another vector
        :return: cosine distance between them with a specified kernel function
        """

        from sklearn.metrics.pairwise import pairwise_distances
        mix_mat = sp.sparse.csr_matrix([matrix1, matrix2])

        distances =  pairwise_distances(
            mix_mat, mix_mat[0], metric='cosine').ravel() * 100

        weights = self.kernel_fn(distances)
        return weights

    def cal_perturbation_coef(self, node_idx):
        """
        calculate the coefficients for each perturbed instance that more
        masked signal on more small hop results on more small coefficients

        :param node_idx: index of target node
        :return: self.instance_weight - obtained coefficients depending on
        each perturbed instance (shape: nb. of instances)
        each perturbed instance (shape: nb. of instances)
        """

        # Energy consumed by feature perturbation
        pertubred_features \
            = (self.x.reshape((1, self.x.shape[1] * self.x.shape[2])) \
              * self.feat_mask).cpu().detach().numpy()

        original_features \
            = self.x.reshape((1, self.x.shape[1] * self.x.shape[2]))\
            .cpu().detach().numpy()

        weight_hop = []
        for i in range(1, self.args.num_gc_layers+1):
            weight_hop.append(float(math.pow(self.args.num_gc_layers, 1)/i))
        weight_hop = self.softmax(weight_hop)

        self.instance_weight = []
        for j in range(0, self.mask.shape[0]):
            coef = 0
            for i in range(1, self.args.num_gc_layers+1):
                reach_mat = graph_utils.neighborhoods(adj=self.adj, n_hops=i,
                                                      use_cuda=use_cuda)

                # actual entries in a reachability matrix
                actual_nodes = reach_mat[0][node_idx]
                masked_nodes = actual_nodes * np.array(self.mask[j].data)

                distance = self.cosine_distance(actual_nodes, masked_nodes)[1]
                perturbed_rate = distance * weight_hop[i-1]
                coef += perturbed_rate

            feature_energy = self.cosine_distance(original_features[0],
                                      pertubred_features[j])[1]

            if coef == 1.0: coef = 0
            if feature_energy == 1.0: feature_energy = 0

            self.instance_weight.append(self.args.topology_energy * coef +
                                        self.args.feature_energy *
                                        feature_energy)

    # Todo
    # The correctness of it deservers test
    def split_array(self, ary_vector):
        """
        linearized vector with only 1 dimension is reshaped into 2
        dimension form
        :param ary_vector: vector used to reshape (shape: )
        :return: reshaped vector (shape: nb. of instances * 1 * nb. of nodes
        dimension of feature)
        """
        with torch.no_grad():
            splited_vector_stack = []
            for sig_vector in ary_vector:
                reshaped_vec \
                    = torch.Tensor([sig_vector.detach().numpy().reshape((
                                    self.x.shape[1], self.x.shape[2]))])

                reshaped_vec \
                    = reshaped_vec.cuda() if self.args.gpu else reshaped_vec

                splited_vector_stack.append(reshaped_vec)

            return splited_vector_stack

    # Todo
    # codes need to be changed mainly
    def loss(self, pred, pred_label, node_idx, epoch, stragtgy="target_class",
             l1_regu=False):
        """
        calculate the loss value produced from explain and original mode
        maninly

        :param pred: prediction made by current explain model
        :param pred_label: the label predicted by the original and dumped model
        :param node_idx: node index that deserves explanation
        :param epoch: current epoch index
        :param stragtgy: strategy to determine what all or only
        target label class participates in backpropagation
        :return:
        """
        if self.args.g_model == "NN":
            criterion = nn.BCELoss()
        else:
            criterion = torch.nn.MSELoss()

        # go through all perturbed instances
        loss = None
        gt_label_node = self.label if self.graph_mode \
            else self.label[0][node_idx]

        # loss is only calculated from target class (same as class in label)
        # or all classes
        if stragtgy == "target_class":
            lst_classes = [gt_label_node]
        elif stragtgy == "all_class":
            lst_classes = range(0, pred.shape[1])
        else:
            print("Undifined Strategy")

        for i in range(0, pred.shape[0]):
            if loss is None:

                loss = self.instance_weight[i] \
                       * criterion(pred[i][lst_classes],
                                   pred_label[i][lst_classes])
            else:
                loss += self.instance_weight[i]\
                        * criterion(pred[i][lst_classes],
                                    pred_label[i][lst_classes])

        if self.args.l1 > 0.0:
            regularization_loss = 0
            for param in self.g_model.parameters():
                # torch.sum(abs(param))
                regularization_loss += torch.norm(param, 1)
            loss += regularization_loss
        return loss

    # Todo
    # transformation and re-transformation operation need to be adapted
    def _masked_adj(self):
        """
        combine mask matrix and adjacency matrix

        :return: masked adjacency matrix
        """
        # with torch.no_grad():
        masked_adj_stack = []

        for sig_instance in self.inv_mapped_adj_stack:

            sym_mask = sig_instance[0]
            sym_mask = (sym_mask + sym_mask.t()) / 2
            sym_mask = sym_mask.cuda() if self.args.gpu else sym_mask
            adj = self.adj.cuda() if self.args.gpu else self.adj
            masked_adj = adj * sym_mask
            # The elements on the diagonal lines must be "0", thus take
            # no effect on prediction of dumped GNNs
            masked_adj = masked_adj * self.diag_mask
            masked_adj = masked_adj.cuda() if self.args.gpu else masked_adj
            masked_adj_stack.append(masked_adj)
        return masked_adj_stack

    def forward(self, node_idx, unconstrained=False, mask_features=True,
                marginalize=False, final_feat=None):
        """

        :param node_idx: index of node used to explain
        :param unconstrained: unused
        :param mask_features: determine whether feature is masked
        :param marginalize: unused
        :return: softmax ouput from f function; attention ouput from f
        function (keep empty currently); output of g function
        """
        exp_prd = self.exp_forward()
        org_prd, org_att = self.org_forward(node_idx, final_feat,
                                            mask_features)
        return org_prd, org_att, exp_prd

    def exp_forward(self):
        """
        make prediciton for each perturbed instance through g function with
        explicable feature

        :return: output of g function (shape: nb. of perturbed instances *
        nb. of label class)
        """
        # archieve masked reachability vector for each sampled instance
        masked_adj = np.array(self.mask.data)
        lst_expended_masked_adj = []
        for i in range(0, masked_adj.shape[0]):
            lst_expended_masked_adj.append(masked_adj[i].repeat(
                self.x.shape[2]))
        masked_adj = torch.Tensor(np.array(lst_expended_masked_adj))

        linearized_exp_feat_without_adj \
            = self.exp_feat.reshape((1, self.x.shape[1] * self.x.shape[2])) \
                              * self.feat_mask
        linearized_exp_feat = linearized_exp_feat_without_adj * masked_adj
        linearized_exp_feat = linearized_exp_feat.cuda() \
            if self.args.gpu else linearized_exp_feat
        out = self.g_model(linearized_exp_feat)
        return out

    def org_forward(self, node_idx, final_feat, mask_features=True):
        """
        make prediciton for each perturbed instance through f function with
        non-explicable feature

        :param node_idx: index of node used to explain
        :param mask_features: determine whether feature masking trchnique
        is applied
        :return: softmax ouput from f function; att - keep empty currently
        """

        # print("org_forward starting point:" + str(time.time()))

        # It's different from GNNExplainer that dumped GNN is changed with
        # training process
        with torch.no_grad():
            x = self.x.cuda() if self.args.gpu else self.x
            feat_mask = final_feat
            res_stack, adj_att_stack = [], []

            for i in range(0, len(self.masked_adj)):
                sig_feat_mask = feat_mask[i]
                sig_masked_x = x * sig_feat_mask
                ypred, adj_att = self.model(sig_masked_x, self.masked_adj[i])
                if self.graph_mode:
                    res = nn.Softmax(dim=0)(ypred[0])
                else:
                    node_pred = ypred[self.graph_idx, node_idx, :]
                    res = nn.Softmax(dim=0)(node_pred)
                res_stack.append(res)
                adj_att_stack.append(adj_att)

            return res_stack, adj_att_stack


if __name__ == "__main__":
    LGLIMExplainer(None, None, None, None, None, 3)
