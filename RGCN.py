import torch

import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn



class RelationalGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(RelationalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        'weight is for all (W0 in the paper; other two are separate weights'
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_dc = Parameter(torch.Tensor(in_features, out_features))
        self.weight_dd = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight_dc)
        torch.nn.init.xavier_uniform_(self.weight_dd)

    def forward(self, input, adj):
        '''
        TODO:

        :param x: x is a list of features: whole, dd and dc (same shape)
        :param adj: adj is a list of features: whole, dd and dc (same shape)
        :return:
        '''
        'adj will be a list of adj, list of 2'
        input = F.dropout(input, self.dropout, self.training)

        'all_adj is all of the adj'
        all_adj = adj[0].add(adj[1]).sub(torch.eye(adj[0].shape[0]).to_sparse())

        # for over-all
        support = torch.mm(input, self.weight)
        output = torch.spmm(all_adj, support)

        # for dc
        support_dc = torch.mm(input, self.weight_dc)
        output_dc = torch.spmm(adj[0], support_dc)

        # for dd
        support_dd = torch.mm(input, self.weight_dd)
        output_dd = torch.spmm(adj[1], support_dd)

        # import pdb;pdb.set_trace()
        # add all
        # final_output = (output + output_dc + output_dd)/3
        # final_output = torch.add(output,torch.div(output_dc + output_dd,2))
        # final_output = ((output_dc + output_dd)/2 + output)/2


        # works for tf-idf
        final_output = (output + output_dc + output_dd)/3

        return final_output

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        # self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        # self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc1 = RelationalGraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = RelationalGraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = RelationalGraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        'adj is a list of adj vars'
        'mu = hidden layer, logvar = last layer'

        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar