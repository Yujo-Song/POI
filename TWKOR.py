import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import numpy as np
import pandas as pd
import collections
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings


class TWKOR(nn.Module):

    def __init__(self, args, data):

        super(TWKOR, self).__init__()
        #预训练，暂时用不到
        # self.use_pretrain = args.use_pretrain

        self.n_users = data.n_users
        self.n_home_pois = data.n_home_pois
        self.n_out_pois = data.n_out_pois
        self.n_pois = self.n_home_pois + self.n_out_pois
        self.n_home_entities = data.n_home_entities
        self.n_out_entities = data.n_out_entities
        self.n_entities = self.n_home_entities + self.n_out_entities
        self.n_relations = data.n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.tway_ckg_lambda = args.two_way_ckg_lambda
        self.tway_ckg_threshold = args.two_way_ckg_threshold

        # 嵌入
        # 用于训练kg
        self.user_poi_entities_embed = nn.Embedding(self.n_users + self.n_pois + self.n_entities, self.embed_dim)
        # home用于训练GAT
        self.home_user_poi_entities_embed = nn.Embedding(self.n_users + self.n_users + self.n_home_pois + self.n_home_pois + self.n_home_entities, self.embed_dim)
        # out训练GCN
        self.out_poi_embed_1 = nn.Embedding(self.n_out_pois, self.embed.dim)
        # out训练双视角poi
        self.out_poi_embed_2 = nn.Embedding(self.n_out_pois + self.n_out_pois, self.embed_dim)
        #关系嵌入
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        # nn.init.xavier_uniform_(self.user_embed_a.weight)
        # # nn.init.xavier_uniform_(self.user_embed_p.weight)
        # nn.init.xavier_uniform_(self.home_poi_embed_a.weight)
        # nn.init.xavier_uniform_(self.home_poi_embed_p.weight)
        # nn.init.xavier_uniform_(self.out_poi_embed.weight)
        # nn.init.xavier_uniform_(self.out_poi_embed_p.weight)
        nn.init.xavier_uniform_(self.user_poi_entities_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)


        # if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
        #     other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
        #     nn.init.xavier_uniform_(other_entity_embed)
        #     entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
        #     self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        # else:
        #     nn.init.xavier_uniform_(self.entity_user_embed.weight)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        self.laplacian_type = args.laplacian_type
        #矩阵的行列 user和poi有两个节点
        self.A_row = self.n_users + self.n_users + self.n_home_pois + self.n_home_pois + self.n_home_entities
        self.A_in = nn.Parameter(torch.saprse.FloatTensor(self.A_row, self.A_row))



    #construct two-way ckg
    def construct_tway_ckg(self,data):
        '''
        data.kg_data : all kg_data,train and test
        data.cf_home_train_data
        data.cf_home_test_data
        data.train_user_dict
        data.test_user_dict

        return:pos_kg:pos_user and neg_poi,
               neg_kg:neg_user and pos_poi
        '''
        # user list
        users = list(set(data.cf_home_train_data[0]))
        home_pois = list(set(data.cf_home_train_data[1]))

        '''
        添加interaction信息
        user和poi之间存在interaction：设定relation为1，并且relation weight设为1
        user和poi之间没有interaction：计算相似度，超过阈值则存在被动关系，设relation为0：，weight为相似度
        '''

        #对两个基本kg中的关系remap
        kg_home_data = data.kg_home_data['r'] + 2
        # calculate weight
        # pos_kg: if have interaction： ralation is 1
        pos_kg = pd.DataFrame(np.ones((data.n_cf_home_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        pos_kg['h'] = data.cf_home_train_data[0]
        pos_kg['t'] = data.cf_home_train_data[1]

        inverse_pos_kg = pd.DataFrame(np.ones((data.n_cf_home_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_pos_kg['h'] = data.cf_home_train_data[1]
        inverse_pos_kg['t'] = data.cf_home_train_data[0]

        self.pos_kg = pd.concat([kg_home_data, pos_kg, inverse_pos_kg], ignore_index=True)
        pois_in_count = self.pos_kg.shape[0] / 2  # 所有交互中poi的总入度

        # neg_kg: if not have interaction： doing calculation
        n_u = len(users)
        n_p = len(home_pois)
        n_neg_interaction = n_u * n_p - data.n_cf_home_train
        # neg_kg = pd.DataFrame(np.zeros((n_neg_interaction, 3), dtype=np.int32), columns=['h', 'r', 't'])
        neg_kg = pd.DataFrame(columns=['h', 'r', 't'])
        self.neg_weight_matrix = np.zeros((self.n_users, self.n_home_pois), dtype=np.int32)
        for user in users:
            user_embed = self.user_poi_entities_embed(user)
            for poi in home_pois:
                #similarity
                poi_users = self.pos_kg['t'][self.pos_kg['h'] == poi]
                poi_users_array = pd.to_numeric(poi_users['t'])
                neigh_users_embed = self.user_poi_entities_embed(poi_users_array.tolist())
                user_mean_embed = sum(neigh_users_embed)/len(neigh_users_embed)
                similarity = cosine_similarity(user_embed, user_mean_embed)

                #hot degree
                poi_in_count = len(poi_users_array.tolist) #当前poi的总入度
                hot_degree = poi_in_count/pois_in_count

                score_result = similarity + self.tway_ckg_lambda * hot_degree
                if score_result >= self.tway_ckg_threshold:
                    neg_kg.loc[len(neg_kg.index)] = [user, 0, poi]
                    neg_kg.loc[len(neg_kg.index)] = [poi, 0, user]
                    self.neg_weight_matrix[user, poi] = score_result

        self.neg_kg = pd.concat([kg_home_data, neg_kg], ignore_index=True)

        self.kg_home_data = pd.concat([kg_home_data, pos_kg, neg_kg], ignore_index=True)

        # reconstruct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_home_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.user_poi_entities_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.user_poi_entities_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.user_poi_entities_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            #r == 1表示pos_user和neg_poi之间存在交互，关系强度为1
            if r == 1:
                rows = [e[0] for e in ht_list]
                cols = [e[1] for e in ht_list]
                vals = [1] * len(rows)

                cols_2 = cols + self.n_users + self.n_home_pois
                adj = sp.coo_matrix((vals, (rows, cols_2)), shape=(self.A_row, self.A_row))
            # r == 0表示neg_user和pos_poi之间存在交互,关系强度存在matrix中
            elif r == 0:
                rows = [e[0] for e in ht_list]
                cols = [e[1] for e in ht_list]

                rows_2 = rows + self.n_users
                cols_2 = cols + self.n_users

                vals = []
                for i in range(len(rows)):
                    vals.append(self.neg_weight_matrix[rows[i], cols[i]])
                adj = sp.coo_matrix((vals, (rows_2, cols_2)), shape=(self.A_row, self.A_row))
            #其他关系是user、poi和entity之间的关系，pos和neg都存在
            else:

                rows = [e[0] for e in ht_list]
                cols = [e[1] for e in ht_list]
                rows_2 = []
                cols_2 = []
                for i in rows:
                    if i < self.n_users:
                        rows_2.append(i) #pos_user
                        rows_2.append(i + self.n_users) #neg_user
                        cols_2.append(cols[i] + self.n_users + self.n_home_pois)
                        cols_2.append(cols[i] + self.n_users + self.n_home_pois)
                    if self.n_users <= i < self.n_users + self.n_home_pois:
                        rows_2.append(i) #pos_poi
                        rows_2.append(i + self.n_users + self.n_home_pois) #neg_poi
                        cols_2.append(cols[i] + self.n_users + self.n_home_pois)
                        cols_2.append(cols[i] + self.n_users + self.n_home_pois)
                for i in cols:
                    if i < self.n_users:
                        cols_2.append(i) #pos_user
                        cols_2.append(i + self.n_users) #neg_user
                        rows_2.append(cols[i] + self.n_users + self.n_home_pois)
                        rows_2.append(cols[i] + self.n_users + self.n_home_pois)
                    if self.n_users <= i < self.n_users + self.n_home_pois:
                        cols_2.append(i) #pos_poi
                        cols_2.append(i + self.n_users + self.n_home_pois) #neg_poi
                        rows_2.append(cols[i] + self.n_users + self.n_home_pois)
                        rows_2.append(cols[i] + self.n_users + self.n_home_pois)
                vals = [1] * len(rows)
                adj = sp.coo_matrix((vals, (rows_2, cols_2)), shape=(self.A_row, self.A_row))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        A_in_tensor = self.convert_coo2tensor(A_in.tocoo())
        self.A_in.data = A_in_tensor
        self.A_in.requires_grad = False

    def create_attention_matrix(self):
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def create_home_embed(self):
        # 初始化双节点的嵌入
        user = [i for i in range(self.n_users)]
        poi = [i for i in range(self.n_users, self.n_users + self.n_home_pois)]
        entity = [i for i in range(self.n_users + self.n_home_pois, self.n_users + self.n_home_pois + self.n_home_entities)]

        pos_user = [i for i in range(self.n_users)]
        neg_user = [i for i in range(self.n_users, self.n_users * 2)]
        pos_poi = [i for i in range(self.n_users * 2, self.n_users * 2 + self.n_home_pois)]
        neg_poi = [i for i in range(self.n_users * 2 + self.n_home_pois, self.n_users * 2 + self.n_home_pois * 2)]
        entity_2 = [i for i in range(self.n_users * 2 + self.n_home_pois * 2, self.n_users * 2 + self.n_home_pois * 2 + self.n_home_entities)]

        self.home_user_poi_entities_embed.weight[pos_user] = self.user_poi_entities_embed.weight[user]
        self.home_user_poi_entities_embed.weight[neg_user] = self.user_poi_entities_embed.weight[user]
        self.home_user_poi_entities_embed.weight[pos_poi] = self.user_poi_entities_embed.weight[poi]
        self.home_user_poi_entities_embed.weight[neg_poi] = self.user_poi_entities_embed.weight[poi]
        self.home_user_poi_entities_embed.weight[entity_2] = self.user_poi_entities_embed.weight[entity]

    def create_out_embed(self):

        out_poi = [i for i in range(self.n_users + self.n_home_pois, self.n_users + self.n_home_pois + self.n_out_pois)]

        self.out_poi_embed_1.weight = self.user_poi_entities_embed.weight[out_poi]


    def calc_cf_home_embeddings(self):

        self.create_home_embed()
        ego_embed = self.home_user_poi_entities_embed.weight  # pos user and neg poi and entity
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed

    def calc_cf_out_embeddings(self, gcn_embed):

        #计算out poi 的pos和neg嵌入
        ...

    def transform(self, pos_embed, neg_embed):
        ...

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, user_neg_ids, gcn_embed):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        home_all_embed = self.calc_cf_home_embeddings()  # (n_users + n_home_pois, n_entities, concat_dim)

        home_user_pos_embed = home_all_embed[user_ids]  # (cf_batch_size, concat_dim)
        home_user_neg_embed = home_all_embed[user_ids + self.n_users]

        out_user_pos_embed, out_user_neg_embed = self.transform(home_user_pos_embed, home_user_neg_embed)

        out_user_neg_pos_embed = out_user_pos_embed[user_neg_ids]
        out_user_neg_neg_embed = out_user_neg_embed[user_neg_ids]

        self.out_poi_embed_2 = self.calc_cf_out_embeddings(gcn_embed)
        #第一个pos表示采样的正例，第二个pos表示正节点
        #neg同样
        item_pos_pos_embed = self.out_poi_embed_2[item_pos_ids - self.n_users - self.n_home_pois]  # (cf_batch_size, concat_dim)
        item_pos_neg_embed = self.out_poi_embed_2[item_pos_ids - self.n_users - self.n_home_pois + self.n_out_pois]  # (cf_batch_size, concat_dim)
        item_neg_pos_embed = self.out_poi_embed_2[item_neg_ids - self.n_users - self.n_home_pois]  # (cf_batch_size, concat_dim)
        item_neg_neg_embed = self.out_poi_embed_2[item_neg_ids - self.n_users - self.n_home_pois + self.n_out_pois]  # (cf_batch_size, concat_dim)

        # Equation (12)
        # 正例和正例
        pos_pos_score = torch.sum(out_user_pos_embed * item_pos_neg_embed, dim=1) + torch.sum(out_user_neg_embed * item_pos_pos_embed, dim=1)# (cf_batch_size)
        # 正例和负例
        pos_neg_score = torch.sum(out_user_pos_embed * item_neg_neg_embed, dim=1) + torch.sum(out_user_neg_embed * item_neg_pos_embed, dim=1)  # (cf_batch_size)
        # 负例和正例
        neg_pos_score = torch.sum(out_user_neg_pos_embed * item_pos_neg_embed, dim=1) + torch.sum(out_user_neg_neg_embed * item_pos_pos_embed, dim=1)
        # Equation (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_pos_score - (1/2) * pos_neg_score - (1/2) * neg_pos_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(out_user_pos_embed) + _L2_loss_mean(out_user_neg_embed) + \
                  _L2_loss_mean(out_user_neg_pos_embed) + _L2_loss_mean(out_user_neg_neg_embed) + \
                  _L2_loss_mean(item_neg_pos_embed) + _L2_loss_mean(item_pos_neg_embed) + \
                  _L2_loss_mean(item_neg_pos_embed) + _L2_loss_mean(item_neg_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        # all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        # user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        # item_embed = all_embed[item_ids]                # (n_items, concat_dim)
        #
        # # Equation (12)
        # cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        # return cf_score


    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)