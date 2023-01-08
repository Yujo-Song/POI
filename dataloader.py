import os
import re
import time
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoader(object):

    def __init__(self, args):
        self.args = args
        self.data_name = args.data_name
        # self.use_pretrain = args.use_pretrain
        # self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.home_data_dir = os.path.join(args.data_dir, args.data_name, args.home_name)
        self.out_data_dir = os.path.join(args.data_dir, args.data_name, args.out_of_town_name)

        self.home_train_file = os.path.join(self.home_data_dir, 'train.txt')
        self.home_test_file = os.path.join(self.home_data_dir, 'test.txt')
        #out部分只需要poi的数据，后续进行处理
        self.out_train_file = os.path.join(self.out_data_dir, 'train.txt')
        self.out_test_file = os.path.join(self.out_data_dir, 'test.txt')

        self.home_user_kg_file = os.path.join(self.home_data_dir, "user_kg.txt")
        self.home_poi_kg_file = os.path.join(self.home_data_dir,'home_poi_kg.txt')
        self.out_poi_kg_file = os.path.join(self.home_data_dir, "out_poi_kg.txt")
        #out部分不需要知识图谱的帮助，后续会根据poi的距离构建地理知识图谱

        #处理数据：home
        self.cf_home_train_data, self.home_train_user_dict = self.load_cf(self.home_train_file)
        self.cf_home_test_data, self.home_test_user_dict = self.load_cf(self.home_test_file)
        # 处理数据：out;out的poi应该包括所有的poi，所以train和test要加起来，使用相同的poi列表
        self.cf_out_train_data, self.out_train_user_dict = self.load_cf(self.out_train_file)
        self.cf_out_test_data, self.out_test_user_dict = self.load_cf(self.out_test_file)

        # self.cf_out_data = self.cf_out_train_data + self.cf_out_test_data

        self.statistic_cf()

        # if self.use_pretrain == 1:
        #     self.load_pretrained_data()

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        self.kg_user_data = self.load_kg(self.home_user_kg_file)
        self.kg_home_poi_data = self.load_kg(self.home_poi_kg_file)
        self.kg_out_poi_data = self.load_kg(self.out_poi_kg_file)

        self.construct_home_data(self.kg_user_data, self.kg_home_poi_data, self.kg_out_poi_data)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    # 加载文件的数据
    def load_cf(self, filename):
        user = []
        poi = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines[1:]:
            tmp = l.rstrip()
            inter = [int(i) for i in re.split('\t|  ', tmp)]
            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))  # set函数为去重，得到一个dict，转化为list

                for item_id in item_ids:
                    user.append(user_id)
                    poi.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        poi = np.array(poi, dtype=np.int32)
        return (user, poi), user_dict


    def statistic_cf(self):

        self.n_users = len(list(set(self.cf_home_train_data[0]))) + len(list(set(self.cf_home_test_data[0])))
        self.n_home_pois = len(list(set(self.cf_home_train_data[1]))) + len(list(set(self.cf_home_test_data[1])))
        self.n_out_pois = len(list(set(self.cf_out_train_data[1]))) + len(list(set(self.cf_out_test_data[1])))
        self.n_pois = self.n_home_pois + self.n_out_pois

        self.n_cf_home_train = len(self.cf_home_train_data[0])
        self.n_cf_home_test = len(self.cf_home_test_data[0])
        self.n_cf_out_train = len(self.cf_out_train_data[0])
        self.n_cf_out_test = len(self.cf_out_test_data[0])

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep='\t', skiprows=[0], names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_home_data(self, kg_user_data, kg_home_poi_data, kg_out_poi_data):

        #user
        # self.user = np.unique(list(kg_user_data['h']))
        # self.remap_user = {index : value for index, value in enumerate(self.user)}
        # self.inverse_remap_user = {v: k for k,v in self.remap_user.items()}
        # self.user = [index for index in self.remap_user.keys()]
        # self.n_users = len(self.user)
        #poi
        # self.home_poi = np.unique(list(kg_poi_data['h']))
        # self.remap_home_poi = {index + self.n_users : value for index, value in enumerate(self.home_poi)}
        # self.inverse_remap_home_poi = {v : k for k, v in self.remap_home_poi.items()}
        # self.home_poi = [index for index in self.remap_home_poi.keys()]
        # self.n_home_pois = len(self.home_poi)
        #entity
        # self.user_entities = np.unique(list(kg_user_data['t']))
        # self.poi_entities = np.unique(list(kg_poi_data['t']))
        # self.home_entities = self.user_entities + self.poi_entities
        # self.remap_home_emtities = {index + self.n_users + self.n_home_pois: value for index, value in enumerate(self.home_entities)}
        # self.inverse_remap_home_emtities = {v : k for k, v in self.remap_home_emtities.items()}
        # self.home_entities = [index for index in self.remap_home_emtities.keys()]
        # self.n_home_entities = len(self.home_entities)

        #三个kg中user和poi和entity都是从0开始，重新映射
        #按照user home_poi out_poi home_entity out_entity 的顺序映射

        self.n_home_entities = len(list(set(list(kg_home_poi_data['t']))))
        self.n_out_entities = len(list(set(list(kg_out_poi_data['t']))))

        kg_user_data['t'] += self.n_users + self.n_home_pois + self.n_out_pois
        kg_home_poi_data['h'] += self.n_users
        kg_home_poi_data['t'] += self.n_users + self.n_home_pois + self.n_out_pois
        kg_out_poi_data['h'] += self.n_users
        kg_out_poi_data['t'] += self.n_users + self.n_home_pois + self.n_out_pois

        kg_home_data = pd.concat([kg_user_data, kg_home_poi_data], axis=0, ignore_index=True, sort=False)
        # add inverse kg data
        n_relations = max(kg_home_data['r']) + 1
        inverse_kg_home_data = kg_home_data.copy()
        inverse_kg_home_data = inverse_kg_home_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_home_data['r'] += n_relations
        self.kg_home_data = pd.concat([kg_home_data, inverse_kg_home_data], axis=0, ignore_index=True, sort=False)


        kg_data = pd.concat([kg_user_data, kg_home_poi_data, kg_out_poi_data], axis=0, ignore_index=True, sort=False)
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        self.kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)


        # kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = self.n_home_entities + self.n_out_entities
        # self.n_users_entities = self.n_users + self.n_entities

        self.cf_home_train_data = (self.cf_home_train_data[0].astype(np.int32), np.array(list(map(lambda d: d + self.n_users, self.cf_home_train_data[1]))).astype(np.int32))
        self.cf_home_test_data = (self.cf_home_test_data[0].astype(np.int32), np.array(list(map(lambda d: d + self.n_users, self.cf_home_test_data[1]))).astype(np.int32))

        self.home_train_user_dict = {k : np.unique(v + self.n_users).astype(np.int32) for k, v in
                                self.home_train_user_dict.items()}
        self.home_test_user_dict = {k : np.unique(v + self.n_users).astype(np.int32) for k, v in
                               self.home_test_user_dict.items()}

        self.cf_out_train_data = (self.cf_out_train_data[0].astype(np.int32),
                                   np.array(list(map(lambda d: d + self.n_users, self.cf_out_train_data[1]))).astype(
                                       np.int32))
        self.cf_out_test_data = (self.cf_out_test_data[0].astype(np.int32),
                                  np.array(list(map(lambda d: d + self.n_users, self.cf_out_test_data[1]))).astype(
                                      np.int32))

        self.out_train_user_dict = {k: np.unique(v + self.n_users).astype(np.int32) for k, v in
                                     self.out_train_user_dict.items()}
        self.out_test_user_dict = {k: np.unique(v + self.n_users).astype(np.int32) for k, v in
                                    self.out_test_user_dict.items()}

        # add interactions to kg data
        # cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        # cf2kg_train_data['h'] = self.cf_train_data[0]
        # cf2kg_train_data['t'] = self.cf_train_data[1]
        #
        # inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        # inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        # inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        # kg_data = kg_user_data + kg_poi_data

        # add inverse kg data
        #计算关系的数量
        # n_relations = len(np.unique(list(kg_data['r'])))
        # inverse_kg_data = kg_data.copy()
        # inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        # inverse_kg_data['r'] += n_relations    #每个新关系标号都加上一个总关系数，变为新关系编号
        # self.kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # self.n_relations = n_relations * 2
        self.n_kg_train = len(self.kg_data)

        # 训练数据也要重新编号
        #self.n_user, self.n_home_train_poi
        # self.cf_home_train_data = (np.array(list(map(lambda d: self.inverse_remap_user[d], self.cf_home_train_data[0]))).astype(np.int32), np.array(list(map(lambda d: self.inverse_remap_home_poi[d], self.cf_home_train_data[1]))).astype(np.int32))
        # self.cf_home_test_data = (
        # np.array(list(map(lambda d: self.inverse_remap_user[d], self.cf_home_test_data[0]))).astype(np.int32),
        # np.array(list(map(lambda d: self.inverse_remap_home_poi[d], self.cf_home_test_data[1]))).astype(np.int32))
        #
        # for k, v in self.home_train_user_dict.items():
        #     k = self.inverse_remap_user(k)
        #     v = [self.inverse_remap_home_poi(i) for i in v]
        #     self.home_train_user_dict[k] = v
        # for k, v in self.home_test_user_dict.items():
        #     k = self.inverse_remap_user(k)
        #     v = [self.inverse_remap_home_poi(i) for i in v]
        #     self.home_test_user_dict[k] = v

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)

        for row in self.kg_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items


    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=self.n_users + self.n_home_pois, high=self.self.n_users + self.n_pois, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def sample_neg_users_for_i(self, user_dict, item_id, n_sample_neg_items):

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_user_id = np.random.randint(low=0, high=self.self.n_users, size=1)[0]
            if item_id not in user_dict[neg_user_id] and neg_user_id not in sample_neg_items:
                sample_neg_items.append(neg_user_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = user_dict.keys()
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item, batch_neg_user = [], [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        for i in batch_pos_item:
            batch_neg_user += self.sample_neg_users_for_i(user_dict, i ,1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        batch_neg_user = torch.LongTensor(batch_neg_user)
        return batch_user, batch_pos_item, batch_neg_item, batch_neg_user


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break
            #poi的编号在user后面，所以采样的时候从user开始
            tail = np.random.randint(low=self.n_users, high=highest_neg_idx + self.n_users, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    # def load_pretrained_data(self):
    #     pre_model = 'mf'
    #     pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
    #     pretrain_data = np.load(pretrain_path)
    #     self.user_pre_embed = pretrain_data['user_embed']
    #     self.item_pre_embed = pretrain_data['item_embed']
    #
    #     assert self.user_pre_embed.shape[0] == self.n_users
    #     assert self.item_pre_embed.shape[0] == self.n_items
    #     assert self.user_pre_embed.shape[1] == self.args.embed_dim
    #     assert self.item_pre_embed.shape[1] == self.args.embed_dim