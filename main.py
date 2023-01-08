import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from dataloader import *
from parser1.parser import *
from TWKOR import *
from RGCN import *
from utils1.metrics import *
from utils1.model_helper import *


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoader(args)
    # if args.use_pretrain == 1:
    #     user_pre_embed = torch.tensor(data.user_pre_embed)
    #     item_pre_embed = torch.tensor(data.item_pre_embed)
    # else:
    #     user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = TWKOR(args, data)
    gcn_model = GCNModelVAE(args.embed_dim, args.gcn_hidden1, args.gcn_hidden2, args.gcn_dropout)
    # if args.use_pretrain == 2:
    #     model = load_model(model, args.pretrain_model_path)

    model.to(device)
    gcn_model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    gcn_optimizer = optim.Adam(gcn_model.parameters(), lr=args.gcn_lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train kg
        time1 = time()
        kg_total_loss = 0

        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time2 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_pois)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                print('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, iter, n_kg_batch, time() - time2, kg_batch_loss.item(), kg_total_loss / iter))
        print('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time1, kg_total_loss / n_kg_batch))

        #construct two_way ckg
        model.construct_tway_ckg(data)
        print('two-way ckg have been constructed')

        #attention matrix
        model.create_attention_matrix()

        #out poi的初始嵌入（transR处理后）
        model.create_out_embed()

        # train GCN
        time3 = time()
        hidden_emb = None
        for iter in range(args.gcn_epochs):
            t = time.time()
            gcn_model.train()
            gcn_optimizer.zero_grad()

            # import pdb;pdb.set_trace

            recovered, mu, logvar = gcn_model(model.out_poi_embed_1, [adj_norm_cd, adj_norm_dd])
            gcn_loss = loss_function_relation(preds=recovered, labels=(adj_label_cd, adj_label_dd),
                                          mu=mu, logvar=logvar, n_nodes=n_nodes,
                                          norm=(norm_cd, norm_dd), pos_weight=(pos_weight_cd, pos_weight_dd))

            gcn_loss.backward()
            gcn_optimizer.step()

            hidden_emb = mu.data.numpy()

            if np.isnan(gcn_loss.cpu().detach().numpy()):
                print('ERROR (GCN Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, args.gcn_epochs))
                sys.exit()

        print('GCN Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, args.gcn_epochs, time() - time3, cf_total_loss / n_cf_batch))


        # train cf
        time4 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_home_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time5 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, cf_batch_neg_user = data.generate_cf_batch(data.out_train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_neg_user = cf_batch_neg_user.to(device)

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, cf_batch_neg_user, hidden_emb, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                print('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time4, cf_batch_loss.item(), cf_total_loss / iter))
        print('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time5, cf_total_loss / n_cf_batch))



        # update attention
        time6 = time()
        h_list = model.h_list.to(device)
        t_list = model.t_list.to(device)
        r_list = model.r_list.to(device)
        relations = list(model.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        print('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time6))

        print('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time7 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            print(('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time7, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg'])))
            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch



if __name__ == '__main__':
    args = parse_args()
    train(args)
    # predict(args)