import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run TWKOR.")

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='yelp',
                        help='Choose a dataset from {yelp, foursquare}')
    parser.add_argument('--home_name', nargs='?', default='Indianapolis',
                        help='Choose a home country')
    parser.add_argument('--out_of_town_name', nargs='?', default='Philadelphia',
                        help='Choose a out-of-town country')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    # parser.add_argument('--use_pretrain', type=int, default=1,
    #                     help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    # parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
    #                     help='Path of learned embeddings.')
    # parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
    #                     help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')
    #
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')


    parser.add_argument('--gcn_hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--gcn_hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gcn_epochs', type=int, default=100, help='Number of epochs to train.')

    #
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')
    #
    parser.add_argument('--two_way_ckg_lambda', type=float, default=1e-5,
                        help='Lambda when constructing Two_Way_CKG.')
    parser.add_argument('--two_way_ckg_threshold', type=float, default=1e-5,
                        help='Threshold when constructing Two_Way_CKG.')
    #
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')
    #
    args = parser.parse_args()

    return args