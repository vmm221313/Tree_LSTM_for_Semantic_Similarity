import argparse

#python main.py --lr 0.05 --wd 0.0001 --optim adagrad --batchsize 25 --freeze_embed --epochs 30
def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # data arguments
    parser.add_argument('--data', default='data/sick/',
                        help='path to dataset')
    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='test',
                        help='Name to identify experiment')
    # model arguments
    #parser.add_argument('--model_type', required=True, help ='type of model')
    parser.add_argument('--model_type', default="mdep", help ='type of model')
    parser.add_argument('--input_dim', default=300, type=int,
                        help='Size of sparse word vector')
    parser.add_argument('--mem_dim', default=150, type=int, #150
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=50, type=int, #50
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='Number of classes in dataset')
    parser.add_argument('--freeze_embed', action='store_false',
                        help='Freeze word embeddings')
    parser.add_argument('--load_model', action='store_true', #ADDED
                        help='Load a saved model')
    parser.add_argument('--evaluate', action='store_true', #ADDED
                        help='evaluate a model')
    parser.add_argument('--saved_model', help='Name of saved model') 

    # training arguments
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.05, type=float, #0.01
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args

# '''
# #python main.py --lr 0.05 --wd 0.0001 --optim adagrad --batchsize 25 --freeze_embed --epochs 30
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
#     # data arguments
#     parser.add_argument('--data', default='../data/sick/',
#                         help='path to dataset')
#     parser.add_argument('--glove', default='../data/glove/',
#                         help='directory with GLOVE embeddings')
#     parser.add_argument('--save', default='../checkpoints/',
#                         help='directory to save checkpoints in')
#     parser.add_argument('--expname', type=str, default='test',
#                         help='Name to identify experiment')
#     # model arguments
#     #parser.add_argument('--model_type', required=True, help ='type of model')
#     parser.add_argument('--model_type', default="mdep", help ='type of model')
#     parser.add_argument('--input_dim', default=300, type=int,
#                         help='Size of sparse word vector')
#     parser.add_argument('--mem_dim', default=150, type=int, #150
#                         help='Size of TreeLSTM cell state')
#     parser.add_argument('--hidden_dim', default=50, type=int, #50
#                         help='Size of classifier MLP')
#     parser.add_argument('--num_classes', default=5, type=int,
#                         help='Number of classes in dataset')
#     parser.add_argument('--freeze_embed', action='store_false',
#                         help='Freeze word embeddings')
#     parser.add_argument('--load_model', action='store_true', #ADDED
#                         help='Load a saved model')
#     parser.add_argument('--evaluate', action='store_true', #ADDED
#                         help='evaluate a model')
#     parser.add_argument('--saved_model', help='Name of saved model') 
#
#     # training arguments
#     parser.add_argument('--epochs', default=20, type=int,
#                         help='number of total epochs to run')
#     parser.add_argument('--batchsize', default=25, type=int,
#                         help='batchsize for optimizer updates')
#     parser.add_argument('--lr', default=0.05, type=float, #0.01
#                         metavar='LR', help='initial learning rate')
#     parser.add_argument('--wd', default=1e-4, type=float,
#                         help='weight decay (default: 1e-4)')
#     parser.add_argument('--sparse', action='store_true',
#                         help='Enable sparsity for embeddings, \
#                               incompatible with weight decay')
#     parser.add_argument('--optim', default='adagrad',
#                         help='optimizer (default: adagrad)')
#     # miscellaneous options
#     parser.add_argument('--seed', default=123, type=int,
#                         help='random seed (default: 123)')
#     cuda_parser = parser.add_mutually_exclusive_group(required=False)
#     cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
#     cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
#     parser.set_defaults(cuda=True)
#
#     args = parser.parse_args()
#     return args
# '''
