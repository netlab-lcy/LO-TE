import argparse
# TODO: add running config here, we could set two type of config for link weight routing and MCF routing
def get_arg():
    parser = argparse.ArgumentParser(description='NN-simulator')

    # Add config here
    # NN model config
    parser.add_argument('--input-state-dim', type=int, default=8, 
        help='Input state dimension')
    parser.add_argument('--head-num', type=int, default=1, 
        help='Number of attention heads')
    parser.add_argument('--hidden-dim', type=int, default=8,
        help='Hidden units of GAT model')
    
    
    

    # running config
    parser.add_argument('--use-cuda', action='store_true', default=False,
        help='Use cuda to speed up algorithm.')
    parser.add_argument('--training-epochs', type=int, default=10,
        help='number of training epochs')
    parser.add_argument('--max-flow-num', type=int, default=10000000,
        help='Maximum number of flows in one input graph for the model inference') 
    parser.add_argument('--num-sample-process', type=int, default=0,
        help='Maximum number of sampling processing, 0: never use multiprocessing')
    parser.add_argument('--init-te-solution', default="his",
        help='Initial TE solution, [lb|his]') # deprecated
    parser.add_argument('--objective', default="max_throughput",
        help='TE objective, [min_weight|max_throughput|min_mlu]. This setting will be applied in solution finetuning phase.')
    parser.add_argument('--train-data-dir', default="./data/train",
        help='Training data directory')
    parser.add_argument('--test-data-dir', default="./data/test",
        help='Test data directory')
    parser.add_argument('--log-dir', default="test",
        help='name of model and log dir.')
    parser.add_argument('--model-load-dir', default=None,
        help='Load trained model from target dir')
    parser.add_argument('--training-mode', default="SL",
        help='mode: [SL|USL] -> SL: supervised learning, USL: unsupervised learning.')
    parser.add_argument('--alpha', type=float, default=10,
        help='Factor of maximum selected paths.')
    parser.add_argument('--T', type=int, default=1,
        help='Iteration steps of the path decision')
    parser.add_argument('--K', type=int, default=1,
        help='Sampling times for policy gradient loss')
    
    args = parser.parse_args()

    return args