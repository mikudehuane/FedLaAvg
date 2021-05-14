import argparse

parser = argparse.ArgumentParser(
        description=
        """
        Simulate a federated learning process with give algorithm.
        """
    )
parser.add_argument('-rn', '--run_name', default=None,
                    help="run name to be specified.If not give, generated according to parameters.")
# dataset
parser.add_argument('-ds', '--dataset', default='cifar10', choices=('cifar10', 'sentiment140', 'mnist', 'ali'),
                    help='dataset to use')
# focused parameters
parser.add_argument('-alg', '--algorithm', default='lastavg', choices=('gd', 'sgd', 'lastavg', 'fedavg', 'waitavg'),
                    help="algorithm to be used")
parser.add_argument('-N', '--num_total_clients', default=1000, type=int, help="total number of clients")
parser.add_argument('-be', '--beta', default=0.1, type=float, help="proportion of selected clients")
parser.add_argument('-C', '--num_local_iterations', default=10, type=int, help="number of iterations per round")
parser.add_argument('-E', default=10, type=int, help="controlling availability, defined in paper")
parser.add_argument('--momentum', default=0.0, type=float, help="momentum for fedavg (not for sgd gd)")
parser.add_argument('-mu', '--mu_FedProx', default=0.0, type=float, help="proxy term form FedProx")
parser.add_argument('--availability_file', default="client_availability", type=str,
                    help="the filter name without extension recording the client availability, "
                         "works only for alibaba dataset")
# learning rate parameters
lr_group = parser.add_argument_group(title="learning rate")
lr_group.add_argument('--lr_strategy', default='exp', choices=('const', 'exp', 'multi'), help="learning rate strategy")
lr_group.add_argument('-lr', '--init_lr', default=0.01, type=float, help="initial learning rate")
lr_group.add_argument('--lr_decay', default=0.9999, type=float, help="learning rate decay for exp")
lr_group.add_argument('--lr_dstep', default=1, type=int, help='learning rate decay every <?> rounds')
lr_group.add_argument('--lr_indicator', default="?", help="for multistep decay, specify to make each run distinct")
lr_group.add_argument('--lr_config', default=None, help="learning rate configuration file to be loaded for multistep")
# parameters controlling logging
print_group = parser.add_argument_group(title="control print")
print_group.add_argument('-pe', '--print_every', default=1, type=int, help="control printings without computation")
print_group.add_argument('-te', '--tb_every', default=20, type=int, help="control printings with computation")
print_group.add_argument('-ce', '--checkpoint_every', default=100, type=int, help="control checkpoint interval")
print_group.add_argument('-se', '--statistics_every', default=100, type=int, help="control statistics interval")
print_group.add_argument('--max_test', default=10000, type=int, help="maximum number of samples to be tested when logging")
# uncared parameters
parser.add_argument('--batch_size', default=-1, type=int, help="batch size for each client")
parser.add_argument('--batch_size_when_test', default=100, type=int, help="batch size when testing the model")
parser.add_argument('--num_rounds', default=50000, type=int, help="number of rounds")
parser.add_argument('--num_non_update_rounds', default=0, type=int,
                    help="for FedLaAvg, do not update in the first <?> rounds, only collect gradients")
parser.add_argument('--balanced', default=0, choices=(0, 1), type=int, help="1 if partition dataset with equal #sample")
parser.add_argument('--shuffle', default=1, choices=(0, 1), type=int, help="1 if shuffle the dataset each epoch")
parser.add_argument("--filter_clients", default=-1, type=int,
                     help="filter clients possessing data <= this threshold, "
                     "work only for sentiment140 (include SGD and GD) / ali (only for fedavg lastavg)")
parser.add_argument("--filter_clients_up", default=2**32, type=int,
                    help="filter clients with more than <?> samples")
parser.add_argument('--scale_in_it', default=0, choices=(0, 1), type=int,
                    help="if set to 1, scale C rather than the update")
parser.add_argument('--num_threads', default=1, type=int, help="number of threads for clients parallel")
# image classification parameters
image_group = parser.add_argument_group("Image classification customized parameters")
image_group.add_argument('--alpha', default=-1, type=float, help="controlling availability, defined in paper")
image_group.add_argument('-nm', '--normalize_mean', default=0.5, type=float, help='target mean when normalizing image data')
image_group.add_argument('-ns', '--normalize_std', default=0.5, type=float, help='target std when normalizing image data')
image_group.add_argument('-ni', '--num_iid', default=0, type=int, help='each client will possess NUM_IID IID distributed image data')
image_group.add_argument('-stg', '--strategy', default='time', choices=('time', 'number', 'mix', 'block'),
                         help='when simulate availability, unbalance with time or number of clients')
# model parameters
ml_model_group = parser.add_argument_group('Machine learning model')
ml_model_group.add_argument('--model_ori', default=None,
                            help="model id specified for the give algorithm")
ml_model_group.add_argument('--nlp_algorithm', default='embedding', choices=('bow', 'embedding'),
                            help='NLP algorithm')
ml_model_group.add_argument('--glove_model', default='glove.840B.300d',
                            help='glove model file name without extension')
# sentiment140
sentiment140_group = parser.add_argument_group("Sentiment140 customized parameters")
sentiment140_group.add_argument("-am", "--availability_model", default="blocked", choices=("blocked", "modeled_mid"),
                                help="indicate how to get the availability of clients")
sentiment140_group.add_argument('-fv', '--force_variation', default=0, choices=(0, 1), type=int,
                                help="whether to drop samples to form a distribution variation with time")
sentiment140_group.add_argument('--fv_num_blocks', default=24, type=int,
                                help="num_blocks passed to time_variation_()")
sentiment140_group.add_argument('--fv_min_prop', default=0.0, type=float,
                                help="min_prop passed to time_variation_()")
sentiment140_group.add_argument('--fv_max_prop', default=1.0, type=float,
                                help="max_prop passed to time_variation_()")
