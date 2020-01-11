import argparse

def training_args():
    """
    Retrieves and parses the command line arguments
    """

    # Create a parser
    parser = argparse.ArgumentParser()

    # args.data_dir path to the directory that contains the data
    parser.add_argument('data_dir', type=str, help='directory that contains the data')

    # args.save_dir path to the directory to save the checkpoints in
    parser.add_argument('--save_dir', type=str, default='/home/workspace/ImageClassifier',
                        help='directory to save checkpoints')

    # args.arch the CNN model to use for classification
    parser.add_argument('--arch', type=str, default='vgg',
                        help='chosen model architecture')

    # args.learning_rate the rate at which the model is adapted to the problem
    parser.add_argument('--learning_rate', type=float, default='0.001',
                        help='rate at which the model is adapted to the problem')

    # args.hidden_units the number of hidden neural network nodes
    parser.add_argument('--hidden_units', type=int, default='512',
                        help='number of hidden neural network nodes')

    # args.epochs the number of passes through the entire training dataset
    parser.add_argument('--epochs', type=int, default='20',
                        help='number of passes through the entire training dataset')

    # args.gpu use GPU for inference
    parser.add_argument('--gpu', action="store_true", dest="gpu",
                        help='use GPU for inference')

    # returns parse arguments
    return parser.parse_args()


def prediction_args():
    """
    Retrieves and parses the command line arguments
    """

    # Create a parser
    parser = argparse.ArgumentParser()

    # args.input path to the image
    parser.add_argument('input', type=str, help='path to the image')

    # args.checkpoint the model checkpoint to use
    parser.add_argument('checkpoint', type=str, help='the model checkpoint to use')

    # args.top_k returns a number of top K most likely classes
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of top K most likely classes')

    # args.category_names mapping of categories to real names
    parser.add_argument('--category_names', type=str, help='mapping of categories to real names')

    # args.gpu use GPU for inference
    parser.add_argument('--gpu', action="store_true", dest="gpu",
                        help='use GPU for inference')

    # returns parse arguments
    return parser.parse_args()
