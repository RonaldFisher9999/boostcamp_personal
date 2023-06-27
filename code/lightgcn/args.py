import argparse
# import subprocess


# 조건에 맞는 입력이면 True, 나머지는 False
def bool_type_casting(x):
    return str(x).lower() in ("true", "1", "yes")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", default=42, type=int, help="Set random seed. default=42"
    )
    
    parser.add_argument(
        "--use_cuda_if_available",
        default=True,
        type=bool_type_casting,
        help="Use GPU. default=True",
    )

    parser.add_argument(
        "--data_dir",
        default="../data/dkt",
        type=str,
        help="Set data dir. default=../data/dkt",
    )
    
    parser.add_argument(
        "--model_dir",
        default="./lightgcn/models/",
        type=str,
        help="Set model dir to save and load. default=./lightgcn/models/",
    )

    parser.add_argument(
        "--output_dir",
        default="./lightgcn/outputs/",
        type=str,
        help="Set output dir. default=./lightgcn/outputs/",
    )

    parser.add_argument(
        "--valid_size", default=0.1, type=float, help="Set valid data size. default=0.1"
    )

    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,
        help="Set embedding dimension. default=128"
    )
    
    parser.add_argument(
        "--n_layers", default=2, type=int, help="Set number of layers. default=2"
    )

    parser.add_argument(
        "--train_n_epochs",
        default=10,
        type=int,
        help="Set number of epochs for train data. default=10"
    )
    
    parser.add_argument(
        "--valid_n_epochs",
        default=2,
        type=int,
        help="Set number of epochs for valid/test data. default=2"
    )
    
    parser.add_argument(
        "--train_lr",
        default=0.01,
        type=float,
        help="Set learning rate for train data. defaut=0.01"
    )
    
    parser.add_argument(
        "--valid_lr",
        default=0.01,
        type=float,
        help="Set learning rate for train data. defaut=0.01"
    )
    
    parser.add_argument(
        "--use_best",
        default=False,
        type=bool_type_casting,
        help="Use best model for valid/test training. default=False",
    )
    

    ### custom ###
    # res = subprocess.run(["git", "config", "user.name"], stdout=subprocess.PIPE)
    # git_username = res.stdout.strip().decode()
    # if git_username == "":
    #     git_username = "unknown"
    # parser.add_argument("--user_name", default=git_username, type=str, help="user name")

    args = parser.parse_args()

    return args
