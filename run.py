from preprocess import *
from network import *
from utils import *
import argparse
import pandas as pd
import scanpy.api as sc


if __name__ == "__main__":
    random_seed = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]

    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default = "Park", type = str)
    parser.add_argument("--model", default = "multinomial")
    parser.add_argument("--mode", default="indirect")
    parser.add_argument("--adaptive", default = True)
    parser.add_argument("--dims", default = [500, 256, 64, 32])
    parser.add_argument("--highly_genes", default = 500)
    parser.add_argument("--alpha", default = 0.001, type = float)
    parser.add_argument("--sigma", default = 1.0, type = float)
    parser.add_argument("--theta", default=1.0, type=float)
    parser.add_argument("--learning_rate", default = 0.0001, type = float)
    parser.add_argument("--batch_size", default = 256, type = int)
    parser.add_argument("--update_epoch", default = 10, type = int)
    parser.add_argument("--pretrain_epoch", default = 1000, type = int)
    parser.add_argument("--funetrain_epoch", default = 2000, type = int)
    parser.add_argument("--noise_sd", default = 1.5)
    parser.add_argument("--error", default = 0.001, type = float)
    parser.add_argument("--gpu_option", default = "0")

    args = parser.parse_args()

    X, Y = prepro(args.dataname)
    X = np.ceil(X).astype(np.int)
    count_X = X

    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = normalize(adata, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["Group"])
    high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
    count_X = count_X[:, high_variable]
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    cluster_number = int(max(Y) - min(Y) + 1)

    result = []

    for seed in args.random_seed:
        np.random.seed(seed)
        tf.reset_default_graph()
        scClustering = scDMFK(args.dataname, args.dims, cluster_number, args.alpha, args.sigma, args.learning_rate, args.noise_sd,
                             adaptative=args.adaptive, model=args.model, mode=args.mode)
        scClustering.pretrain(X, count_X, size_factor, args.batch_size, args.pretrain_epoch, args.gpu_option)
        accuracy, ARI, NMI = scClustering.funetrain(X, count_X, Y, size_factor, args.batch_size, args.funetrain_epoch, args.update_epoch, args.error)
        result.append([args.dataname, seed, accuracy, ARI, NMI])

    output = np.array(result)
    output = pd.DataFrame(output, columns=["dataset name", "random seed", "accuracy", "ARI", "NMI"])
    print(output)




