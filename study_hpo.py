from HPO_B.hpob_handler import HPOBHandler
from HPO_B.methods.random_search import RandomSearch
from HPO_B.methods.botorch import GaussianProcess
import matplotlib.pyplot as plt
import multiprocessing as mp
from DE import DE_search
from fsbo import FSBO, get_input_dim
import numpy as np
import scipy
import gpytorch
import time
import pickle
import configs as conf

# For memory management
import gc

def store_object(obj, obj_name):
    with open(obj_name, "wb") as fp:
        pickle.dump(obj, fp)

def load_object(obj_name):
    with open(obj_name, "rb") as fp:
        return pickle.load(fp)

# created as a stub for parallel evaluations.
def evaluation_worker(hpob_hdlr, method, args):
    search_space, dataset, seed, n_trials = args
    print(search_space, dataset, seed, n_trials)
    res = []
    try:
        t_start = time.time()
        res = hpob_hdlr.evaluate(method,
                                  search_space_id=search_space,
                                  dataset_id=dataset,
                                  seed=seed,
                                  n_trials=n_trials)
        t_end = time.time()
        print(search_space, dataset, seed, n_trials, "Completed in", t_end - t_start, "s")
    # This exception needs to be ignored due to issues with GP fitting the HPO-B data
    except gpytorch.utils.errors.NotPSDError:
        print("Ignoring the error and not recording this as a valid evaluation combination")
        res = []
    return (search_space, dataset, seed, n_trials), res

def get_all_combinations(hpob_hdlr, n_trials):
    # A total of 430 combinations are present in this if all seeds are used.
    seed_list = ["test0", "test1", "test2", "test3", "test4"]
    evaluation_list = []
    for search_space in hpob_hdlr.get_search_spaces():
        for dataset in hpob_hdlr.get_datasets(search_space):
            for seed in ["test2"]:  # seed_list: # use this for running on all possible seeds
                evaluation_list += [(search_space, dataset, seed, n_trials)]

    return evaluation_list

def evaluate_combinations(hpob_hdlr, method, keys_to_evaluate):

    print("Evaluating for", method)

    evaluation_list = []
    for key in keys_to_evaluate:
        search_space, dataset, seed, n_trials = key
        evaluation_list += [(search_space, dataset, seed, n_trials)]

    performance = []
    run_i = 0
    for eval_instance in evaluation_list:
        result = evaluation_worker(hpob_hdlr, method, eval_instance)
        performance.append(result)
        run_i = run_i + 1
        print("Completed Running", run_i, end="\n")
        gc.collect()

    return performance

def evaluate_DE(hpob_hdlr, keys_to_evaluate):
    performance = []
    mse = []
    variance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = DE_search(input_dim=input_dim)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res
        mse += [method.mse_acc]
        variance += [method.variance_acc]

    store_object(mse, "mse_32x32_E1000_l0_02_random_start")
    store_object(variance, "variance_32x32_E1000_l0_02_random_start")

    plt.plot(np.mean(np.array(mse), axis=0))
    plt.legend(["mse"])
    plt.show()
    plt.plot(np.mean(np.array(variance), axis=0))
    plt.legend(["variance"])
    plt.show()

    return performance

def evaluate_FSBO(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space_id, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space_id, dataset)
        method_fsbo = FSBO(search_space_id, input_dim=input_dim,
                          latent_dim=32, batch_size=70, num_batches=50)
        res = evaluate_combinations(hpob_hdlr, method_fsbo, keys_to_evaluate=[key])
        performance += res

    return performance

def plot_rank_graph(n_keys=2, n_trials=3):
    # Loading previous outputs
    gp_performance = load_object("./optimization_results/gp_performance")[:n_keys, :n_trials]
    rs_performance = load_object("./optimization_results/rs_performance")[:n_keys, :n_trials]
    de = load_object("./optimization_results/de_performance_32x32_E1000_l0_02_random_start")[:n_keys, :n_trials]
    dkt = load_object("./optimization_results/dkt_performance_500_003_cosA")[:n_keys, :n_trials]
    ####
    # Creating a rank graph for all above methods
    performance = np.stack((rs_performance, gp_performance, de, dkt), axis=-1)
    # Since rank data ranks in the increasing order, we need to multiply by -1
    rg = scipy.stats.rankdata(-1 * performance, axis=-1)
    rank_rs = np.mean(rg[:, :, 0], axis=0)
    rank_gp = np.mean(rg[:, :, 1], axis=0)
    rank_de = np.mean(rg[:, :, 2], axis=0)
    rank_dkt = np.mean(rg[:, :, 3], axis=0)
    plt.figure(101)
    plt.plot(rank_rs)
    plt.plot(rank_gp)
    plt.plot(rank_de)
    plt.plot(rank_dkt)
    legend = ["RS Rank",
              "GP Rank",
              "DE Rank [32,32] ep=1000 lr=0.02 (Rand.)",
              "DKT Rank [32x4] ft=500 lr=0.03 (CosAnne)"
              ]
    plt.legend(legend)
    plt.show()


# For studying guassina pre-traning is not required.
# Hence directly using the meta test set
def study_gaussian(n_trials):
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    method = GaussianProcess(acq_name="EI")
    all_keys = get_all_combinations(hpob_hdlr, n_trials)
    performance = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=all_keys)
    gp_keys = []
    gp_performance = []
    for key, performance_list in performance:
        if performance_list:
            gp_keys += [key]
            gp_performance += [performance_list]
    gp_performance = np.array(gp_performance, dtype=np.float32)
    print("GP performance shape:", gp_performance.shape)
    # Store results
    store_object(gp_keys, "./optimization_results/gp_keys")
    store_object(gp_performance, "./optimization_results/gp_performance")

def study_random_search(n_trials):
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    # Loading previous outputs
    gp_keys = load_object("./optimization_results/gp_keys")
    # Evaluate Random search
    method = RandomSearch()
    rs_performance = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=gp_keys)
    rs_performance = [performance_list for _, performance_list in rs_performance]
    rs_performance = np.array(rs_performance, dtype=np.float32)
    # Store results
    store_object(rs_performance, "./optimization_results/rs_performance")

def study_DE(n_trails):
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    # Loading previous outputs
    gp_keys = load_object("./optimization_results/gp_keys")
    # Evaluate DE
    de_performance = evaluate_DE(hpob_hdlr, keys_to_evaluate=gp_keys)
    de_performance = [performance_list for _, performance_list in de_performance]
    de_performance = np.array(de_performance, dtype=np.float32)
    # Store results
    store_object(de_performance, "./optimization_results/de_performance_32x32_E1000_l0_02_random_start")

def study_FSBO(conf_fsbo, n_keys, n_trails):

    if conf_fsbo.pretrain:
        hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")
        print("Pretrain FSBO with all search spaces (training meta dataset)")
        for search_space_id in hpob_hdlr.get_search_spaces():
            meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
            meta_val_data = hpob_hdlr.meta_validation_data[search_space_id]
            method_fsbo = FSBO(search_space_id, input_dim=get_input_dim(meta_train_data),
                               latent_dim=32, batch_size=70, num_batches=50)
            method_fsbo.train(meta_train_data, meta_val_data)

    if conf_fsbo.evaluate:
        print("Evaluate test set (training meta dataset)")
        hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
        gp_keys = load_object("./optimization_results/gp_keys")[:n_keys]
        # Change the number of trails in the key to the required amount
        for i in range(len(gp_keys)):
            key_new = list(gp_keys[i])
            key_new[3] = n_trails
            gp_keys[i] = tuple(key_new)
        dkt_performance = evaluate_FSBO(hpob_hdlr, keys_to_evaluate=gp_keys)
        dkt_performance = [performance_list for _, performance_list in dkt_performance]
        dkt_performance = np.array(dkt_performance, dtype=np.float32)
        store_object(dkt_performance, "./optimization_results/dkt_performance_500_003_cosA")


def main():
    n_trails = 100
    n_keys = 77

    if conf.evaluate_gaussian:
        study_gaussian(n_trails)

    if conf.evaluate_random:
        study_random_search(n_trails)

    if conf.evaluate_DE:
        study_DE(n_trails)

    if conf.FSBO.pretrain or conf.FSBO.evaluate:
        study_FSBO(conf.FSBO, n_keys=n_keys, n_trails=n_trails)

    plot_rank_graph(n_keys=n_keys, n_trials=n_trails)
    return

if __name__ == '__main__':
    mp.freeze_support()
    main()


"""
# Average plotting code. (Backup)
    # avg_gp_performance = np.mean(gp_performance, axis=0)
    # plt.figure(1)
    # plt.plot(avg_gp_performance)
    # plt.legend(["GP Average"])
    # plt.savefig("Average_GP.png")
    # plt.show()
    
    # avg_rs_performance = np.mean(rs_performance, axis=0)
    # plt.figure(2)
    # plt.plot(avg_rs_performance)
    # plt.plot(avg_gp_performance)
    # plt.legend(["RS Average", "GP Average"])
    # plt.savefig("Average_RS_GP.png")
    # plt.show()

    # ####
    
    plt.figure(4)
    plt.plot(np.mean(rs_performance, axis=0))
    plt.plot(np.mean(gp_performance, axis=0))
    plt.plot(np.mean(de_performance, axis=0))
    plt.legend(["RS Average", "GP Average", "DE Average [32,32] ep=1000 lr=0.02 (Rand. Start)"])
    plt.savefig("Average_RS_GP_DE.png")
    # plt.show()
    
        # Loading previous outputs
    gp_keys = load_object("gp_keys")
    gp_performance = load_object("gp_performance")
    rs_performance = load_object("rs_performance")
    de1 = load_object("de_performance_32x32_E1000_l0_01")
    de2 = load_object("de_performance_32x32_E1000_l0_02_random_start")
    ####
    plt.figure(40)
    plt.plot(np.mean(rs_performance, axis=0))
    plt.plot(np.mean(gp_performance, axis=0))
    plt.plot(np.mean(de1, axis=0))
    plt.plot(np.mean(de2, axis=0))
    plt.legend(["RS Average", "GP Average", "DE Average [32,32] ep=1000 lr=0.01", "DE Average [32,32] ep=1000 lr=0.02 (Rand.)"])
    plt.show()
    
     mse = load_object("mse")
    mse_random_start = load_object("mse_32x32_E1000_l0_02_random_start")
    mse_2000 = load_object("mse_32x32_E2000_l0_01")
    plt.plot(np.nanmean(np.array(mse), axis=0))
    plt.plot(np.nanmean(np.array(mse_random_start), axis=0))
    plt.plot(np.nanmean(np.array(mse_2000), axis=0))
    plt.legend(["mse", "mse_random_start", "mse_2000"])
    plt.show()


    variance = load_object("variance")
    variance_random_start = load_object("variance_32x32_E1000_l0_02_random_start")
    variance_2000 = load_object("variance_32x32_E2000_l0_01")
    plt.plot(np.nanmean(np.array(variance), axis=0))
    plt.plot(np.nanmean(np.array(variance_random_start), axis=0))
    plt.plot(np.nanmean(np.array(variance_2000), axis=0))
    plt.legend(["variance", "variance_random_start", "variance_2000"])
    plt.show()
"""