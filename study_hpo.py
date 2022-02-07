from HPO_B.hpob_handler import HPOBHandler
from HPO_B.methods.random_search import RandomSearch
from HPO_B.methods.botorch import GaussianProcess
import matplotlib.pyplot as plt
import multiprocessing as mp
from DE import DE_search
import numpy as np
import scipy
import gpytorch
import time
import pickle

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
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = DE_search(input_dim=input_dim)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res
    return performance

def main():
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    n_trials = 100

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
    avg_gp_performance = np.mean(gp_performance, axis=0)
    plt.figure(1)
    plt.plot(avg_gp_performance)
    plt.legend(["GP Average"])
    plt.savefig("Average_GP.png")
    # plt.show()
    # Store results
    store_object(gp_keys, "gp_keys")
    store_object(gp_performance, "gp_performance")
    # ####

    # Loading previous outputs
    gp_keys = load_object("gp_keys")
    gp_performance = load_object("gp_performance")
    ####
    method = RandomSearch()
    rs_performance = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=gp_keys)
    rs_performance = [performance_list for _, performance_list in rs_performance]
    rs_performance = np.array(rs_performance, dtype=np.float32)
    avg_rs_performance = np.mean(rs_performance, axis=0)
    plt.figure(2)
    plt.plot(avg_rs_performance)
    plt.plot(avg_gp_performance)
    plt.legend(["RS Average", "GP Average"])
    plt.savefig("Average_RS_GP.png")
    # plt.show()
    # Store results
    store_object(rs_performance, "rs_performance")
    # ####

    # Creating a rank graph from the given data
    performance = np.stack((rs_performance, gp_performance), axis=-1)
    # Since rank data ranks in the increasing order, we need to multiply by -1
    rg = scipy.stats.rankdata(-1 * performance, axis=-1)
    rank_rs = np.mean(rg[:, :, 0], axis=0)
    rank_gp = np.mean(rg[:, :, 1], axis=0)
    plt.figure(3)
    plt.plot(rank_rs)
    plt.plot(rank_gp)
    plt.legend(["RS Rank", "GP Rank"])
    plt.savefig("Rank_RS_GP.png")
    # plt.show()

    # Loading previous outputs
    gp_keys = load_object("gp_keys")
    gp_performance = load_object("gp_performance")
    rs_performance = load_object("rs_performance")
    ####
    de_performance = evaluate_DE(hpob_hdlr, keys_to_evaluate=gp_keys)
    de_performance = [performance_list for _, performance_list in de_performance]
    de_performance = np.array(de_performance, dtype=np.float32)
    plt.figure(4)
    plt.plot(np.mean(rs_performance, axis=0))
    plt.plot(np.mean(gp_performance, axis=0))
    plt.plot(np.mean(de_performance, axis=0))
    plt.legend(["RS Average", "GP Average", "DE Average [32,32] ep=500 lr=0.02"])
    plt.savefig("Average_RS_GP_DE.png")
    # plt.show()
    # Store results
    store_object(de_performance, "de_performance_32_32_500_0_02")
    # ####

    # Creating a rank graph for all 3 methods
    performance = np.stack((rs_performance, gp_performance, de_performance), axis=-1)
    # Since rank data ranks in the increasing order, we need to multiply by -1
    rg = scipy.stats.rankdata(-1 * performance, axis=-1)
    rank_rs = np.mean(rg[:, :, 0], axis=0)
    rank_gp = np.mean(rg[:, :, 1], axis=0)
    rank_de = np.mean(rg[:, :, 2], axis=0)
    plt.figure(5)
    plt.plot(rank_rs)
    plt.plot(rank_gp)
    plt.plot(rank_de)
    plt.legend(["RS Rank", "GP Rank", "DE Rank [32,32] ep=500 lr=0.02"])
    plt.savefig("Rank_RS_GP_DE.png")
    # plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()
