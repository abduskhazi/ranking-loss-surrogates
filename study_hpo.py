from HPO_B.hpob_handler import HPOBHandler
from HPO_B.methods.random_search import RandomSearch
from HPO_B.methods.botorch import GaussianProcess
import matplotlib.pyplot as plt
import multiprocessing as mp
from DE import DE_search
import numpy as np

def evaluate_all_combinations(hpob_hdlr, method, n_trials):
    print("Evaluating all elements of set {search_space} X {dataset} X {seed} for", method)
    # A total of 430 combinations are present in this if all seeds are used.
    performance = []
    run_i = 0
    seed_list = ["test0", "test1", "test2", "test3", "test4"]
    for search_space in hpob_hdlr.get_search_spaces():
        for dataset in hpob_hdlr.get_datasets(search_space):
            for seed in ["test2"]:  # use seed_list for running on all possible seeds
                result = hpob_hdlr.evaluate(method,
                                            search_space_id=search_space,
                                            dataset_id=dataset,
                                            seed=seed,
                                            n_trials=n_trials)
                performance.append(result)
                run_i = run_i + 1
                print("Completed Running", run_i, end="\n")
    print()
    return performance

def main():
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    n_trials = 100

    method = RandomSearch()
    rs_performance = evaluate_all_combinations(hpob_hdlr, method, n_trials)
    avg_rs_performance = np.mean(rs_performance, axis=0)
    plt.plot(avg_rs_performance)
    # plt.legend(["Random Search Average"])
    # plt.show()

    method = GaussianProcess(acq_name="EI")
    gp_performance = evaluate_all_combinations(hpob_hdlr, method, n_trials)
    avg_gp_performance = np.mean(gp_performance, axis=0)
    plt.plot(avg_gp_performance)
    plt.legend(["Random Search Average", "GP Average"])
    plt.show()

    search_space_id = hpob_hdlr.get_search_spaces()[0]
    dataset_id = hpob_hdlr.get_datasets(search_space_id)[2]
    seed = "test1"
    n_trials = 20
    method = DE_search(input_dim=3)
    acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id,
                                            dataset_id = dataset_id,
                                            seed = seed,
                                            n_trials = n_trials)
    plt.plot(acc)

    legend = ["Random Search", "Gaussian Processes", "Deep Ensembles"]
    plt.legend(legend)
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()