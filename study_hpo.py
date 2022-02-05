from HPO_B.hpob_handler import HPOBHandler
from HPO_B.methods.random_search import RandomSearch
from HPO_B.methods.botorch import GaussianProcess
import matplotlib.pyplot as plt
import multiprocessing as mp
from DE import DE_search
import numpy as np
import scipy

# created as a stub for parallel evaluations.
def evaluation_worker(args):
    hpob_hdlr, method, search_space, dataset, seed, n_trials = args
    res = hpob_hdlr.evaluate(method,
                              search_space_id=search_space,
                              dataset_id=dataset,
                              seed=seed,
                              n_trials=n_trials)
    print("Completed 1 evaluation instance")
    return res

def evaluate_all_combinations(hpob_hdlr, method, n_trials, parallelize=False):
    # A total of 430 combinations are present in this if all seeds are used.
    print("Evaluating all elements of set {search_space} X {dataset} X {seed} for", method)
    seed_list = ["test0", "test1", "test2", "test3", "test4"]
    evaluation_list = []
    for search_space in hpob_hdlr.get_search_spaces():
        for dataset in hpob_hdlr.get_datasets(search_space):
            # search_space: '5965' X dataset '9946' is having a problem for GPs, hence excluding this combination
            # from our evaluation list.
            if not (search_space == '5965' and dataset == '9946'):
                for seed in ["test2"]:  # seed_list: # use this for running on all possible seeds
                    evaluation_list += [(hpob_hdlr, method, search_space, dataset, seed, n_trials)]

    if parallelize:
        with mp.Pool(mp.cpu_count()) as p:
           performance = p.map(evaluation_worker, evaluation_list)
           return performance
    else:
        performance = []
        run_i = 0
        for eval in evaluation_list:
            result = evaluation_worker(eval)
            performance.append(result)
            run_i = run_i + 1
            print("Completed Running", run_i, end="\n")
        return performance

def main():
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    n_trials = 100

    method = RandomSearch()
    rs_performance = evaluate_all_combinations(hpob_hdlr, method, n_trials, parallelize=False)
    rs_performance = np.array(rs_performance, dtype=np.float32)
    avg_rs_performance = np.mean(rs_performance, axis=0)
    plt.plot(avg_rs_performance)
    plt.legend(["Random Search Average"])
    plt.savefig("Average_RS.png")
    # plt.show()

    method = GaussianProcess(acq_name="EI")
    gp_performance = evaluate_all_combinations(hpob_hdlr, method, n_trials, parallelize=True)
    gp_performance = np.array(gp_performance, dtype=np.float32)
    avg_gp_performance = np.mean(gp_performance, axis=0)
    plt.figure(2)
    plt.plot(avg_rs_performance)
    plt.plot(avg_gp_performance)
    plt.legend(["RS Average", "GP Average"])
    plt.savefig("Average_RS_GP.png")
    # plt.show()

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


    search_space_id = hpob_hdlr.get_search_spaces()[0]
    dataset_id = hpob_hdlr.get_datasets(search_space_id)[2]
    seed = "test1"
    n_trials = 20
    method = DE_search(input_dim=3)
    acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id,
                                            dataset_id = dataset_id,
                                            seed = seed,
                                            n_trials = n_trials)
    plt.figure(4)
    plt.plot(acc)

    legend = ["Random Search", "Gaussian Processes", "Deep Ensembles"]
    plt.legend(legend)
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()
