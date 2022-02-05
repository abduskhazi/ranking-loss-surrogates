from HPO_B.hpob_handler import HPOBHandler
from HPO_B.methods.random_search import RandomSearch
from HPO_B.methods.botorch import GaussianProcess
import matplotlib.pyplot as plt
import multiprocessing as mp
from DE import DE_search

def main():
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")

    search_space_id = hpob_hdlr.get_search_spaces()[0]
    dataset_id = hpob_hdlr.get_datasets(search_space_id)[2]
    seed = "test1"
    n_trials = 20

    method = RandomSearch()
    acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id,
                                            dataset_id = dataset_id,
                                            seed = seed,
                                            n_trials = n_trials)
    plt.plot(acc)

    method = GaussianProcess(acq_name="EI")
    acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id,
                                            dataset_id = dataset_id,
                                            seed = seed,
                                            n_trials = n_trials)
    plt.plot(acc)
    # plt.legend(["Random Search", "Gaussian Processes"])
    # plt.show()

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