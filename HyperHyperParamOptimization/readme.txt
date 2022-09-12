There are 2 stages of hyper-hyperparameter optimization.
    1. Selection of the best configuration for the scorer/ranker in the ensemble.
    2. Selection of the best configuration for the deep set.

Stage 1:
    Model                : Ensemble of NN without Deepsets
    Loss function        : Listwise Weighted + Transfer
    Acquisition function : Expected Improvement (As this gives results comparable to "FSBO")
    Transfer learning is used as it is better than non-transfer learning.
    The following steps are followed in the chornological order:
        a. Training data is used to train the model (epochs=5000, batch_size=100, list_size=100)
        b. Validation data is used to plot the rank graph of the model.
        c. The following configurations are compared with each other
            i.   NN = [input_dim, 32, 32, 1]x10
            ii.  NN = [input_dim, 16, 16, 1]x10
            iii. NN = [input_dim, 48, 48, 1]x10
            iv.  NN = [input_dim, 64, 64, 1]x10
            v.   NN = [input_dim, 32, 32, 32, 1]x10
            vi.  NN = [input_dim, 32, 32, 32, 32, 1]x10

    Name prefix used: paper_hhpo  (Where hhpo stands for hyper-hyperparameter optimization)
    RESULT : The deeper scorer gave the best reults


Stage 2:
    The configuration of the deep set.
