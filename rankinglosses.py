import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# Local imports
from HPO_B.hpob_handler import HPOBHandler

# Defining our ranking model as a DNN.
# Keeping the model simple for now.
class Scorer(nn.Module):
    # Output dimension by default is 1 as we need a real valued score.
    def __init__(self, input_dim=1):
        super(Scorer, self).__init__()
        self.input_dim = input_dim
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # The last layer must not be passed through relu

        # however we pass it through torch.tanh to keep the output in a resonable range
        # I had answered regarding this on stackoverflow
        # https://ai.stackexchange.com/questions/31595/are-the-q-values-of-dqn-bounded-at-a-single-timestep/31648#31648
        # The range of our modelled function should be large enough to correctly map all input domain values.
        return 2 * torch.tanh(0.01 * x)


#######################################################################################
# LIST WISE LOSS FUNCTION MLE
#######################################################################################
"""
Understanding loss_list_wise_mle:
Calculates and returns the probability of the required permutation given 
a list of actual and predicted scores of a set of objects.
E.g: Let the list of scores predicted = [20, 3, 56, 9].
     Let the actual scores = [2.4, 1.3, 3.1, 4.9].
     Given the list of predicted scores, the probability of selecting the given objects
     based on actual scores (required permutation) is to be increased.
     Here this probability is the following:
     Object 4 has the best actual score. It need to be ranked first. Hence the expected ranked list:
        object4 --> (Has actual score 4.9)
        object3 --> (Has actual score 3.1)
        object1 --> (Has actual score 2.4)
        object2 --> (Has actual score 1.3)
     The probability of selecting this list using the predicted score is (without replacement):
     (9/88) * (56/79) * (20/23) * (3/3)
     Applying log to this gives
     log(9/88) + log(56/79) + log(20/23) + log(3/3)
     Since the scores can be negative, we use a strictly increasing positive function like exp
     as a helper and convert all the predictions to exp(predictions)
@Interface:
    predicted_scores: Score list predicted by a score function. Shape expected = [..., listsize]
    actual_scores: Score list known to us. Shape expected = [..., listsize]
@Return:
    Negative log likelihood of required permutation being selected using the predicted score list.
"""


def loss_list_wise_mle(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    # Making sure that the tensor sizes are the same.
    if predicted_scores.size() != actual_scores.size():
        raise Exception("loss_list_wise_mle: predicted_scores.size() != actual_scores.size()")

    # Numerical stability:
    #     Subtracting a constant from predicted_scores has no impact on result.
    #     This is because we use exp as our strictly positive increasing function
    #     to make sure that the calculated probabilities are positive.
    max_predicted, _ = torch.max(predicted_scores, dim=-1, keepdim=True)
    predicted_scores = predicted_scores - max_predicted
    predicted_scores = torch.exp(predicted_scores)

    # Making sure that the given actual score list is unchanged.
    # We need to change the predicted score list for the autograd functionality to work.
    actual_scores = actual_scores.clone()

    # Loop through the object selection process (without replacement) and sum results
    log_probability_sum = 0
    n = actual_scores.shape[-1]  # Number of elements in a list / list size.
    for _ in range(n):
        # Calculate the top 1 log probability and remove the items from both
        # actual and predicted list (Hence the cloning was important)
        log_prob = top_1_log_prob(predicted_scores, actual_scores)
        predicted_scores, actual_scores = remove_top_1_element(predicted_scores, actual_scores)
        log_probability_sum += log_prob

    return -1 * log_probability_sum


def top_1_log_prob(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    # Viewing everything as a 2D tensor.
    y_actual = actual_scores.view(-1, actual_scores.shape[-1])
    y_predicted = predicted_scores.view(-1, predicted_scores.shape[-1])

    # Calculate the max indices and values in the 2D view.
    max_indices = torch.argmax(y_actual, dim=-1)
    max_values = y_predicted[np.arange(max_indices.shape[0]), max_indices]

    # View the max values in (n-1)D view.
    max_values = max_values.view(predicted_scores.shape[:-1])
    prob = max_values / torch.sum(predicted_scores, dim=-1)

    return torch.log(prob)


def remove_top_1_element(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    # Calculate the max indices in the 2D view.
    y_actual = actual_scores.view(-1, actual_scores.shape[-1])
    max_indices = torch.argmax(y_actual, dim=-1)

    # Get the nD max indices however viewing it as 1D. (Note: torch.argmax reduces the dimension by 1)
    max_indices = np.arange(max_indices.shape[0]) * predicted_scores.shape[-1] + max_indices.numpy()
    max_indices = torch.from_numpy(max_indices)

    # View everything with a 1D view.
    y_predicted = predicted_scores.view(-1)
    y_actual = actual_scores.view(-1)

    # Create the mask indices in 1D view and calculate the new shape.
    mask = torch.ones_like(y_predicted, dtype=torch.bool)
    mask[max_indices] = False
    new_view_shape = actual_scores.shape[:-1] + (actual_scores.shape[-1]-1,)

    # Return the remaining elements in the new shaped view.
    return y_predicted[mask].view(new_view_shape), y_actual[mask].view(new_view_shape)


def is_sorted(l, reverse=False):
    # https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not
    if reverse:
        # check if descent sorted
        return all(l[:-1] >= l[1:])
    else:
        # check if ascendent sorted
        return all(l[:-1] <= l[1:])

def test_toy_problem():
    # IDEA:
    #   Can our loss function train the model to sort numbers?
    #   List of numbers to train with = {1 to 100}
    #   List of numbers to validate with = {1 to 100}
    #       Result -> Sorting possible provided the output range of the model not restricted
    #       For now we sample train/val data from the same distribution.
    #   List of numbers to test with = {-1 to -100}
    #       Result -> This is not possible because the model will map this to extreme values as it
    #                 is not distribution dependent.
    #   Check the percentage of the lists in the correct sorted order.
    epochs = 1000
    sc = Scorer(input_dim=1)

    # Creating a uniform distribution between [r1, r2]
    r1 = 0
    r2 = 100
    get_training_data = lambda x: (r1 - r2) * torch.rand(x + (1,)) + r2

    optimizer = torch.optim.Adam(sc.parameters(), lr=0.0001)
    for _ in range(epochs):

        optimizer.zero_grad()

        # Changing the size for it to pass through our scorer
        train_data = get_training_data((10, 10, 3))
        prediction = sc(train_data)

        flatten_from_dim = len(train_data.shape) - 2
        prediction = torch.flatten(prediction, start_dim=flatten_from_dim)
        train_data = torch.flatten(train_data, start_dim=flatten_from_dim)

        loss = loss_list_wise_mle(prediction, -1 * train_data)
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()

        print("Epoch[", _, "] ==>", loss.item())

    print("Now testing")
    sorted_lists = 0
    for i in range(1000):

        # Changing the size for it to pass through our scorer
        train_data = get_training_data((100,))
        prediction = sc(train_data)

        prediction = prediction.flatten()
        train_data = train_data.flatten()

        sorted_indices = torch.argsort(prediction)
        train_data_predicted_rank = train_data[sorted_indices].numpy()

        if is_sorted(train_data_predicted_rank, reverse=True):
            sorted_lists += 1

    print("Sorted percentage : ", sorted_lists * 100 / 1000)


def get_batch_HPBO(meta_data, batch_size, list_size):
    batch = []
    batch_labels = []
    # Sample all tasks and form a high dimensional tensor of size
    #   (tasks, batch_size, list_size, input_dim)
    for data_task_id in meta_data.keys():
        data = meta_data[data_task_id]
        X = data["X"]
        y = data["y"]
        idx = np.random.choice(X.shape[0], size=(batch_size, list_size), replace=True)
        batch += [torch.from_numpy(X[idx])]
        batch_labels += [torch.from_numpy(y[idx])]

    return torch.stack(batch), torch.stack(batch_labels)

def convert_meta_data_to_np_dictionary(meta_data):
    temp_meta_data = {}
    for k in meta_data.keys():
        X = np.array(meta_data[k]["X"], dtype=np.float32)
        y = np.array(meta_data[k]["y"], dtype=np.float32)
        temp_meta_data[k] = {"X": X, "y": y}

    return temp_meta_data

if __name__ == '__main__':
    # Unit testing our loss functions
    # test_toy_problem()

    # Pretrain Ranking loss surrogate with a single search space
    search_space_id = '5859'
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")
    meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
    meta_val_data = hpob_hdlr.meta_validation_data[search_space_id]

    dataset_key = list(meta_train_data.keys())[0]
    input_dim = np.array(meta_train_data[dataset_key]["X"]).shape[1]
    print("Input dim of", search_space_id, "=", input_dim)

    meta_train_data = convert_meta_data_to_np_dictionary(meta_train_data)
    meta_val_data = convert_meta_data_to_np_dictionary(meta_val_data)

    epochs = 500
    batch_size = 100
    list_size = 100
    sc = Scorer(input_dim=input_dim)

    optimizer = torch.optim.Adam(sc.parameters(), lr=0.01)
    loss_list = []
    val_loss_list = []
    for _ in range(epochs):
        sc.train()
        optimizer.zero_grad()

        train_data, train_labels = get_batch_HPBO(meta_train_data, batch_size, list_size)
        prediction = sc(train_data)

        flatten_from_dim = len(prediction.shape) - 2
        prediction = torch.flatten(prediction, start_dim=flatten_from_dim)
        train_labels = torch.flatten(train_labels, start_dim=flatten_from_dim)

        loss = loss_list_wise_mle(prediction, train_labels)
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sc.eval()
            val_data, val_labels = get_batch_HPBO(meta_val_data, batch_size, list_size)
            pred_val = sc(val_data)

            flatten_from_dim = len(pred_val.shape) - 2
            pred_val = torch.flatten(pred_val, start_dim=flatten_from_dim)
            val_labels = torch.flatten(val_labels, start_dim=flatten_from_dim)

            val_loss = loss_list_wise_mle(pred_val, val_labels)
            val_loss = torch.mean(val_loss)

        print("Epoch[", _, "] ==> Loss =", loss.item()/list_size, "; Val_loss =", val_loss.item()/list_size)
        loss_list += [loss.item()/list_size]
        val_loss_list += [val_loss.item()/list_size]

    plt.figure(np.random.randint(999999999))
    plt.plot(np.array(loss_list, dtype=np.float32))
    plt.plot(np.array(val_loss_list, dtype=np.float32))
    legend = ["Loss",
              "Validation Loss"
              ]
    plt.legend(legend)
    plt.show()
