import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.hidden_layers = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # The last layer must not be passed through relu

        return x


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
    predicted_scores: Score list predicted by a score function
    actual_scores: Score list known to us
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
    predicted_scores = predicted_scores - torch.max(predicted_scores)
    predicted_scores = torch.exp(predicted_scores)

    # Making sure that the given actual score list is unchanged.
    # We need to change the predicted score list for the autograd functionality to work.
    actual_scores = actual_scores.clone()

    # Loop through the object selection process (without replacement) and sum results
    log_probability_sum = 0
    n = torch.numel(actual_scores)
    for _ in range(n):
        # Calculate the top 1 log probability and remove the items from both
        # actual and predicted list (Hence the cloning was important)
        log_prob = top_1_log_prob(predicted_scores, actual_scores)
        predicted_scores, actual_scores = remove_top_1_element(predicted_scores, actual_scores)
        log_probability_sum += log_prob

    return -1 * log_probability_sum


def top_1_log_prob(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    max_index = torch.argmax(actual_scores)
    prob = predicted_scores[max_index] / torch.sum(predicted_scores)
    return torch.log(prob)


def remove_top_1_element(predicted_scores: torch.Tensor, actual_scores: torch.Tensor):
    max_index = torch.argmax(actual_scores)
    mask = torch.ones_like(predicted_scores, dtype=torch.bool)
    mask[max_index] = False
    return predicted_scores[mask], actual_scores[mask]
