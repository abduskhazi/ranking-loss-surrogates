import torch
import torch.nn as nn
import numpy as np
import gpytorch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

class DKT(nn.Module):
    def __init__(self, backbone, batch_size):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone  # Sent as NN --> extract the input in Latent space

        train_x = torch.ones(batch_size, backbone.output_dim).to(device)
        train_y = torch.ones(batch_size).to(device)
        self.get_model_likelihood_mll(train_x, train_y)  # Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel='rbf')

        self.model      = model.to(device)
        self.likelihood = likelihood.to(device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(device)
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def get_training_batch(self, data, b_n, batch_size):
        X = np.array(data["X"], dtype=np.float32)
        y = np.array(data["y"], dtype=np.float32)

        batch = []
        batch_labels = []
        for i in range(b_n):
            idx = np.random.choice(np.arange(len(X)), batch_size)
            batch += [torch.from_numpy(X[idx])]
            batch_labels += [torch.from_numpy(y[idx]).flatten()]

        return torch.stack(batch), torch.stack(batch_labels)

    def train_loop(self, epoch, optimizer, data, l, u, b_n, batch_size):
        batch, batch_labels = self.get_training_batch(data, b_n, batch_size)
        batch, batch_labels = batch.to(device), batch_labels.to(device)
        # Scale the lables according to the l and u for scale invariance.
        batch_labels = (batch_labels - l) / (u - l)
        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)

            self.model.set_train_data(inputs=z, targets=labels)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    def test_loop(self, n_support, optimizer=None): # no optimizer needed for GP
        inputs, targets = get_batch(test_people)

        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind   = [i for i in range(19) if i not in support_ind]

        x_all = inputs.to(device)
        y_all = targets.to(device)

        x_support = inputs[:,support_ind,:,:,:].to(device)
        y_support = targets[:,support_ind].to(device)
        x_query   = inputs[:,query_ind,:,:,:]
        y_query   = targets[:,query_ind].to(device)

        # choose a random test person
        n = np.random.randint(0, len(test_people)-1)

        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        ## RBF kernel
        if(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

