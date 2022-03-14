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
        self.batch_size = batch_size

        # backbone.output_dim is the latent dimension for the complete kernel
        train_x = torch.ones(batch_size, backbone.output_dim).to(device)
        train_y = torch.ones(batch_size).to(device)
        self.get_model_likelihood_mll(train_x, train_y)  # Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel='matern')

        self.model      = model.to(device)
        self.likelihood = likelihood.to(device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(device)
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

        X = data["X"]
        y = data["y"]

        if batch_size > X.shape[0]:
            batch_size = X.shape[0]

        batch = []
        batch_labels = []
        for i in range(b_n):
            idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            batch += [torch.from_numpy(X[idx])]
            batch_labels += [torch.from_numpy(y[idx]).flatten()]

        return torch.stack(batch), torch.stack(batch_labels)

    def get_validation_batch(self, val_data):
        # This is different from the training batch because we require
        # support and query sets for testing the working.
        X = val_data["X"]
        y = val_data["y"]

        # 100 support points and 200 (or remaining) query points distinct sets
        n_support_points = 50
        n_query_points = 200 if X.shape[0] > 200 + n_support_points else X.shape[0] - n_support_points

        idx_support = np.random.choice(X.shape[0], size=n_support_points, replace=False)
        query_choice = np.delete(np.arange(X.shape[0]), idx_support)
        idx_query = np.random.choice(query_choice, size=n_query_points, replace=False)

        X_support = torch.from_numpy(X[idx_support])
        y_support = torch.from_numpy(y[idx_support].flatten())
        X_query = torch.from_numpy(X[idx_query])
        y_query = torch.from_numpy(y[idx_query].flatten())

        return X_support, y_support, X_query, y_query

    def train_loop(self, epoch, optimizer, data, meta_val_data, l, u, b_n, batch_size, scaling=True):
        # Required in case of reuse
        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()

        batch, batch_labels = self.get_training_batch(data, b_n, batch_size)
        batch, batch_labels = batch.to(device), batch_labels.to(device)
        if scaling:
            # Scale the labels according to the l and u for scale invariance.
            batch_labels = (batch_labels - l) / (u - l)

        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)

            self.model.set_train_data(inputs=z, targets=labels, strict=False)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

        val_loss_list = []
        for k in meta_val_data.keys():
            val_data = self.get_validation_batch(meta_val_data[k])
            val_loss_list += [self.get_val_loss(val_data)]
        val_loss = sum(val_loss_list)/len(val_loss_list)

        if (epoch%40==0):
            print('[%d] - Loss: %.3f  Val-Loss: %.3f MSE: %.3f noise: %.3f' % (
                epoch, loss.item(), val_loss, mse.item(),
                self.model.likelihood.noise.item()
            ))

        return loss.item(), val_loss

    def get_val_loss(self, val_data):  # Check testing code of DKT Regression.
        X_support, y_support, X_query, y_query = val_data

        z_support = self.feature_extractor(X_support).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(X_query).detach()
            prediction = self.likelihood(self.model(z_query))
            val_loss = -self.mll(prediction, y_query)

        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()

        return val_loss.item()

    def fine_tune_loop(self, epoch, optimizer, X, y):
        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()


        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        z = self.feature_extractor(X)
        self.model.set_train_data(inputs=z, targets=y, strict=False)
        predictions = self.model(z)
        loss = -self.mll(predictions, self.model.train_targets)

        loss.backward()
        optimizer.step()

        mse = self.mse(predictions.mean, y)
        if epoch % 40 == 0:
            print('[%d] - Loss: %.3f MSE: %.3f noise: %.3f' % (
                epoch, loss.item(), mse.item(),
                self.model.likelihood.noise.item()
            ))
        return loss.item()

    def predict(self, x_support, y_support, x_query):
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        x_support = x_support.to(device)
        y_support = y_support.to(device)
        x_query   = x_query.to(device)

        z_support = self.feature_extractor(x_support).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))

        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()

        return pred

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
        ## Another possibility of a kernel - self.covar_module =
        ## gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=config["nu"], ard_num_dims=dims if config["a    rd"] else None))
        if(kernel == "matern" or kernel == "MATERN"):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        elif(kernel=='rbf' or kernel=='RBF'):
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

