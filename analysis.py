# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import plotnine as p9


# %%
#    """ create piecewise constant data """


def create_clean_data(n_rows, n_knots):
    X = torch.linspace(-1, 1, n_rows)[:, None]
    weights = torch.randn((n_knots, 1))
    biases = torch.rand((n_knots,))
    basis = F.linear(X, weights, biases)
    # basis = F.dropout(basis, dropout_p)
    basis = F.relu(basis)
    input = torch.concat((X, basis), dim=1)

    weights_1 = torch.randn((1, 1 + n_knots))  # (n_out, n_in)
    biases_1 = torch.randn((1,))
    out = F.linear(input, weights_1, biases_1)
    linear_piece = X * weights_1[:, 0] + biases_1
    basis_out = basis * weights_1[:, 1:]

    data = {"x": X[:, 0].squeeze(), "l": linear_piece.squeeze(), "y": out.squeeze()}
    for b in range(basis_out.shape[1]):
        data[f"b_{b:02d}"] = basis_out[:, b]
    data = pd.DataFrame(data)
    return data, weights, biases, weights_1, biases_1


def add_noise(x, noise_std):
    x_noise = x + np.random.normal(0, noise_std, x.shape)
    return x_noise


class DFDataset(Dataset):
    def __init__(self, dataframe, indep_cols, dep_cols):
        self.dataframe = dataframe
        self.indep_cols = indep_cols
        self.dep_cols = dep_cols

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        features = row[[self.indep_cols]].to_numpy().astype(np.float32)
        label = row[[self.dep_cols]].to_numpy().astype(np.float32)
        return features, label

    def __len__(self):
        return len(self.dataframe)


class Net(nn.Module):

    def __init__(self, n_basis, dropout_p):
        super(Net, self).__init__()

        self.basis = nn.Linear(1, n_basis, True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(n_basis + 1, 1, True)

    def forward(self, input):

        b_out = F.relu(self.basis(input))
        b_out = self.dropout(b_out)
        output = self.out(torch.concatenate((b_out, input), dim=1))
        return output


def get_weights(net):
    weights_basis = net.basis.weight.detach().squeeze().cpu()
    bias_basis = net.basis.bias.detach().squeeze().cpu()
    df = pd.DataFrame({"source": "model", "wb": weights_basis, "bb": bias_basis})
    return df


def predict(training_loader, net):
    pred_all = []
    with torch.no_grad():
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Set the net to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            net.eval()

            # Make predictions for this batch
            outputs = net(inputs)
            pred_all.append(torch.concat((inputs, labels, outputs), dim=1))
    pred_data = torch.concat(pred_all)
    return pred_data


def train_epochs(
    net, optimizer, training_loader, validation_loader, noise, n_epochs=500
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("data/runs/spline_trainer_{}".format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.0

    for epoch in range(n_epochs):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss = train_one_epoch(
            net, optimizer, training_loader, writer, epoch_number
        )

        running_vloss = 0.0
        # Set the net to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        net.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to("mps")
                vlabels = vlabels.to("mps")
                voutputs = net(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {} optimal{}".format(avg_loss, avg_vloss, noise**2))

        # Log the running loss averaged per batch
        # for both training and validation

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss, 
            "Optimal": noise**2
            },
            epoch_number + 1,
        )
        model_data = get_weights(net)
        true_model_data = pd.concat((true_data, model_data))

        plot = (
            p9.ggplot(true_model_data, p9.aes(x="bb", y="wb", color="source"))
            + p9.geom_point()
        )
        fig = plot.draw(show=False)
        writer.add_figure("hidden units", fig, epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "data/models/model_{}_{}".format(timestamp, epoch_number)
            torch.save(net.state_dict(), model_path)

        epoch_number += 1
    writer.add_hparams(
        {"model_knots": model_knots, "dropout": dropout},
        {"training_loss": avg_loss, "validation_loss": avg_vloss},
    )
    writer.flush()
    return avg_loss, avg_vloss, best_vloss


def train_one_epoch(net, optimizer, training_loader, tb_writer, epoch_index):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    n_batch = len(training_loader)
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to("mps")
        labels = labels.to("mps")
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % n_batch == n_batch - 1:
            last_loss = running_loss / n_batch  # loss over batches
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


# %%
n_rows = 1000
noise = 0.25
n_knots = 3


# %%
data, weights_basis, biases_basis, weights_out, biases_out = create_clean_data(
    n_rows, n_knots
)
data["y_noisy"] = add_noise(data["y"], noise)
noise_actual = (data["y_noisy"] - data["y"]).std()
full_data = DFDataset(data, "x", "y_noisy")
training_data, test_data = torch.utils.data.random_split(full_data, [0.2, 0.8])
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
# %%
true_data = pd.DataFrame(
    {"source": "gen", "wb": weights_basis.squeeze(), "bb": biases_basis.squeeze()}
)

# %%

plot_data = data.melt(id_vars=["x"])
p9.qplot(data=plot_data, x="x", y="value", colour="variable")


# %%


# %%
model_knots = n_knots *20
dropout = 0.4
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

net = Net(model_knots, dropout)
net.to(mps_device)

print(net)
loss_fn = torch.nn.MSELoss()

lr = 0.01
momentum = 0.9
optimizer = torch.optim.Adam(net.parameters(), lr=lr, )


# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
# PyTorch TensorBoard support

n_epochs = 500
avg_loss, avg_vloss, best_vloss = train_epochs(
    net, optimizer, training_loader, test_loader, noise_actual, n_epochs
)
# %%

model_data = get_weights(net)
true_model_data = pd.concat((true_data, model_data))

p9.ggplot(true_model_data, p9.aes(x="bb", y="wb", color="source")) + p9.geom_point()

# %%
pred_data = predict(training_loader, net)
# %%
pred_df = pd.DataFrame(pred_data.numpy(), columns=["x", "y", "y_pred"]).sort_values("x")
# %%
plot_data = pred_df.melt(id_vars=["x"])
p9.ggplot(plot_data, p9.aes(x="x", y="value", color="variable")) + p9.geom_line()
# %%
# %% [markdown]
# tasks
# 1. ~~add validation data~~

# 1. ~~add fig of hidden units~~
# 1. ~~add fig of plot~~

weights_basis_out = weights_basis * weights_out[:, 1:].T
biases_basis_weights_basis = biases_basis[:, None] / weights_basis


# %%

plt = p9.ggplot(plot_data, p9.aes(x="bbwb", y="wbo")) + p9.geom_point()
