#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import plotnine as p9
#%%
n_rows = 1000
noise = 0.1
n_knots = 3

#    """ create piecewise constant data """

def create_clean_data(n_rows, n_knots):
    X = torch.linspace(-1,1,n_rows)[:,None]
    weights = torch.randn((n_knots,1))
    biases = torch.rand((n_knots,)) 
    basis = F.linear(X,weights, biases)
    #basis = F.dropout(basis, dropout_p)
    basis = F.relu(basis)
    input = torch.concat((X, basis),dim=1)

    weights_1 = torch.randn((1,1 + n_knots)) # (n_out, n_in)
    biases_1 = torch.randn((1,))
    out = F.linear(input,weights_1,biases_1)
    linear_piece = X*weights_1[:,0] + biases_1
    basis_out = basis * weights_1[:,1:]

    data = {"x": X[:,0].squeeze(), "l": linear_piece.squeeze(), "y": out.squeeze()}
    for b in range(basis_out.shape[1]):
        data[f"b_{b:02d}"] =  basis_out[:,b]
    data = pd.DataFrame(data)
    return data, weights, biases, weights_1, biases_1

def add_noise(x, noise_std):
    x_noise = x + np.random.normal(0,noise_std, x.shape) 
    return x_noise


# %%
data, weights_basis, biases_basis, weights_out, biases_out = create_clean_data(n_rows, n_knots)
data["y_noisy"] = add_noise(data["y"],.25)
#%%
plot_data= data.melt(id_vars=["x"])

p9.qplot(data=plot_data, x="x",y="value",colour="variable")


# %%

weights_basis_out = weights_basis * weights_out[:,1:].T
biases_basis_weights_basis = biases_basis[:,None]/weights_basis


# %%
true_data = pd.DataFrame({"source": "gen", "wb": weights_basis.squeeze(), "bb": biases_basis.squeeze()})
plt = p9.ggplot(plot_data, p9.aes(x="bbwb", y="wbo")) + p9.geom_point()


# %%
class Net(nn.Module):

    def __init__(self, n_basis, dropout_p):
        super(Net, self).__init__()
        
        self.basis = nn.Linear(1, n_basis, True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(n_basis + 1, 1, True)

    def forward(self, input):
        
        b_out = F.relu(self.basis(input))
        bout = self.dropout(b_out)
        output = self.out(torch.concatenate((bout, input),dim=1))
        return output

model_knots = n_knots*20
dropout = 0.0
net = Net(model_knots, dropout)
print(net)

# %%

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
from torch.utils.data import DataLoader, Dataset

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

training_data = DFDataset(data, "x", "y_noisy")
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_

def get_weights(net):
    weights_basis = net.basis.weight.detach().squeeze()
    bias_basis = net.basis.bias.detach().squeeze()
    df = pd.DataFrame({"source": "model", "wb": weights_basis, "bb": bias_basis})
    return df

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    n_batch = len(training_loader)
    print(f"n_batch={n_batch}")
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

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
        if i % n_batch == n_batch-1:
            last_loss = running_loss / n_batch # loss over batches
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('data/runs/spline_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 500

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the net to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    net.eval()

    # Disable gradient computation and reduce memory consumption.
    # with torch.no_grad():
    #     for i, vdata in enumerate(validation_loader):
    #         vinputs, vlabels = vdata
    #         voutputs = net(vinputs)
    #         vloss = loss_fn(voutputs, vlabels)
    #         running_vloss += vloss

    avg_vloss = 0 #running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_hparams({"model_knots": model_knots, "dropout": dropout}, {"training_loss": avg_loss, "validation_loss": avg_vloss})
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1
# %%
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
            pred_all.append(torch.concat((inputs, labels, outputs),dim=1))
    pred_data = torch.concat(pred_all)
    return pred_data

# %%
pred_data = predict(training_loader, net)
# %%
pred_df = (
    pd.DataFrame(pred_data.numpy(),columns=["x","y", "y_pred"])
    .sort_values("x")
)
# %%
plot_data = pred_df.melt(id_vars=["x"])
p9.ggplot(plot_data, p9.aes(x="x", y="value", color="variable"))+p9.geom_line()
# %%
model_data = get_weights(net)
true_model_data = pd.concat((true_data,model_data))

p9.ggplot(true_model_data,p9.aes(x= "bb", y="wb", color="source"))+p9.geom_point()
# %% [markdown]
# tasks
# 1. add validation data
# 1. add fig of hidden units
# 1. add fig of plot

