# %% [markdown]
# # Test Deep Learning

# %%
%matplotlib inline 
!module load nvhpc 

# %%
import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn

# %%
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pyarrow.feather as feather
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

# %%
USE_ALL_SAMPLES = False

# %%
rna_df = feather.read_feather("shared_data/rna_df.feather")
expression_df = feather.read_feather("shared_data/expression_df.feather").drop(
    ["key"], axis=1
)
methylation_data_df = feather.read_feather(
    "shared_data/methylation_data_df.feather"
).drop(["PMR_INDEX"], axis=1)
mutation_data_df = feather.read_feather("shared_data/mutation_data_df.feather")
meth_data = torch.tensor(
    methylation_data_df.transpose().astype(np.float64).values
).float()
expression_data = torch.tensor(
    expression_df.values.transpose().astype(np.float64)
).float()

# %%
methylation_data_df.shape

# %%
expression_data_train, expression_data_test, meth_data_train, meth_data_test = train_test_split(expression_data,meth_data,  test_size=0.33, random_state=42)

# %%
for l in [expression_data_train, expression_data_test, meth_data_train, meth_data_test]:
    print(l.shape)

# %%
class CancerDataset(Dataset):
    def __init__(self, expression_data, meth_data):
        super(CancerDataset, self).__init__()
        self.dataset = list(zip(expression_data, meth_data))

    def prepare_item(self, item):
        return item[0], item[1]

    def __getitem__(self, index):
        return self.prepare_item(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

# %%
train_dataset = CancerDataset(expression_data_train, meth_data_train, )
val_dataset = CancerDataset(expression_data_test, meth_data_test, )

trainloader = DataLoader(
    train_dataset, batch_size=3
)
valloader = DataLoader(
    val_dataset, batch_size=3
)

# %%
import matplotlib.pyplot as plt

# %%
expression_data_train.size()

# %%
meth_data_train.size()

# %%
torch.cuda.is_available()

# %%
torch.cuda.device_count()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

# %% [markdown]
# ## 2. Define Model

# %%
model = nn.Sequential(
    nn.Linear(60660, 100),
    nn.ReLU(),
    nn.Linear(100, 98)
).to(device)

# %%
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
# objective = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# %% [markdown]
# ## 3. Train Model

# %%
val_every = 5
train_losses = []
val_losses = []
epochs = 20

# %%
def evaluate(model, mse, kl, val_loader):

    val_losses = []

    model.eval()
    with torch.no_grad():
        for expression, methylation in val_loader:

            expression, methylation = expression.to(device), methylation.to(device)
            pred = model(expression)

            mse = mse_loss(pred, methylation)
            kl = kl_loss(model)
            cost = mse + kl_weight*kl

            val_losses.append(cost.item())

    model.train()

    return torch.mean(torch.Tensor(val_losses))

# %%
loop = tqdm(total=len(trainloader) * epochs)
last_val_loss = None
for epoch in range(epochs):

    for i, (expression, methylation) in enumerate(trainloader):
        expression, methylation = expression.to(device), methylation.to(device)
        optimizer.zero_grad()

        pred = model(expression)
        mse = mse_loss(pred, methylation)
        kl = kl_loss(model)
        cost = mse + kl_weight*kl
        train_losses.append(cost.item())

        cost.backward()
        optimizer.step()

    
        if (i+1) % val_every == 0:
            val_loss = evaluate(model, mse, kl, valloader)
            last_val_loss = val_loss
            val_losses.append((len(train_losses), val_loss))
        loop.set_description('train - Cost: %2.2f MSE : %2.2f, KL : %2.2f' % (cost.item(), mse.item(), kl.item()) + '\nval loss:{:.4f}'.format(last_val_loss.item() ) if last_val_loss is not None else '')
        loop.update(1)
    


# %%
plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
plt.title("Linear Model Gene Expression to Methylaton")

plt.xlabel('Train Time')
plt.ylabel('Loss')
plt.legend()
plt.show()

x, val_loss = zip(*val_losses)
plt.plot(x, val_loss, label='Val Loss')
plt.title("Linear Model Gene Expression to Methylaton")
x_text = (len(x)) * 0.05
y_text = max(val_loss) * 0.9
plt.text(
    x_text,
    y_text,
    f"MSE: {last_val_loss.item():.2f}",
    fontsize=12,
)
plt.xlabel('Train Time')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
model_scripted = torch.jit.script(model)  # Export to TorchScript
if USE_ALL_SAMPLES:
    model_scripted.save("linear_model_script.all_samples.pt")
else:
    model_scripted.save("linear_model_script.pt")

# %%
linear_model = torch.jit.load("linear_model_script.pt").to(device)

# %%
linear_model.state_dict()

# %%
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score

import random
ones = [i for i in range(100)]
ones_2 = [i for i in range(100)]
random.shuffle(ones_2)
pearsonr(ones, ones_2)

# %%
actual = []
predicted = []
linear_model.eval()
with torch.no_grad():
    for expression, methylation in valloader:

        expression, methylation = expression.to(device), methylation.to(device)
        pred = linear_model(expression)
        actual.extend(methylation.cpu().numpy().flatten().tolist())
        predicted.extend(pred.cpu().numpy().flatten().tolist())


pearson_corr, _ = pearsonr(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

# Plot the Results
plt.scatter(actual, predicted)
plt.xlabel("True Methylation Levels")
plt.ylabel("Predicted Methylation Levels")
plt.title("Neual Network Gene Expression Regressor Predictions vs True Values")
plt.plot(
    [min(actual), max(predicted)],
    [min(actual), max(predicted)],
    "k--",
    lw=4,
)

x_text = min(actual) + (max(actual) - min(actual)) * 0.9
y_text = min(predicted) + (max(predicted) - min(predicted)) * 0.05

plt.text(
    x_text,
    y_text,
    f"MSE: {mse:.2f}\nR²: {r2:.2f}\nPearson Correlation: {pearson_corr:.2f}",
    fontsize=12,
)

plt.show()

# %%



