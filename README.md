# Binary Classification with Neural Networks on the Census Income Dataset

This project demonstrates how to develop and train a **Neural Network** using **PyTorch** for binary income classification on the **Census Income (Adult)** dataset.  
The goal is to predict whether a person earns **more than $50,000 per year** based on demographic and work-related attributes.



##  AIM
To build and evaluate a **Neural Network Classification Model** capable of predicting income levels from the Census Income dataset.



##  THEORY

The **Census Income dataset**  includes various demographic and occupational features.  
It is a **binary classification problem**, where the model learns to predict one of two classes:

- `<=50K`
- `>50K`

A neural network is effective for this task as it can capture nonlinear relationships among mixed (categorical and numerical) variables.



## DESIGN STEPS
### STEP 1: 

Load the Dataset.

### STEP 2: 
Preprocess Data.

### STEP 3: 

Build Neural Network.

### STEP 4: 

Train the Model.

### STEP 5: 

Evaluate the Model.

### STEP 6: 

Display the test training loss plot, accuracy, and predict new data.

## PROGRAM 


### Name: MUKESH KUMAR S

### Register Number: 212223230099

```
# Imports & Setup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
# Load Data
df = pd.read_csv('income.csv')
df = shuffle(df, random_state=101).reset_index(drop=True)
print(len(df))
df.head()
df['label'].value_counts()
# 1. Separate categorical, continuous, label
cat_cols  = ['sex','education','marital-status','workclass','occupation']
cont_cols = ['age','hours-per-week']
y_col     = ['label']

print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')
# 2. Convert to category dtype
for col in cat_cols:
    df[col] = df[col].astype('category')
# 3. Embedding sizes
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs
# 4. Create an array of categorical values
cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)

cats[:5]

# 5. Convert categorical values to tensor (dtype int64)
cats = torch.from_numpy(cats).clone().detach().to(torch.int64)
cats.dtype
# 6. Stack continuous columns into a NumPy array
conts = np.stack([df[col].values for col in cont_cols], axis=1)

conts[:5]
# 7. Convert continuous values to tensor (dtype float32)
conts = torch.tensor(conts, dtype=torch.float32)

conts.dtype
y = torch.tensor(df['label'].values, dtype=torch.int64).flatten()
# 8. Train/test split
b = 30000
t = 5000
cat_train, cat_test = cats[:b-t], cats[b-t:b]
con_train, con_test = conts[:b-t], conts[b-t:b]
y_train,   y_test   = y[:b-t],    y[b-t:b]
# 9. Define model
class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()

        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Assign a variable to hold a list of layers
        layerlist = []

        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont

        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))

        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)

        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        # Set up model layers
        x = self.layers(x)
        return x
# 10. Set the random seed
torch.manual_seed(33)
#11. Create a TabularModel instance
model = TabularModel(emb_szs=emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
print(model)
# 12. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#Train the model
import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
# 13. Plot Loss
import matplotlib.pyplot as plt

plt.plot([l.item() for l in losses])
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")
plt.show()
#14. Evaluate the test set
with torch.no_grad():
    # forward pass on the test set
    y_val = model(cat_test, con_test)

    # calculate loss against y_test
    loss = criterion(y_val, y_test)

print(f'CE Loss: {loss:.8f}')
#15. Calculate the overall percent accuracy
# Get predicted class (0 or 1) from model output
preds = torch.argmax(y_val, dim=1)

# Compare predictions to true labels
correct = (preds == y_test).sum().item()
total = y_test.size(0)
accuracy = correct / total * 100

print(f"{correct} out of {total} = {accuracy:.2f}% correct")
# BONUS: Predict New Data

def predict_new(age, sex, education, marital_status, workclass, occupation, hours_per_week):
    model.eval()
    cat_dicts = {col: {cat: i for i, cat in enumerate(df[col].cat.categories)} for col in cat_cols}
    cat_input = [
        cat_dicts['sex'][sex],
        cat_dicts['education'][education],
        cat_dicts['marital-status'][marital_status],
        cat_dicts['workclass'][workclass],
        cat_dicts['occupation'][occupation]
    ]
    cont_input = [age, hours_per_week]
    cat_t = torch.tensor([cat_input], dtype=torch.int64)
    cont_t = torch.tensor([cont_input], dtype=torch.float32)
    out = model(cat_t, cont_t)
    pred = torch.argmax(out, dim=1).item()
    return pred

# Example usage
prediction = predict_new(22, "Male", "12th", "Married", "Private", "Sales", 40)
print(f"The predicted label is {prediction}")

```

### Dataset Information

<img width="1073" height="212" alt="image" src="https://github.com/user-attachments/assets/0e027b04-db6e-4d9a-be79-19703e8f6c19" />

### OUTPUT

## Training Loss plot

<img width="703" height="541" alt="image" src="https://github.com/user-attachments/assets/1e1b6d31-02a5-4f64-a718-7719d479ca4e" />

<img width="211" height="72" alt="image" src="https://github.com/user-attachments/assets/90a5370c-c2b6-450a-a373-3e6e3e422254" />

## Overall Percent Accuracy

<img width="363" height="45" alt="image" src="https://github.com/user-attachments/assets/bac2e983-9788-4aa9-a2c1-f3355da82069" />

## Predict New Data

<img width="262" height="41" alt="image" src="https://github.com/user-attachments/assets/6b73948d-ff62-49b0-ab70-6c7eb33e6691" />

## RESULT

A neural network classification model was successfully developed using PyTorch for the Census Income dataset, achieving strong predictive accuracy in classifying income categories.




