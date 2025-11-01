import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=["order0", "order1", "order2"])
    y_train = train_df["order0"]

    X_val = val_df.drop(columns=["order0", "order1", "order2"])
    y_val = val_df["order0"]

    return X_train, y_train, X_val, y_val, test_df

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        

def init_model(lr):
    model = MLP()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    print(f"Cuda = {torch.cuda.is_available()}")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, criterion, optimizer


def evaluate(model, X, y):
    ### YOUR CODE HERE
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    ### YOUR CODE HERE: train the model and validate it every epoch on X_val, y_val

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            outputs = model.forward(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print(f'\t Train: Epoch {epoch}, train Loss: {loss.item()}')
            train_loss += loss.item()

        train_loss /= X_train.size(0)
        if epoch % 1 == 0:
            model.eval()
            val_outputs = model.forward(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
            print(f'Val: Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss.item()}')
    return model


def main(args):
    ### YOUR CODE HERE

    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)

    # Initialize model
    model, criterion, optimizer = init_model(args.lr)

    # Train model
    model = train(model, criterion, optimizer, 
                X_train, y_train, X_val, y_val, 
                args.num_epoches, args.batch_size)

    # Predict on test set
    y_pred = model.forward(X_test)
    df = pd.DataFrame(predictions, columns=['Prediction'])

    df['Id'] = range(1, len(predictions)+1)
    df = df[['Id', 'Prediction']] 

    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='./data/train.csv')
    parser.add_argument('--val_csv', default='./data/val.csv')
    parser.add_argument('--test_csv', default='./data/test.csv')
    parser.add_argument('--out_csv', default='./data/submission.csv')
    parser.add_argument('--lr', default=0)
    parser.add_argument('--batch_size', default=0)
    parser.add_argument('--num_epoches', default=0)

    args = parser.parse_args()
    main(args)
