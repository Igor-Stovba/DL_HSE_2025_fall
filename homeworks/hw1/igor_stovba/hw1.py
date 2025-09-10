import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Добавим dx, dy
    for y in range(1, 30):
        for b in [0, 1, 2]:
            train_df[f"y{y}_b{b}_dx"] = train_df[f"y{y}_b{b}_x"] - train_df[f"y{y-1}_b{b}_x"]
            train_df[f"y{y}_b{b}_dy"] = train_df[f"y{y}_b{b}_y"] - train_df[f"y{y-1}_b{b}_y"]

            val_df[f"y{y}_b{b}_dx"] = val_df[f"y{y}_b{b}_x"] - val_df[f"y{y-1}_b{b}_x"]
            val_df[f"y{y}_b{b}_dy"] = val_df[f"y{y}_b{b}_y"] - val_df[f"y{y-1}_b{b}_y"]

            test_df[f"y{y}_b{b}_dx"] = test_df[f"y{y}_b{b}_x"] - test_df[f"y{y-1}_b{b}_x"]
            test_df[f"y{y}_b{b}_dy"] = test_df[f"y{y}_b{b}_y"] - test_df[f"y{y-1}_b{b}_y"]

    # Оставляем только скорости и приращения
    keywords = ["order0", "vx", "vy", "dx", "dy"]
    
    train_df = train_df[[col for col in train_df.columns if any(kw in col for kw in keywords)]]
    val_df = val_df[[col for col in val_df.columns if any(kw in col for kw in keywords)]]
    test_df = test_df[[col for col in test_df.columns if any(kw in col for kw in keywords)]]

    # Скорости по x и по y заменим на одну по пифагору
    for y in range(0, 30):
        for b in [0, 1, 2]:
            # train_df
            train_df[f'y{y}_b{b}_v'] = np.sqrt(np.pow(train_df[f"y{y}_b{b}_vx"], 2) +
                                               np.pow(train_df[f"y{y}_b{b}_vy"], 2))
            train_df.drop(columns=[f"y{y}_b{b}_vx", f"y{y}_b{b}_vy"] ,inplace=True)

            # val_df
            val_df[f'y{y}_b{b}_v'] = np.sqrt(np.pow(val_df[f"y{y}_b{b}_vx"], 2) + 
                                             np.pow(val_df[f"y{y}_b{b}_vy"], 2))
            val_df.drop(columns=[f"y{y}_b{b}_vx", f"y{y}_b{b}_vy"] ,inplace=True)

            # test_df
            test_df[f'y{y}_b{b}_v'] = np.sqrt(np.pow(test_df[f"y{y}_b{b}_vx"], 2) + 
                                              np.pow(test_df[f"y{y}_b{b}_vy"], 2))
            test_df.drop(columns=[f"y{y}_b{b}_vx", f"y{y}_b{b}_vy"] ,inplace=True)

    # Добавим фичу - длина траектории, sum(sqrt(dx^2 + dy^2))
    train_df['len_traj0'] = 0.0
    train_df['len_traj1'] = 0.0
    train_df['len_traj2'] = 0.0

    val_df['len_traj0'] = 0.0
    val_df['len_traj1'] = 0.0
    val_df['len_traj2'] = 0.0

    test_df['len_traj0'] = 0.0
    test_df['len_traj1'] = 0.0
    test_df['len_traj2'] = 0.0

    for y in range(1, 30):
        train_df["len_traj0"] += np.sqrt(np.pow(train_df[f"y{y}_b0_dx"] , 2) + 
                                         np.pow(train_df[f"y{y}_b0_dy"] , 2))
        train_df["len_traj1"] += np.sqrt(np.pow(train_df[f"y{y}_b1_dx"] , 2) + 
                                         np.pow(train_df[f"y{y}_b1_dy"] , 2))
        train_df["len_traj2"] += np.sqrt(np.pow(train_df[f"y{y}_b2_dx"] , 2) + 
                                         np.pow(train_df[f"y{y}_b2_dy"] , 2))

        val_df["len_traj0"] += np.sqrt(np.pow(val_df[f"y{y}_b0_dx"] , 2) + 
                                         np.pow(val_df[f"y{y}_b0_dy"] , 2))
        val_df["len_traj1"] += np.sqrt(np.pow(val_df[f"y{y}_b1_dx"] , 2) + 
                                         np.pow(val_df[f"y{y}_b1_dy"] , 2))
        val_df["len_traj2"] += np.sqrt(np.pow(val_df[f"y{y}_b2_dx"] , 2) + 
                                         np.pow(val_df[f"y{y}_b2_dy"] , 2))

        test_df["len_traj0"] += np.sqrt(np.pow(test_df[f"y{y}_b0_dx"] , 2) + 
                                         np.pow(test_df[f"y{y}_b0_dy"] , 2))
        test_df["len_traj1"] += np.sqrt(np.pow(test_df[f"y{y}_b1_dx"] , 2) + 
                                         np.pow(test_df[f"y{y}_b1_dy"] , 2))
        test_df["len_traj2"] += np.sqrt(np.pow(test_df[f"y{y}_b2_dx"] , 2) + 
                                         np.pow(test_df[f"y{y}_b2_dy"] , 2))

    # Добавим фичу - средняя скорость, mean(v)
    cols = []

    for y in range(0, 30):
        for b in [0, 1, 2]:
            cols.append(f"y{y}_b{b}_v")
    
    train_df["mean_v"] = train_df[cols].mean(axis=1)
    val_df["mean_v"] = val_df[cols].mean(axis=1)
    test_df["mean_v"] = test_df[cols].mean(axis=1)

    # Добавим фичу - максимальная скорость, max(v)
    train_df["max_v"] = train_df[cols].max(axis=1)
    val_df["max_v"] = val_df[cols].max(axis=1)
    test_df["max_v"] = test_df[cols].max(axis=1)

    # scaling
    feature_cols = [c for c in train_df.columns if c not in ["order0", "order1", "order2"]]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    X_train_scaled = scaler.transform(train_df[feature_cols])
    X_val_scaled   = scaler.transform(val_df[feature_cols])
    X_test_scaled  = scaler.transform(test_df[feature_cols])

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_df["order0"].values, dtype=torch.long).to(device)

    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_df["order0"].values, dtype=torch.long).to(device)

    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    return X_train, y_train, X_val, y_val, X_test

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(269, 269),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(269, 200),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(150, 50),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),

            nn.Linear(50, 3),
        )  
        # python3 hw1.py --num_epoches=250 --batch_size=4096 --lr=0.0012

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
        

def init_model(lr):
    print(f"Using {device} device")
    print(f"Cuda = {torch.cuda.is_available()}")
    model = MLP().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= X_train.size(0)
        if epoch % 1 == 0:
            model.eval()
            val_outputs = model.forward(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)

            score = accuracy_score(y_val.detach().cpu().numpy(),
                                   torch.argmax(val_outputs, dim=1).detach().cpu().numpy())
            print(f'Val: Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss.item()}, score: {score}')
    return model


def main(args):
    ### YOUR CODE HERE

    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)

    # Initialize model
    model, criterion, optimizer = init_model(float(args.lr))

    # Train model
    model = train(model, criterion, optimizer, 
                X_train, y_train, X_val, y_val, 
                int(args.num_epoches), int(args.batch_size))

    # Predict on test set
    model.eval()
    y_pred = model.forward(X_test) 
    y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
    

    df = pd.DataFrame(y_pred, columns=['order0'])

    df['Id'] = range(1, len(y_pred)+1)
    df = df[['Id', 'order0']] 

    df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='../data/train.csv')
    parser.add_argument('--val_csv', default='../data/val.csv')
    parser.add_argument('--test_csv', default='../data/test.csv')
    parser.add_argument('--out_csv', default='../data/submission.csv')
    parser.add_argument('--lr', default=0)
    parser.add_argument('--batch_size', default=0)
    parser.add_argument('--num_epoches', default=0)

    args = parser.parse_args()
    main(args)
