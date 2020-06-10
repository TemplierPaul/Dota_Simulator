from DotaUNet.UNet import *

import joblib
import glob

import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time


def action_to_tensor(action):
    vect = preprocessing.action_to_vect(action)
    vect = torch.from_numpy(vect).unsqueeze(dim=0).unsqueeze(dim=0)
    return vect




class DotaData(torch.utils.data.Dataset):
    def __init__(self, state, action, next_state):
        self.state = state.double()
        self.action = action.double()
        self.next_state = next_state.double()

    def __getitem__(self, item):
        sample = {
            "state":self.state[item],
            "action":self.action[item],
            "next_state":self.next_state[item]
        }
        return sample

    def __len__(self):
        assert self.state.shape[0] == self.action.shape[0]
        return self.state.shape[0]

    @property
    def shape(self):
        return self.state.shape




class DotaSim():
    def __init__(self, state_length=310, action_length=23):
        self.model = None
        self.data = None

        self.scaler = None
        self.remove_constant_features = True

        self.dota_state = None
        self.step_nb = 0

        self.state = None
        self.state_length = state_length
        self.action_length = action_length


    def set_model(self, type=None, name=None, model=None):
        if name is not None:
            self.model = joblib.load(name+".dotanet")
            self.scaler = joblib.load(name+'.scaler')
        elif model is not None:
            self.model = model
        elif type == "unet":
            self.model = DotaUNet(self.state_length, self.action_length).double()
        elif type == "ffnet":
            self.model = FFNet(self.state_length, self.action_length,
                             n_hidden=256, n_hidden_layer=1)
        else:
            raise Exception('No such net')
        print("Model dimensions: (%d, %d) => %d" %(self.state_length, self.action_length, self.state_length))
        print("Model type:\n", self.model)
        return self

    def save_model(self, name):
        joblib.dump(self.model, name+'.dotanet')
        joblib.dump(self.scaler, name+'.scaler')

    def load_data(self, path="../games_data/rd_cumulated*", replace=True, remove_constant_features=True, verbose=True):
        self.remove_constant_features = remove_constant_features
        if replace:
            self.scaler = None

        files = glob.glob(path)
        print(len(files), 'Files to load')
        if len(files)==0:
            raise Exception("No files to load")
        df_state, df_action, df_next_state = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for f in files:
            # Get csv
            df = pd.read_csv(f, index_col=0)

            # remove constant features
            if self.remove_constant_features:
                df = preprocessing.remove_constants(df)

            # Create the inputs and outputs
            next_df = df.drop(columns=["action"])
            next_df = next_df.drop(index=0, axis=0)
            df_next_state = pd.concat([df_next_state, next_df], axis=0)


            df = df.drop(index=len(df) - 1) # Drop last record (unknown output)

            actions = preprocessing.actions_to_vect(df['action'])
            df_state=pd.concat([df_state, df.drop(columns="action")], axis=0)
            df_action = pd.concat([df_action, actions], axis=0)



            if verbose: print('%s   \tImported: %d lines' % (f, len(df)), end='\n')

        # Scale
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(df_state)

        df_state = self.scaler.transform(df_state)
        df_next_state = self.scaler.transform(df_next_state)


        # Add to the dataset structure
        if replace:
            df_state = torch.unsqueeze(torch.from_numpy(df_state), 1)
            df_action = torch.unsqueeze(torch.from_numpy(df_action.values), 1)
            df_next_state = torch.unsqueeze(torch.from_numpy(df_next_state), 1)
        else:
            df_state = torch.cat((self.data.state, torch.unsqueeze(torch.from_numpy(df_state), 1)), dim=0)
            df_action = torch.cat((self.data.action, torch.unsqueeze(torch.from_numpy(df_action.values), 1)), dim=0)
            df_next_state = torch.cat((self.data.next_state, torch.unsqueeze(torch.from_numpy(df_next_state), 1)),
                                      dim=0)

        self.data = DotaData(df_state, df_action, df_next_state)

        # Update model dimensions
        self.state_length  = self.data.shape[2]
        print(self.data.shape[0], "Lines in the dataset")
        return self

    def train(self, epochs=100, batch_size=32, limit_overfit=False, validation_split=0.2):
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size,
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size,
                                                        sampler=valid_sampler)

        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_func = torch.nn.MSELoss()

        # Stopping criterion
        epoch = 0
        epoch_losses = {"train": [], "val": []}

        print("Training...")
        t0 = time.time()
        for epoch in range(epochs):
            # Train:
            batch_losses = []
            for i_batch, sample in enumerate(train_loader):
                state = sample['state']
                action = sample['action']
                next_state = sample['next_state']

                pred_next_state = self.model(state, action)

                loss = loss_func(pred_next_state, next_state)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

                batch_losses.append(loss.item())

            val_losses=[]
            with torch.no_grad():
                for i_batch, sample in enumerate(validation_loader):
                    state = sample['state']
                    action = sample['action']
                    next_state = sample['next_state']

                    pred_next_state = self.model(state, action)

                    loss = loss_func(pred_next_state, next_state)
                    val_losses.append(loss)
            loss = np.array(batch_losses).mean()
            val_loss = np.array(val_losses).mean()
            epoch_losses['train'].append(loss)
            epoch_losses['val'].append(val_loss)
            print("> %d \t/ %d \tTrain Loss: %.4f  \t| Validation Loss: %.4f" %(epoch+1, epochs, loss, val_loss))

            if limit_overfit :
                if len(val_losses)>1 and val_loss > max(val_losses[-6:-1]):
                    print("Overfitting:\n", np.array(val_losses[-6:-1]))
                    break

        print("Training time: %.1f" %(time.time()-t0))
        return self

    def transform(self, vect):
        assert self.scaler is not None

        # Remove constants
        if self.remove_constant_features:
            vect_transformed  = preprocessing.remove_constants(vect)
        else:
            vect_transformed = vect

        # Transform state : remove features and scale
        vect_transformed  = self.scaler.transform(vect_transformed.reshape(1, -1))
        vect_transformed = torch.from_numpy(vect_transformed).unsqueeze(dim=0)
        return vect_transformed

    def inverse_transform(self, vect):
        print(0, vect.shape)
        vect_transformed = torch.squeeze(vect.detach())
        print(1, vect_transformed.shape)

        # Scale back
        vect_transformed = self.scaler.inverse_transform(vect_transformed)
        print(2, vect_transformed.shape)
        # Add constants
        if self.remove_constant_features:
            vect_transformed = preprocessing.add_constants(vect_transformed)
        print(3, vect_transformed.shape)
        return vect_transformed

    def render(self):
        t = self.dota_state[56]
        x, y = self.dota_state[26:28]
        print("%d > Time: %d | (%d , %d)" % (self.step_nb, t, x, y))
        return self.dota_state

    def reset(self):
        self.dota_state = np.array(pd.read_csv("Init_state.csv", index_col=0, names=['init'])['init'])
        self.state = self.transform(self.dota_state)
        self.step_nb = 0
        print("Env Reset")

    def step(self, action):
        if self.dota_state is None:
            self.reset()

        # Predict and store next state
        self.state = self.model(self.state, action_to_tensor(action))

        self.dota_state = self.inverse_transform(self.state)

        self.step_nb += 1

        return self.dota_state

if __name__=="__main__":
    # sim = DotaSim()\
    #     .load_data("games_data/rd_cumulated_*")\
    #     .set_model(name='unet')\
    #     .train(epochs=100, batch_size=128, limit_overfit=True)
    t = action_to_tensor(8)
    print(t.shape)
    print(type(t))



























