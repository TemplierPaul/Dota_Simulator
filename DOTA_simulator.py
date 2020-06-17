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


    def set_model(self, type=None, name=None, model=None, depth=1, height=256):
        if name is not None:
            self.model = joblib.load(name+".dotanet")
            self.scaler = joblib.load(name+'.scaler')
        elif model is not None:
            self.model = model
        elif type == "unet":
            self.model = DotaUNet(self.state_length, self.action_length).double()
        elif type == "small-unet":
            self.model = SmallUNet(self.state_length, self.action_length).double()
        elif type == "deep-unet":
            self.model = DeepUNet(self.state_length, self.action_length).double()
        elif type == "ffnet":
            self.model = FFNet(self.state_length, self.action_length,
                             n_hidden=height, n_hidden_layer=depth).double()
        else:
            raise Exception('No such net')
#         print("Model dimensions: (%d, %d) => %d" %(self.state_length, self.action_length, self.state_length))
        print("Model type:\n", self.model)
        return self

    def save_model(self, name):
        joblib.dump(self.model, name+'.dotanet')
        joblib.dump(self.scaler, name+'.scaler')

    def load_data(self, path="../games_data/*", replace=True, remove_constant_features=True, verbose=True):
        import json
        with open('features_list.json', 'r') as fp:
            features_list = json.load(fp)

        self.remove_constant_features = remove_constant_features
        if replace:
            self.scaler = None

        files = glob.glob(path)
        print(len(files), 'Files to load')
        if len(files)==0:
            raise Exception("No files to load")
        df_state, df_action, df_next_state = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for f in files:
            df = pd.read_csv(f, index_col=0)

            # remove constant features
            if self.remove_constant_features:
                df = preprocessing.remove_constants(df)

            next_df = df.drop(columns=["action"])
            next_df = next_df.drop(index=0, axis=0)  # Drop first
            df_next_state = pd.concat([df_next_state, next_df], axis=0)

            actions = preprocessing.actions_to_vect(df['action'])
            actions = actions.drop(index=0, axis=0)  # Drop first
            df_action = pd.concat([df_action, actions], axis=0)

            df = df.drop(index=len(df) - 1)  # Drop last record (unknown output)
            df_state = pd.concat([df_state, df.drop(columns="action")], axis=0)

            if verbose: print('%s   \tImported: %d lines' % (f, len(df)), end='\n')

        print(df_state.shape, df_action.shape, df_next_state.shape)

        # Reset index
        for d in [df_state, df_action, df_next_state]:
            d.reset_index(inplace=True, drop=True)
            print(d.shape)

        # Filter
        filter_new_games = df_state['56'] < df_next_state['56']
        filter_loc_diff = df_next_state['26'] - df_state['26'] > -4000
        filter_dead = df_state['2'] > 0

        df_state = df_state[filter_new_games & filter_loc_diff & filter_dead]
        df_next_state = df_next_state[filter_new_games & filter_loc_diff & filter_dead]
        df_action = df_action[filter_new_games & filter_loc_diff & filter_dead]

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

    def train(self, epochs=100, batch_size=32, limit_overfit=False, validation_split=0.2, plot=True):
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

        learning_rate = 0.01
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)
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
                if type(limit_overfit)==int:
                    last_losses = epoch_losses['val'][(-1 * limit_overfit - 1):-1]
                else:
                    last_losses = epoch_losses['val'][-21:-1]
                if len(last_losses)>=1 and val_loss > max(last_losses):
                    print("Stopped because of stagnating loss. Previous loss values:\n", np.array(last_losses))
                    break

        print("Training time: %.1fs" %(time.time()-t0))
        if plot:
            plt.figure(figsize=(8, 8))
            for t in epoch_losses.keys():
                plt.plot(epoch_losses[t], label=t + ' loss')
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
            plt.ylim(0, max(max(epoch_losses['train']), max(epoch_losses['val'])))
            plt.legend()
            plt.show()
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
        vect_transformed = torch.squeeze(vect.detach())

        # Scale back
        vect_transformed = self.scaler.inverse_transform(vect_transformed)
        # Add constants
        if self.remove_constant_features:
            vect_transformed = preprocessing.add_constants(vect_transformed)
        return vect_transformed

    def render(self):
        t = self.dota_state[56]
        x, y = self.dota_state[26:28]
        print("%d > Time: %d | (%d , %d)" % (self.step_nb, t, x, y))
        return self.dota_state

    def reset(self, init_state="Init_state.csv"):
        self.dota_state = np.array(pd.read_csv(init_state, index_col=0, names=['init'])['init'])
        self.state = self.transform(self.dota_state)
        self.step_nb = 0
#         print("Env Reset")
        return self.dota_state

    def step(self, action):
        if self.dota_state is None:
            self.reset()

        # Predict and store next state
        self.state = self.model(self.state, action_to_tensor(action))

        self.dota_state = self.inverse_transform(self.state)

        self.step_nb += 1

        return self.dota_state

    def run_steps(self, decision_function, n_steps=100, render=True):
        assert self.model is not None
        s = self.reset()
        if render:
            self.render()
        actions = []
        for i in range(n_steps):
            a = decision_function(s)
            actions.append(a)
            s = self.step(a)
            if render:
                self.render()
        return actions


if __name__=="__main__":
    # sim = DotaSim()\
    #     .load_data("games_data/rd_cumulated_*")\
    #     .set_model(name='unet')\
    #     .train(epochs=100, batch_size=128, limit_overfit=True)
    t = action_to_tensor(8)
    print(t.shape)
    print(type(t))



























