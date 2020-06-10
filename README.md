# Dota 2 Simulator
Deep Learning-based simulator of DOTA 2 for ML training purposes.

Import the module:
```
from DOTA_simulator import *
```

## Training the model
Create a simulator:
```
sim = DotaSim()
```

Load data from a path regex:
```
sim.load_data("games_data/rd_cumulated_*")
```

Select a model (`unet` for UNet, `ffnet` for a feed-forward deep network):
```
sim.set_model('unet')
```

Train the model for 3 epochs:
```
sim.train(epochs=3, batch_size=128)
```
Train until it overfits, for max 100 epochs:
```
sim.train(epochs=100, batch_size=128, limit_overfit=True)
```

Save the model as  `test_model.dotanet`, and the scaler as `test_model.scaler`:
```
sim.save_model("test_model")
```
## Using a pre-built model:
```
sim.set_model(name="test_model")
```

## Gym-like simulator interface:
Reset the environment to first game state:
```
sim.reset()
```

Simulate one step with the 8th action (move NorthWest) and get the state:
```
state = sim.step(action=8)
```

Print time and position:
```
sim.render()
```

Get the state:
```
sim.dota_state  # With all features
sim.state       # With only the non-constant features
```

## Constant features management
Constant features are removed to allow for a smaller model and to reduce training time.
All constant features observed in random play are listed in `constant_features.py`, in the `constant_features` dictionary in the form: `{feature_id : constant_value # Feature name}`. Some of them may stay constant beacause of the random agent's very basic behavior, it is hence strongly suggested to comment some of the lines before training, based on your knowledge of the game and expectations.
