from sklearn.ensemble import RandomForestRegressor
import random as rnd
from DOTA_simulator import *


### DATA IMPORT

removed_columns_nb = len(preprocessing.constant_features)

test_sim = DotaSim()
def test_import_data():
    test_sim.load_data("games_data/rd_cumulated_*")
    assert type(test_sim.data == DotaData)
    print(test_sim.data.shape)
    assert test_sim.data.shape == (64414, 1, 310-removed_columns_nb)

    assert test_sim.state_length == 310-removed_columns_nb
    assert test_sim.action_length == 23


def test_add_data():
    test_sim.load_data("games_data/rd_cumulated_1*", replace=False)
    assert test_sim.data.shape == (77648, 1, 310-removed_columns_nb)

def test_constants_modifier():
    v = np.array(pd.read_csv("Init_state.csv", index_col=0, names=['init'])['init'])
    v_short = preprocessing.remove_constants(v)
    print(v.shape, "=>", v_short.shape)
    v_long = preprocessing.add_constants(v_short)
    print(v_short.shape, "=>", v_long.shape)
    assert sum(v - v_long) ==0

def test_data_with_constants():
    test_sim.load_data("games_data/rd_cumulated_1*", remove_constant_features=False)
    assert test_sim.data.shape == (13234, 1, 310)

def test_transforms():
    sim = DotaSim().load_data("games_data/rd_cumulated_1*").set_model('unet')#.train(epochs=3)
    v = np.array(pd.read_csv("Init_state.csv", index_col=0, names=['init'])['init'])
    v_t = sim.transform(v)
    print("Transformed", v_t.shape)
    assert v_t.shape == (1, 1, 310-removed_columns_nb)

    next_v = sim.model(v_t, action_to_tensor(8))
    print(next_v.shape)
    next_v = sim.model(next_v, action_to_tensor(8))
    print(next_v.shape)

def test_inverse_transform():
    sim = DotaSim().load_data("games_data/rd_cumulated_1*")
    v = np.array(pd.read_csv("Init_state.csv", index_col=0, names=['init'])['init'])
    print("Before transform", v.shape)
    v_t = sim.transform(v)
    v2 = sim.inverse_transform(v_t)
    print("Inverse transformed", v2.shape)
    assert len(v2) == 310
    assert sum(v-v2) <= 1e-12

### MODEL

def test_set_model():
    test_sim.set_model(type='unet')
    assert test_sim.model is not None

    test_sim.set_model(type='ffnet')
    assert test_sim.model is not None

    rfr = RandomForestRegressor()
    test_sim.set_model(model=rfr)
    assert type(test_sim.model) == RandomForestRegressor

def test_gym_interface():
    sim = DotaSim()\
        .load_data("games_data/rd_cumulated_1*")\
        .set_model('unet').train(epochs=3)
    for i in range(10):
        a = rnd.randint(0, 29)
        s = sim.step(a)
        assert len(s)==310
        sim.render()

def test_train_epochs():
    test_sim.load_data("games_data/rd_cumulated_1*")\
        .set_model(type='unet')\
        .train(epochs=3, batch_size=128)


def test_save_load_model():
    test_sim = DotaSim().load_data("games_data/rd_cumulated_1*") \
        .set_model(type='unet') \
        .train(epochs=3, batch_size=128)
    test_sim.save_model("test_model")
    sim= DotaSim().set_model(name="test_model")
    assert type(sim.model) == DotaUNet




