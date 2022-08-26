
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cwd = os.getcwd()
BASE_DIR = os.path.dirname(os.path.dirname(cwd))
sys.path.insert(1, BASE_DIR)
from base.train_scinet import train_scinet
from base.preprocess_data import preprocess
from exp.training_scinet.utils.plotting import plot_raw_data
from exp.training_scinet.utils.plotting import plot_preprocessed_data
from exp.training_scinet.utils.plotting import plot_loss_curves
from exp.training_scinet.utils.plotting import plot_prediction_examples
import tensorflow as tf

print(BASE_DIR)
raw_data_df = pd.read_csv(BASE_DIR+'\exp\pruebas\datasets\\bs_as_temp_dataset.csv')
raw_data_df.index=raw_data_df["time"]
#raw_data_df["time"]=np.arange(0,len(raw_data_df))#.drop("time",inplace=True,axis=1)
raw_data = raw_data_df.to_numpy()
raw_data_df.plot()

plt.show()
print('Total number of features: {}'.format(raw_data.shape[1]))
print('Total number of timesteps: {}'.format(raw_data.shape[0]))
#print('Size of timestep (dt): {}'.format(raw_data[1, 0]-raw_data[0, 0]))

train=True

#Figure settings
#ax1_x_len = 50
#ax2_x_len = 100
#ax3_x_len = 1000

#plot_raw_data(raw_data, [ax1_x_len, ax2_x_len, ax3_x_len])
#plt.show()

column_names = ["time","temp", "dwpt", "rhum", "prcp","wspd","wdir","pres"]
X_len = 40 #Los reduje 4 veces por la cantidad de muestras que tengo (multiplos de 4)
Y_len = 20
raw_data_pd = pd.DataFrame(raw_data, columns=column_names)

standardization_settings = {
    'per_sample': False,
    'leaky': False,
    'mode': 'lin',  # only if per sample is false, choose from log, sqrt or lin
                    'sqrt_val': 2,  # of course only if mode is sqrt
                    'total mean': [],
                    'total std': []}

preprocessed_data = preprocess(
    data={'bs_as': raw_data_df},
    symbols=['bs_as'],
    data_format=column_names,
    fraction=1,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    X_LEN=X_len,
    Y_LEN=Y_len,
    OVERLAPPING=True,
    STANDARDIZE=True,
    standardization_settings=standardization_settings,
)
print(preprocessed_data['X_train'].shape[0])
print(preprocessed_data['y_train'].shape[0])
print(preprocessed_data['X_test'].shape[0])
print(preprocessed_data['y_test'].shape[0])
print(preprocessed_data['X_val'].shape[0])
print(preprocessed_data['y_val'].shape[0])
#sample_number = np.random.randint(preprocessed_data['X_train'].shape[0])
#X_sample = preprocessed_data['X_train'][sample_number, :, :]
#Y_sample = preprocessed_data['y_train'][sample_number, :, :]

#plot_preprocessed_data(X_sample, Y_sample)
#plt.show()
if train:
    n_epochs = 200
    n_features = 7
    n_features_list = np.arange(n_features)

    results = train_scinet(
        X_train=preprocessed_data['X_train'],
        y_train=preprocessed_data['y_train'],
        X_val=preprocessed_data['X_val'],
        y_val=preprocessed_data['y_val'],
        X_test=preprocessed_data['X_test'],
        y_test=preprocessed_data['y_test'],
        epochs=n_epochs,
        batch_size=64,
        X_LEN=X_len,
        Y_LEN=[Y_len, Y_len],
        output_dim=[n_features, n_features],
        selected_columns=[n_features_list, n_features_list],
        hid_size=32,
        num_levels=2,
        kernel=5,
        dropout=0.3,
        loss_weights=[0.3, 0.7],
        learning_rate=0.005,
        probabilistic=False,
    )

    model = results[0]
    history = results[1]


    #losses = list(history.history.values())

    #plot_loss_curves(n_epochs, losses[1:3]+losses[4:])
    #plt.show()
    model.save('saved_models/modelo_prueba')

else:
    model=new_model = tf.keras.models.load_model('saved_models/modelo_prueba')

n_samples = 6
test_samples = np.random.randint(
    low=0, high=preprocessed_data['X_test'].shape[0], size=n_samples)

x_samples = preprocessed_data['X_train'][test_samples, :, :]
y_true_samples = preprocessed_data['y_train'][test_samples, :, :]
y_pred_samples = model.predict(
    preprocessed_data['X_train'][test_samples, :, :])[-1]

plot_prediction_examples(n_samples,x_samples , y_true_samples, y_pred_samples, )
plt.show()
