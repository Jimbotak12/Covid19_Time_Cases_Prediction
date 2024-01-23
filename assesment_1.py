# %%
#1. Setup - mainly importing packages
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from window_generator import WindowGenerator

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
# load data
URL = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv"
# Load dataset as DataFrame
df = pd.read_csv(URL)
# Contain timesteps into a separate variable
date_time = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
# %%
print("Data info:\n",df.info())
print("--------------------")
print("Data describe:\n",df.describe().transpose())
print("--------------------")
print("First 5 data:\n",df.head())
# %%
# Remove the cluster dataset
df = df.drop(['cluster_import','cluster_religious','cluster_community','cluster_highRisk','cluster_education','cluster_detentionCentre','cluster_workplace'], axis=1)
# Take the selected data colomn
# selected_columns = ['cases_new','cases_import','cases_recovered','cases_active']
# df = df[selected_columns]
# %%
# Data Inspection
plot_cols = ['cases_new']
plot_features = df[plot_cols]
plot_features_index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
# %%
# train-test split
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
# %%
# Data Normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
# %%
# Data inspection after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# %%
# Create the single step model
wide_window = WindowGenerator(
    input_width=30, 
    label_width=30, 
    shift=0, 
    label_columns=['cases_new'],
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    batch_size=73
    )
# single_step.plot(plot_col='cases_new')
# single_step
# %%
# create single step LSTM model
model = keras.Sequential()
model.add(keras.layers.LSTM(55, return_sequences=True))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(42, return_sequences=True))
model.add(keras.layers.Dense(1))
# %%
# Create the tensorboard callback project
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log_single", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(logpath)

def compile_and_fit(model, window, patience=3, max_epochs=20, lr=0.001):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min',
 restore_best_weights=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                metrics=[(tf.keras.metrics.MeanAbsolutePercentageError())])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[tb,early_stopping])
    return history
# %% 
# compile and fit the single step model
hist_1 = compile_and_fit(model, wide_window, patience=3, max_epochs=60, lr=0.001)
keras.utils.plot_model(model)
# %%
fig = plt.figure()
plt.plot(hist_1.history['loss'], color='red', label='loss')
plt.plot(hist_1.history['val_loss'], color='blue', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
plt.plot(hist_1.history['mean_absolute_percentage_error'], color='red', label='mape')
plt.plot(hist_1.history['val_mean_absolute_percentage_error'], color='blue', label='val_mape')
fig.suptitle('MAPE', fontsize=20)
plt.legend(loc='lower right')
plt.show()
# %%
# Evaluate the model
test_1 = lstm_single_step.evaluate(wide_window.test)
val_1 = lstm_single_step.evaluate(wide_window.val)
print("\nValidation MAPE: ",round((val_1[1]),4))
print("Test MAPE: ",round((test_1[1]),4))
# %%
# Plot the resultt
wide_window.plot(model=model, plot_col='cases_new')
# %%
