from ucimlrepo import fetch_ucirepo 
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization, ActivityRegularization, Dropout
from keras.regularizers import l2

# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data  
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 

inputs = keras.Input(shape=(21,))
preFirstL =layers.Dense(128, activation = "relu")(inputs)
firstL =layers.Dense(64, activation = "relu")(preFirstL)
secondL = layers.Dense(32, activation="relu")(firstL)
thirdL = layers.Dense(16, activation="relu")(secondL) #model
fourthL = layers.Dense(8, activation="relu")(thirdL)
fifthL = layers.Dense(4, activation="relu")(fourthL)
sixthL = layers.Dense(2, activation="relu")(fifthL)


model = keras.Model(inputs=inputs, outputs=sixthL,name = "diabetes")

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],

)
fitting = model.fit(X_train, y_train, batch_size=16, epochs=5)
testing = model.evaluate(X_test, y_test, verbose=2)
print(testing[1])