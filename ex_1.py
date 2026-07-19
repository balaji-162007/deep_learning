import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])
model = Sequential()
model.add(Dense(units=4, input_dim=2, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(x, y, epochs=500, verbose=0)
prediction = model.predict(x)
print("Name: Balaji.s")
print("Reg.No: 814724243022")
print("Input:")
print(x)
print("\nExpected Output:")
print(y)
print("\nPredicted Output:")
print(prediction)
print("\nRounded Prediction:")
print(np.round(prediction))
