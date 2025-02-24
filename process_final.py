import pickle
import numpy as np
import pandas as pd
from datasets import Dataset

with open("best_predictions.pkl", 'rb') as f:
    data = pickle.load(f)

print(data)

valid_prediction = data['valid_predictions'].predictions
valid_prediction = np.argmax(valid_prediction, axis=1)

with open("dev.txt", "w") as f:
    f.writelines([str(p) + '\n' for p in valid_prediction])

final_prediction = data['final_predictions'].predictions
final_predictions = np.argmax(final_prediction, axis=1)

with open("test.txt", "w") as f:
    f.writelines([str(p) + '\n' for p in final_predictions])