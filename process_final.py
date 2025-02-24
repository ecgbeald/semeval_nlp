import pickle
import numpy as np
import pandas as pd
from datasets import Dataset

# with open("best_predictions.pkl", 'rb') as f:
#     data = pickle.load(f)

# print(data)

# valid_prediction = data['valid_predictions'].predictions
# valid_prediction = np.argmax(valid_prediction, axis=1)

# with open("dev.txt", "w") as f:
#     f.writelines([str(p) + '\n' for p in valid_prediction])

# final_prediction = data['final_predictions'].predictions
# final_predictions = np.argmax(final_prediction, axis=1)

# with open("test.txt", "w") as f:
#     f.writelines([str(p) + '\n' for p in final_predictions])


def parse_test():
    dataset = "~/task4_test.tsv"
    test_df = pd.read_csv(dataset, sep="\t", header=None, names=['par_id', 'art_id', 'keyword', 'country_code', 'text'])
    test_df.drop(columns=['art_id', 'keyword', 'country_code'], inplace=True)
    print(test_df.head())
    dataset_test = Dataset.from_pandas(test_df)
    return dataset_test


dataset = parse_test()
print(dataset)