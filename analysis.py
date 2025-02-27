import requests
import pandas as pd
import matplotlib.pyplot as plt

dataset_url = "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/refs/heads/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv"
with open("dataset.tsv",'wb') as outf:
  response = requests.get(dataset_url)
  outf.write(response.content)

pcl_df = pd.read_csv("dataset.tsv", sep="\t", skiprows=lambda x: x in range(2), header=0, names=['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label'])
pcl_df.set_index('par_id', inplace=True)

dev_labels_url = "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/practice%20splits/dev_semeval_parids-labels.csv"
with open("dev_labels.csv",'wb') as outf:
  response = requests.get(dev_labels_url)
  outf.write(response.content)

dev_labels_df = pd.read_csv("dev_labels.csv")
dev_labels_df.set_index('par_id', inplace=True)

train_labels_url = "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/practice%20splits/train_semeval_parids-labels.csv"
with open("train_labels.csv",'wb') as outf:
  response = requests.get(train_labels_url)
  outf.write(response.content)

train_labels_df = pd.read_csv("train_labels.csv")
train_labels_df.set_index('par_id', inplace=True)

def merge_with_labels(df, binary=True):
    df = pd.merge(df, pcl_df, on='par_id', how='left')
    df.drop(columns=['label_x', 'art_id', 'country_code'], inplace=True)
    df.rename(columns={'label_y': 'label'}, inplace=True)
    if binary:
        df['label'] = df['label'].apply(lambda x: 1 if x > 1 else 0)
    else:
        df['label'] = df['label'].apply(lambda x: int(x))
    return df

dev_df = merge_with_labels(dev_labels_df)
dev_original_labels_df = merge_with_labels(dev_labels_df, binary=False)
train_df = merge_with_labels(train_labels_df)
train_original_labels_df = merge_with_labels(train_labels_df, binary=False)

with open("dev.txt", "r") as f:
    lines = [int(line.strip()) for line in f.readlines()]
    output_df = pd.DataFrame(lines, columns=['prediction'])

row_num = 0
for i, row in dev_df.iterrows():
    dev_df.at[i, 'prediction'] = str(output_df.at[row_num, 'prediction'])
    dev_original_labels_df.at[i, 'prediction'] = str(output_df.at[row_num, 'prediction'])
    row_num += 1

def print_metrics(df, binary=True, cutoff=1):
    if binary:
        tp = df[(df['prediction'] == "1") & (df['label'] == 1)].shape[0]
        tn = df[(df['prediction'] == "0") & (df['label'] == 0)].shape[0]
        fn = df[(df['prediction'] == "0") & (df['label'] == 1)].shape[0]
        fp = df[(df['prediction'] == "1") & (df['label'] == 0)].shape[0]
    else:
        tp = df[(df['prediction'] == "1") & (df['label'] > cutoff)].shape[0]
        tn = df[(df['prediction'] == "0") & (df['label'] <= cutoff)].shape[0]
        fn = df[(df['prediction'] == "0") & (df['label'] > cutoff)].shape[0]
        fp = df[(df['prediction'] == "1") & (df['label'] <= cutoff)].shape[0]

    correct = tp + tn
    total = tp + tn + fp + fn
    if total == 0:
        accuracy = 0
    else:
        accuracy = correct / total

    print(f"TP: {tp} TN: {tn} FN: {fn} FP: {fp}")

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1-Score: {f1_score}")
    print(f"Accuracy: {accuracy}")

    return recall, precision, f1_score, accuracy

print_metrics(dev_df)

dev_df["text_length"] = dev_df["text"].apply(lambda x: len(str(x)))

num_buckets = 5

for i in range(1, num_buckets + 1):
    if i == 1:
        bucket_df = dev_df[dev_df['text_length'] <= 128]
    elif i == 2:
        bucket_df = dev_df[(dev_df['text_length'] > 128) & (dev_df['text_length'] <= 256)]
    elif i == 3:
        bucket_df = dev_df[(dev_df['text_length'] > 256) & (dev_df['text_length'] <= 384)]
    elif i == 4:
        bucket_df = dev_df[(dev_df['text_length'] > 384) & (dev_df['text_length'] <= 512)]
    else:
        bucket_df = dev_df[dev_df['text_length'] > 512]
    print(f"Bucket {i} ({bucket_df['text_length'].min() if len(bucket_df) > 0 else 'N/A'} - {bucket_df['text_length'].max() if len(bucket_df) > 0 else 'N/A'}) length: {len(bucket_df)}")
    recall, precision, f1_score, accuracy = print_metrics(bucket_df)
    print("\n")
    
    # Store metrics for plotting
    if i == 1:
        recalls = []
        precisions = []
        f1_scores = []
        bucket_ranges = []
        accuracies = []
    
    recalls.append(recall)
    precisions.append(precision) 
    f1_scores.append(f1_score)
    accuracies.append(accuracy)
    bucket_ranges.append(f"{bucket_df['text_length'].min():.0f}-{bucket_df['text_length'].max():.0f}")
    
    if i == num_buckets:
        
        x = range(len(bucket_ranges))
        width = 0.2  # Reduced width to fit 4 bars
        
        plt.figure(figsize=(10,6))
        plt.bar([i-1.5*width for i in x], recalls, width, label='Recall', color='darkgray')
        plt.bar([i-0.5*width for i in x], precisions, width, label='Precision', color='gray')
        plt.bar([i+0.5*width for i in x], f1_scores, width, label='F1-Score', color='black')
        
        # # Add trendlines
        # z = np.polyfit(x, recalls, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), "b--", alpha=0.8, label='Recall trend')
        
        # z = np.polyfit(x, precisions, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), "C1--", alpha=0.8, label='Precision trend')
        
        # z = np.polyfit(x, f1_scores, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), "g--", alpha=0.8, label='F1 trend')

        # z = np.polyfit(x, accuracies, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), "r--", alpha=0.8, label='Accuracy trend')

        bucket_text = [f"{r.split('-')[0]} - {r.split('-')[1]}" for r in bucket_ranges]
        bucket_text[0] = "0 - 128"
        bucket_text[1] = "128 - 256"
        bucket_text[2] = "256 - 384"
        bucket_text[3] = "384 - 512"
        bucket_text[4] = "512 - 1500"

        plt.xlabel('Text Length')
        plt.ylabel('Score')
        plt.title('Metrics by Text Length')
        plt.xticks(x, bucket_text, rotation=0)
        plt.legend()
        plt.tight_layout()
        plt.show()



zero_labels = train_df[train_df['label'] == 0]
one_labels = train_df[train_df['label'] == 1]

print(f"Zero labels: {len(zero_labels)}")
print(f"One labels: {len(one_labels)}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# train_df['text_length'] = train_df['text'].apply(len)

# # Plot distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(data=train_df, x='text_length', hue='label', bins=100, kde=False, alpha=1, palette={0: '0.7', 1: '0.2'}, multiple='stack')

# plt.xlim(0, 900)
# plt.xlabel("Text Length")
# plt.ylabel("Count")
# plt.legend(title="Label", labels=["1", "0"])
# plt.show()


train_df = pd.concat([train_df, dev_df])

# Compute text lengths
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x)))
bins = [0, 128, 256, 384, 512, 1500]

# Bin the text lengths
train_df['text_length_binned'] = pd.cut(train_df['text_length'], bins=bins)

# Group by binned text lengths and calculate the count of each label
label_counts_binned = train_df.groupby('text_length_binned')['label'].value_counts().unstack(fill_value=0)

# Calculate the ratio of label 1 to label 0 for each bin
label_counts_binned['ratio'] = label_counts_binned[1] / (label_counts_binned[0] + label_counts_binned[1]) * 100

# Plot the ratio of label 1 to label 0 against binned text length
plt.figure(figsize=(10, 6))
sns.barplot(x=[f"{int(bin.left)} - {int(bin.right)}" for bin in label_counts_binned.index], y=label_counts_binned['ratio'], color='gray')

# Customize the plot
plt.xlabel("Text Length")
plt.ylabel("Percentage of samples containing PCL")
plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility
plt.grid(True)
plt.show()

train_df['avg_word_length'] = train_df['text'].apply(lambda x: len(x.split()) if len(x.split()) > 0 else 0)
sns.histplot(data=train_df, x='avg_word_length', hue='label', bins=40, kde=False, alpha=0.6)
plt.xlabel("Average Text Length")
plt.xlim(0, 175)
plt.show()

Split the dev_original_labels_df into dfs based on pcl value (0-7)

for i in range(0, 5):
    df = dev_original_labels_df[dev_original_labels_df['label'] == i]
    print(df.head())
    print(f"Label {i}: {len(df)}")
    recall, precision, f1_score, accuracy = print_metrics(df, binary=False, cutoff=1)
    print("\n")

    # Store metrics for plotting
    if i == 0:
        recalls = []
        precisions = []
        f1_scores = []
        accuracies = []
        labels = []
    
    recalls.append(recall)
    precisions.append(precision) 
    f1_scores.append(f1_score)
    accuracies.append(accuracy)
    labels.append(i)

    if i == 4:
        plt.figure(figsize=(10, 6))
        
        plt.plot(labels, accuracies, marker='o', label='Accuracy', color='gray')
        plt.xticks(range(0, max(labels)+1, 1))
        
        plt.xlabel('PCL Label')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Metrics by PCL Label')
        plt.grid(True)
        plt.show()

# Print the length of each category in the train_df
category_lengths = train_df['keyword'].value_counts().sort_values(ascending=False)
category_pcl_lengths = train_original_labels_df.groupby('keyword')['label'].value_counts().unstack(fill_value=0)
print(category_lengths)
print(category_pcl_lengths)

# Split dev_df into groups based on the "keyword" column (use group by)
keyword_groups = dev_df.groupby('keyword')

# Store metrics for all keywords first
metrics_data = []
for keyword, group in keyword_groups:
    print(f"Keyword: {keyword}")
    print(f"Number of samples: {len(group)}")
    recall, precision, f1_score, accuracy = print_metrics(group)
    metrics_data.append({
        'keyword': keyword,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'accuracy': accuracy
    })
    print("\n")

# Sort by F1 score
metrics_data.sort(key=lambda x: x['f1_score'], reverse=True)

# Extract sorted metrics
recalls = [d['recall'] for d in metrics_data]
precisions = [d['precision'] for d in metrics_data]
f1_scores = [d['f1_score'] for d in metrics_data]
keywords = [d['keyword'] for d in metrics_data]

# Plot sorted metrics
x = range(len(keywords))
width = 0.2

plt.figure(figsize=(15,6))
plt.bar([i-1.5*width for i in x], recalls, width, label='Recall', color='darkgray')
plt.bar([i-0.5*width for i in x], precisions, width, label='Precision', color='gray')
plt.bar([i+0.5*width for i in x], f1_scores, width, label='F1-Score', color='black')

plt.xlabel('Keywords')
plt.ylabel('Metrics')
plt.title('Model Performance Metrics by Keyword (Sorted by F1-Score)')
plt.xticks(x, keywords, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()


# load in epochs.json
import json
with open("epochs.json", "r") as f:
    epochs = json.load(f)

print([epochs["log_history"][i] for i in range(1, len(epochs["log_history"]), 3)])

# Extract epochs and accuracies
eval_entries = [epochs["log_history"][i] for i in range(1, len(epochs["log_history"]), 3)]
epochs_list = [entry["epoch"] for entry in eval_entries]
accuracies = [entry["eval_accuracy"] for entry in eval_entries]
f1_scores = [entry["eval_f1"] for entry in eval_entries]

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(epochs_list, accuracies, marker='o', color='gray', linewidth=2)
plt.plot(epochs_list, f1_scores, marker='o', color='black', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy') 
plt.title('Model Accuracy vs. Epoch')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
