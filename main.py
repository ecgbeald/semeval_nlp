import os
import pandas as pd
from datasets import Dataset
import bert_based_uncased
import deberta_trainer
from deep_translator import GoogleTranslator

def merge_with_labels(df, pcl_df):
  df = pd.merge(df, pcl_df, on='par_id', how='left')
  df.drop(columns=['label_x', 'art_id', 'keyword', 'country_code'], inplace=True)
  if "text_x" in df.columns.tolist():
    df["text"] = df["text_x"]
    df.drop(columns=['text_x', 'text_y'], inplace=True)
  df.rename(columns={'label_y': 'label'}, inplace=True)

  df['label'] = df['label'].apply(lambda x: 1 if x > 1 else 0)
  return df


def pre_process():
    dataset = "/vol/bitbucket/rm521/cw/dontpatronizeme_pcl.tsv"
    pcl_df = pd.read_csv(dataset, sep="\t", skiprows=lambda x: x in range(2), header=None, names=['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label'])
    pcl_df.set_index('par_id', inplace=True)

    dev_labels_df = pd.read_csv("/vol/bitbucket/rm521/cw/dev_semeval_parids-labels.csv")
    dev_labels_df.set_index('par_id', inplace=True)

    train_labels_df = pd.read_csv("/vol/bitbucket/rm521/cw/train_semeval_parids-labels.csv")
    train_labels_df.set_index('par_id', inplace=True)

    dev_df = merge_with_labels(dev_labels_df, pcl_df) # 2094 rows
    train_df = merge_with_labels(train_labels_df, pcl_df) # 8375 rows

    return dev_df, train_df

def parse_test():
    dataset = "/vol/bitbucket/rm521/cw/task4_test.tsv"
    test_df = pd.read_csv(dataset, sep="\t", header=None, names=['par_id', 'art_id', 'keyword', 'country_code', 'text'])
    test_df.drop(columns=['art_id', 'keyword', 'country_code'], inplace=True)
    dataset_test = Dataset.from_pandas(test_df)
    return dataset_test

def translate_text(text, lang='zh-CN'):
  mid = GoogleTranslator(source='en', target=lang).translate(text)
  return GoogleTranslator(source=lang, target='en').translate(mid)

def augment_data(df):
   df['text'] = df['text'].apply(translate_text)
   df.to_csv('augmented_df.csv', index=False)
   return df

def get_augmented_data_sample(n_samples, random_state=42):
    df = pd.DataFrame()
    dataset = "/vol/bitbucket/rm521/cw/dontpatronizeme_pcl.tsv"
    pcl_df = pd.read_csv(dataset, sep="\t", skiprows=lambda x: x in range(2), header=0, names=['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label'])
    pcl_df.set_index('par_id', inplace=True)

    for f in os.listdir("/vol/bitbucket/rm521/cw/augmented"):
        if not os.path.isfile(os.path.join("/vol/bitbucket/rm521/cw/augmented", f)):
           continue
        labels_df = pd.read_csv(os.path.join("/vol/bitbucket/rm521/cw/augmented", f))
        labels_df.set_index('par_id', inplace=True)
        augmented_df = merge_with_labels(labels_df, pcl_df)
        df = pd.concat([df, augmented_df])
    if n_samples > len(df):
        n_samples = len(df)
    return df.sample(n=n_samples, random_state=random_state)

def downsample(df, downsample_ratio, upsample_ratio):
    zero_labels = df[df['label'] == 0]
    one_labels = df[df['label'] == 1]
    one_count = len(one_labels)
    zero_target = round(one_count * downsample_ratio)
    augmented_ones_count = one_count * upsample_ratio - one_count
    augmented_ones = get_augmented_data_sample(int(augmented_ones_count), random_state=42)
    print(f"0: {zero_target}, 1: {len(one_labels)}, A: {len(augmented_ones)} 1+A: {len(one_labels) + len(augmented_ones)}")
    df = pd.concat([one_labels, augmented_ones, zero_labels.sample(zero_target, random_state=42)])
    df = df.sample(frac=1, random_state=42)
    return df

def split_train_valid(df):
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)
    return train_df, valid_df

if __name__ == "__main__":
    dev_df, train_df = pre_process()
    downsample_ratio = 3
    upsample_ratio = 2.5
    
    train_df = downsample(train_df, downsample_ratio, upsample_ratio)

    train_df, valid_df = split_train_valid(train_df)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    dev_test_dataset = Dataset.from_pandas(dev_df)
    
    test_dataset = parse_test()

    bbt = deberta_trainer.DebertaTrainer(train_dataset, valid_dataset, dev_test_dataset, test_dataset)
    # bbt.load_best()
    
    # bbt.train()
    # bbt.predict(save_model=False)
    # bbt.test_predict(test_dataset)

    results = bbt.hyperparameter_search()
    print(results)
