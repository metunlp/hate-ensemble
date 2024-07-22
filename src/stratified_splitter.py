import pandas as pd
from sklearn.model_selection import StratifiedKFold
import ast
import random, argparse

parser = argparse.ArgumentParser(description='Trainer.')
parser.add_argument('--dataset', default="founta18",
                    help='dataset')
parser.add_argument('--N', default="5",
                    help='dataset')
args = parser.parse_args()

OUT_FOLDER = f"./{args.dataset}/"

DATASET = {
    "founta18": lambda k:  f"{OUT_FOLDER}dataset_founta18_topics_no_other_train_{k}.tsv",
    "toraman22": lambda k: f"{OUT_FOLDER}EN_toraman22_reduced_train_{k}.tsv",
    "waseem16": lambda k: f"{OUT_FOLDER}dataset_waseem16_topics_no_other_binary_train_{k}.tsv",
    "davidson17": lambda k: f"{OUT_FOLDER}dataset_davidson17_topics_no_other_train_{k}.tsv",
    "IHSC": lambda k: f"{OUT_FOLDER}dataset_IHSC_topics_no_other_train_{k}.tsv",
    "toraman22_tr": lambda k: f"{OUT_FOLDER}TR_toraman22_reduced_train_{k}.tsv",
    "toraman22v2_tr": lambda k: f"{OUT_FOLDER}TR_toraman22_reduced_train_{k}.tsv",
}
OUT_DATASET= {
    "founta18": lambda k, i:  f"{OUT_FOLDER}dataset_founta18_topics_no_other_train_{k}_fold_{i}_of_{args.N}.tsv",
    "toraman22": lambda k, i: f"{OUT_FOLDER}EN_toraman22_reduced_train_{k}_fold_{i}_of_{args.N}.tsv",
    "waseem16": lambda k, i: f"{OUT_FOLDER}dataset_waseem16_topics_no_other_binary_train_{k}_fold_{i}_of_{args.N}.tsv",
    "davidson17": lambda k, i: f"{OUT_FOLDER}dataset_davidson17_topics_no_other_train_{k}_fold_{i}_of_{args.N}.tsv",
    "IHSC": lambda k, i: f"{OUT_FOLDER}dataset_IHSC_topics_no_other_train_{k}_fold_{i}_of_{args.N}.tsv",
    "toraman22_tr": lambda k, i: f"{OUT_FOLDER}TR_toraman22_reduced_train_{k}_fold_{i}_of_{args.N}.tsv",
    "toraman22v2_tr": lambda k, i: f"{OUT_FOLDER}TR_toraman22v2_reduced_train_{k}_fold_{i}_of_{args.N}.tsv",

}
random.seed(1000)

LABEL2ID = {"normal": 0, "abusive": 1, "spam": 2, "hateful": 3}
if args.dataset == "founta18":
    LABELING_DICT = {"normal": 0, "abusive": 1, "spam": 2, "hateful": 3}
    NUM_CLASS = 4
elif "toraman22" in args.dataset: 
    LABELING_DICT = {"0": 0, "1": 1, "2": 2}
    NUM_CLASS = 3
    COUNT = 5
elif args.dataset == "waseem16": 
    LABELING_DICT = {"none": 0, "hateful": 1}
    NUM_CLASS = 2
elif args.dataset == "davidson17": 
    LABELING_DICT = {"0": 0, "1": 1, "2": 2}
    NUM_CLASS = 3
elif args.dataset == "IHSC": 
    LABELING_DICT = {"no": 0, "yes": 1}
    NUM_CLASS = 2
    MODEL = "bert-base-italian-uncased"

for k in range(5):
    df = pd.read_csv(DATASET[args.dataset](k), sep="\t", lineterminator="\n", dtype=str)
    df["topic"] = df.topic.map(ast.literal_eval)
    X = list(range(len(df)))
    Y = df.label.map(LABELING_DICT)

    skf = StratifiedKFold(n_splits=int(args.N))
    for i, (train_idx, valid_idx) in enumerate(skf.split(X,Y)):
        split = df.iloc[train_idx]
        split.to_csv(OUT_DATASET[args.dataset](k, i), sep="\t", index=False)
        #split = df.iloc[valid_idx]
        #split.to_csv(f"dataset_founta18_topics_no_other_valid_0_{i}.tsv", sep="\t", index=False)