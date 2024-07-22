import pandas as pd
from sklearn.model_selection import StratifiedKFold
import ast
import argparse

parser = argparse.ArgumentParser(description='Trainer.')
parser.add_argument('--dataset', default="founta18",
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
    "founta18": lambda k, i:  f"{OUT_FOLDER}dataset_founta18_topics_no_other_train_{k}_topic_{i}_of.tsv",
    "toraman22": lambda k, i: f"{OUT_FOLDER}EN_toraman22_reduced_train_{k}_topic_2_{i}_of.tsv",
    "waseem16": lambda k, i: f"{OUT_FOLDER}dataset_waseem16_topics_no_other_binary_train_{k}_topic_{i}_of.tsv",
    "davidson17": lambda k, i: f"{OUT_FOLDER}dataset_davidson17_topics_no_other_train_{k}_topic_{i}_of.tsv",
    "IHSC": lambda k, i: f"{OUT_FOLDER}dataset_IHSC_topics_no_other_train_{k}_topic_{i}_of.tsv",
    "toraman22_tr": lambda k, i: f"{OUT_FOLDER}TR_toraman22_reduced_train_{k}_topic_{i}_of.tsv",
    "toraman22v2_tr": lambda k, i: f"{OUT_FOLDER}TR_toraman22v2_reduced_train_{k}_topic_{i}_of.tsv",

}

for k in range(5):
    df = pd.read_csv(DATASET[args.dataset](k), sep="\t", lineterminator="\n", dtype=str)
    df["topic"] = df.topic.map(ast.literal_eval)

    l = []
    for i in df.topic:
        l += i

    def func2(x):
        def func3(y):
            if x in y:
                return 1
            else: return 0

        return func3

    for i, j in enumerate(set(l)):
        f = func2(j)
        df2 = df.copy()
        df2["topic"] = df.topic.map(f)
        split = df[df2.topic == 1].reset_index(drop=True)
        split.to_csv(OUT_DATASET[args.dataset](k,i), sep="\t", index=False)