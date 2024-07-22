import pandas as pd
import numpy as np
# with urls
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import random, argparse
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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
    "toraman22v2": lambda k: f"{OUT_FOLDER}EN_toraman22v2_reduced_train_{k}.tsv",
    "toraman22_tr": lambda k: f"{OUT_FOLDER}TR_toraman22_reduced_train_{k}.tsv",
    "toraman22v2_tr": lambda k: f"{OUT_FOLDER}TR_toraman22v2_reduced_train_{k}.tsv",
    "waseem16": lambda k: f"{OUT_FOLDER}dataset_waseem16_topics_no_other_binary_train_{k}.tsv",
    "davidson17": lambda k: f"{OUT_FOLDER}dataset_davidson17_topics_no_other_train_{k}.tsv",
    "IHSC": lambda k: f"{OUT_FOLDER}dataset_IHSC_topics_no_other_train_{k}.tsv",
}
OUT_DATASET= {
    "founta18": lambda k, i:  f"{OUT_FOLDER}dataset_founta18_topics_no_other_train_{k}_GMM_{i}_of_{args.N}.tsv",
    "toraman22": lambda k, i: f"{OUT_FOLDER}EN_toraman22_reduced_train_{k}_GMM_{i}_of_{args.N}.tsv",
    "toraman22v2": lambda k, i: f"{OUT_FOLDER}EN_toraman22v2_reduced_train_{k}_GMM_{i}_of_{args.N}.tsv",
    "toraman22_tr": lambda k, i: f"{OUT_FOLDER}TR_toraman22_reduced_train_{k}_GMM_{i}_of_{args.N}.tsv",
    "toraman22v2_tr": lambda k, i: f"{OUT_FOLDER}TR_toraman22v2_reduced_train_{k}_GMM_{i}_of_{args.N}.tsv",

    "waseem16": lambda k, i: f"{OUT_FOLDER}dataset_waseem16_topics_no_other_binary_train_{k}_GMM_{i}_of_{args.N}.tsv",
    "davidson17": lambda k, i: f"{OUT_FOLDER}dataset_davidson17_topics_no_other_train_{k}_GMM_{i}_of_{args.N}.tsv",
    "IHSC": lambda k, i: f"{OUT_FOLDER}dataset_IHSC_topics_no_other_train_{k}_GMM_{i}_of_{args.N}.tsv",
}
if "tr" in args.dataset:
    MODEL = "bert-base-turkish-uncased"
elif args.dataset == "IHSC":
    MODEL = "bert-base-italian-uncased"
else:
    MODEL = "distilbert-base-uncased"
    
random.seed(1000)
os.system("source /etc/environment")
for k in range(5):
    dataset = df = pd.read_csv(DATASET[args.dataset](k), sep="\t", lineterminator="\n", dtype=str)
    # Import our models. The package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained(MODEL, do_lower_case=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to("cuda:0")

    embeddings = []
    predictions = []
    for tweet in tqdm(dataset["text"]):
        inputs = tokenizer(tweet, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            result = model(**inputs, output_hidden_states=True, return_dict=True)
            predictions.append(torch.argmax(result.get('logits')).detach().cpu().numpy())
            embeddings.append(result.hidden_states[-1][:,0,:].detach().cpu().numpy())
    embeddings = np.asarray(embeddings).squeeze()
    predictions = np.asarray(predictions)

    N = int(args.N)

    def get_embeds_v2(embed_file, df_orig, df):
        embeddings = []
        ls = list(df["text"])
        for i, row in df_orig.iterrows():
            if str(row["text"]) in ls:
                embeddings.append(embed_file[i,:])

        return np.asarray(embeddings)

    df = dataset
    embeddings_temp = get_embeds_v2(embeddings, dataset, df)
    estimators = {cov_type: GaussianMixture(n_components=N,
                    covariance_type=cov_type, max_iter=150, random_state=0)
                    for cov_type in ['full']} #'spherical', 'diag', 'tied', 

    for name, estimator in estimators.items():
        predictions = estimator.fit_predict(embeddings_temp)
        for n in range(N):
            temp_df = df.iloc[predictions == n].reset_index(drop=True).to_csv(OUT_DATASET[args.dataset](k, n), sep="\t")
