import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import pickle
from sklearn.decomposition import PCA
import csv
import re
from collections import Counter, defaultdict


class_names = ["s", "m"]


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, indx):
        item = {key: torch.tensor(val[indx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[indx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_lime_explanations(tokenizer, test_trainer, x_val):
    # define prediction function
    def predict_proba(instance):
        x_test_tokenized = tokenizer(instance, padding=True, truncation=True, max_length=512)
        test_dataset = Dataset(x_test_tokenized)
        raw_prediction, _, _ = test_trainer.predict(test_dataset)
        return np.apply_along_axis(softmax, 1, raw_prediction)

    num_features = 20
    list_of_features = []
    predictions = []
    for idx in range(len(x_val)):
        example = x_val[idx]
        a = tokenizer.decode(tokenizer.encode(example, padding=True, truncation=True, max_length=512),
                             skip_special_tokens=True)
        explainer = LimeTextExplainer(class_names=class_names, random_state=5)
        exp = explainer.explain_instance(a, classifier_fn=predict_proba, num_features=num_features, num_samples=5000)
        predictions.append(np.argmax(predict_proba([example])))
        list_of_features.append(exp.as_list())

    # Write intermediate results
    with open("data/LIME_val_cased_data_rs_50_m_s.pkl", "wb") as f:
        pickle.dump((predictions, list_of_features), f)

    return predictions, list_of_features


def aggregate_list_per_party(list_of_features, predictions, y_val):
    correctness = np.equal([class_names[i] for i in y_val], predictions)
    m_correct, s_correct = defaultdict(list), defaultdict(list)
    for i, filter_correct in zip(list_of_features, correctness):
        if filter_correct:
            for j, k in i[0:10]:
                if k > 0:
                    m_correct[j.lower()].append(k)
                else:
                    s_correct[j.lower()].append(-k)
    # print(len(m_correct), len(s_correct), len(set(m_correct.keys()).union(set(s_correct.keys()))))
    return m_correct, s_correct


def get_ext_embeddings(m_correct, s_correct):
    embedding_name = "data/model.txt"
    embeddings = defaultdict()
    with open(embedding_name, encoding="utf8", errors="ignore") as f:
        next(f)
        for o in f:
            token, *vector = o.strip(" \n").split(" ")
            if token in set(m_correct.keys()).union(set(s_correct.keys())):
                embeddings[token] = [eval(i) for i in vector]
    len(embeddings)

    m_token_list = [i for i in m_correct.keys() if i in embeddings.keys()]
    s_token_list = [i for i in s_correct.keys() if i in embeddings.keys()]
    m_embed = [embeddings[i] for i in m_token_list]
    s_embed = [embeddings[i] for i in s_token_list]
    return m_embed, s_embed, m_token_list, s_token_list


def calc_pca(word_vectors, num_components=2):
    pca = PCA(random_state=5).fit(word_vectors)
    term_coordinates = pca.transform(word_vectors)
    variance = pca.explained_variance_ratio_[0:num_components]
    return term_coordinates, variance


def write_stats_to_file(term_coordinates, variance, words, lime_scores, tf, df, file, num_components=2):

    with open("data/" + file, "w", newline="") as csvfile:
        cvs_writer = csv.writer(csvfile, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        cvs_writer.writerow(["variance", "", "", "", "", "", ""] + list(variance))
        cvs_writer.writerow(["", "lime_mean", "lime_sum", "normalized_lime", "lime_len", "doc_freq", "tot_freq"])
        for j in range(len(words)):
            cvs_writer.writerow([words[j]]
                                + [np.mean(lime_scores[words[j]])
                                    , np.sum(lime_scores[words[j]])
                                    , np.sum(lime_scores[words[j]]) / tf[words[j]]
                                    , len(lime_scores[words[j]])
                                    , df[words[j]]
                                    , tf[words[j]]]
                                + list(term_coordinates[j, 0:num_components])
                                )
        for word in lime_scores.keys():
            if word not in words:
                cvs_writer.writerow([word]
                                    + [np.mean(lime_scores[word])
                                        , np.sum(lime_scores[word])
                                        , np.sum(lime_scores[word]) / tf[word]
                                        , len(lime_scores[word])
                                        , df[word]
                                        , tf[word]]
                                    )


def main():

    # LOAD MODEL AND DATA
    token_model = "KB/bert-base-swedish-cased"
    tokenizer = BertTokenizer.from_pretrained(token_model)
    model_path = "models/cased_output_50/checkpoint-6000"
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    with open("data/cased_data_rs_50_m_s.pkl", "rb") as f:
        _, _, x_val, _, _, y_val = pickle.load(f)
    test_trainer = Trainer(model, TrainingArguments(output_dir="lime/", per_device_eval_batch_size=48, disable_tqdm=True))

    # STEP 1 - Instance explanation extraction
    predictions, list_of_features = get_lime_explanations(tokenizer, test_trainer, x_val)

    # STEP 2 - Aggregation
    m_correct, s_correct = aggregate_list_per_party(list_of_features, predictions, y_val)

    # STEP 3 - Sorting
    # calculate term frequency for normalization
    term_freq, doc_freq = Counter(), Counter()
    for i in x_val:
        tokens = re.split(r"[\s\-]", i.lower())
        term_freq.update(tokens)
        doc_freq.update(set(tokens))
    # calculate pca
    m_embed, s_embed, m_token_list, s_token_list = get_ext_embeddings(m_correct, s_correct)
    n_comp = 10
    term_coordinates_s, variance_s = calc_pca(s_embed, num_components=n_comp)
    term_coordinates_m, variance_m = calc_pca(m_embed, num_components=n_comp)

    # WRITE RESULTS
    write_stats_to_file(term_coordinates_s, variance_s, s_token_list, s_correct, term_freq, doc_freq, file="SocialDemocrats_10_all.csv", num_components=n_comp)
    write_stats_to_file(term_coordinates_m, variance_m, m_token_list, m_correct, term_freq, doc_freq, file="Moderates_10_all.csv", num_components=n_comp)


if __name__ == "__main__":
    main()
