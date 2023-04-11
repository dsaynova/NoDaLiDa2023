import pickle
import random
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(5)


def undersample_pd(d_frame, parties):
    size_per_party = d_frame.party.value_counts()
    size_to_sample = min(size_per_party.to_numpy())
    # print("Undersampling to size: {0}".format(size_to_sample))

    undersampled_data = pd.DataFrame()
    for p in sorted(parties):
        to_append = d_frame[d_frame.party == p].sample(size_to_sample, random_state=5)
        undersampled_data = pd.concat([undersampled_data, to_append], axis=0)

    return undersampled_data


def save_annotation_sample(keys, parties):
    file_name = "data/cased_annotation_sample_" + "_".join("{}".format(p) for p in sorted(parties))

    txt = []
    for p in sorted(parties):
        with open("data/text_nnnp_cased_{0}.pkl".format(p), "rb") as f:
            a = [[p, i, t] for i, t in pickle.load(f).items() if i in keys]
            txt.extend(a)

    pd_data = pd.DataFrame(txt, columns=['party', 'id', 'text']).sample(frac=1, random_state=5)
    pd_data.to_csv(file_name + '.csv')

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parties', type=str, choices=['s', 'm', 'v', 'c', 'mp', 'kd', 'sd', 'l', 'nyd'], nargs='+',
                        required=True)
    parser.add_argument('--undersample', type=bool, default=True)
    parser.add_argument('--remove_short', type=int, default=0)
    args = parser.parse_args()

    # select annotation sample ids
    keys = []
    for p in sorted(args.parties):
        with open("data/text_nnnp_cased_{0}.pkl".format(p), "rb") as f:
            text = pickle.load(f)
        keys.extend(random.sample(list(text.keys()), 50))

    # get remaining data
    data = []
    for p in sorted(args.parties):
        with open("data/text_nnnp_cased_{0}.pkl".format(p), "rb") as f:
            text = pickle.load(f)
        for k in text.keys():
            if k not in keys and len(text[k].split()) > args.remove_short:
                data.append([p, k, text[k]])
    pd_data = pd.DataFrame(data, columns=['party', 'id', 'text'])

    if args.undersample:
        pd_data = undersample_pd(pd_data, args.parties)

    x = pd_data.text.to_numpy()
    y = pd_data.party.to_numpy()

    x_tt, x_val, y_tt, y_val = train_test_split(x, y, stratify=y, test_size=1000 * len(args.parties), random_state=5)
    x_train, x_test, y_train, y_test = train_test_split(x_tt, y_tt, stratify=y_tt, test_size=0.1, random_state=5)

    print(len(x_val), len(x_train), len(x_test))
    # rs0: 2000 109998 12222
    # rs50: 2000 108169 12019

    # save to file
    save_annotation_sample(keys, args.parties)

    with open("data/cased_data_" + "rs_" + str(args.remove_short) + "_" + "_".join(
            "{}".format(p) for p in sorted(args.parties)) + ".pkl", "wb") as f:
        pickle.dump((x_train, x_test, x_val, y_train, y_test, y_val), f)


if __name__ == "__main__":
    main()
