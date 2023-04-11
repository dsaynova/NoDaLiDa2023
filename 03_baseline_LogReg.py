import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import f1_score


def main():
    # Read data
    with open("data/cased_data_rs_50_m_s.pkl", "rb") as f:
        raw_x_train, raw_x_test, _, raw_y_train, raw_y_test, _ = pickle.load(f)

    # print(len(raw_X_train),len(raw_X_test),len(raw_X_val))
    # 108169 12019 2000

    input_dim = 50325  # vocab size
    m_iter = 500
    p_dict = {'s': -1, 'm': 1}

    combined_vocab = []
    for i in range(1, 4):
        n_range = (i, i)

        # training data tfidf
        tfidf = TfidfVectorizer(sublinear_tf=True, max_features=input_dim, ngram_range=n_range)
        features = tfidf.fit_transform(raw_x_train).toarray()
        labels = [p_dict[i] for i in raw_y_train]
        combined_vocab.extend(list(tfidf.vocabulary_.keys()))

        # test data tfidf
        features_test = tfidf.transform(raw_x_test).toarray()
        labels_test = [p_dict[i] for i in raw_y_test]

        # train and evaluate model
        clf = linear_model.LogisticRegression(random_state=5, max_iter=m_iter).fit(features, labels)
        acc = clf.score(features_test, labels_test)
        f1 = f1_score(labels_test, clf.predict(features_test))
        print(n_range, round(acc * 100, 2), round(f1 * 100, 2))

        if i == 1:
            # for unigrams - extract most informative features
            clf_unigram = clf
            id_to_gram = {i: k for k, i in tfidf.vocabulary_.items()}
            m_index = sorted(range(len(clf.coef_[0])), key=lambda x: clf.coef_[0][x])[-20:]
            m_index.reverse()
            s_index = sorted(range(len(clf.coef_[0])), key=lambda x: clf.coef_[0][x])[:20]

    # model trained on most frequent unigrams, bigrams and trigrams combined
    tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=combined_vocab)
    features = tfidf.fit_transform(raw_x_train).toarray()
    labels = [p_dict[i] for i in raw_y_train]

    features_test = tfidf.transform(raw_x_test).toarray()
    labels_test = [p_dict[i] for i in raw_y_test]

    clf = linear_model.LogisticRegression(random_state=5, max_iter=m_iter).fit(features, labels)
    acc = clf.score(features_test, labels_test)
    f1 = f1_score(labels_test, clf.predict(features_test))
    print((1, 3), round(acc * 100, 2), round(f1 * 100, 2))
    print()

    # print most informative features
    print("Social Democrats")
    for j in s_index:
        print(id_to_gram[j], clf_unigram.coef_[0][j])
    print("Moderates")
    for i in m_index:
        print(id_to_gram[i], clf_unigram.coef_[0][i])


if __name__ == "__main__":
    main()
