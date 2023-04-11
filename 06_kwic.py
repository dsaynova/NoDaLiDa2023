import pickle
import argparse


def print_latex(party, terms, data, labels):
    for term in terms:
        print('\\noindent --------------------------------------------------------------------------------------------------------------------------------------------------------')
        print()
        print("\\noindent \\textbf{Term}: ", term)
        print()
        print('\\noindent --------------------------------------------------------------------------------------------------------------------------------------------------------')
        print()
        context_count = 1
        for ind, (text, p) in enumerate(zip(data, labels)):
            if p == party:
                word_list = text.replace('###', '---').replace('&&&', '---').lower().split()
                if term in word_list:
                    indices = [i for i, x in enumerate(word_list) if x == term]
                    for i in indices:
                        low, high = max(0, i-20), min(len(word_list), i+20)
                        print("\\noindent Context", context_count, ":")
                        context_count += 1
                        print()
                        print('\\noindent --------------------------------------------------------------------------------------------------------------------------------------------------------')
                        print()
                        print("\\noindent ", " ".join(word_list[low:high]))
                        print()
                        print('\\noindent --------------------------------------------------------------------------------------------------------------------------------------------------------')
                        print()


def print_plain(party, terms, data, labels):
    for term in terms:
        print("-------------")
        print("-------------")
        print("Term: ", term)
        context_count = 1
        for ind, (text, p) in enumerate(zip(data, labels)):
            if p == party:
                word_list = text.replace('###', '---').replace('&&&', '---').lower().split()
                if term in word_list:
                    indices = [i for i, x in enumerate(word_list) if x == term]
                    for i in indices:
                        low, high = max(0, i-20), min(len(word_list), i+20)
                        print("-------------")
                        print("Context", context_count, ":")
                        context_count += 1
                        print(" ".join(word_list[low:high]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--party', type=str, choices=['s', 'm'], nargs=1,
                        required=True)
    parser.add_argument('--sorted_by', type=str, choices=['pca', 'norm'], nargs=1,
                        required=False)
    parser.add_argument('--terms', nargs="*", required=False)
    parser.add_argument('--format', type=str, choices=['latex', 'plain'], nargs=1, required=True)
    args = parser.parse_args()

    with open("data/cased_data_rs_50_m_s.pkl", "rb") as f:
        _, _, x_val, _, _, y_val = pickle.load(f)

    m_pca = ["utgiftsområde", "budgetpropositionen", "jobbskatteavdrag", "arbetslöshetsförsäkringen", "skattehöjningar"]
    m_norm = ["vänsterregering", "fattigdomsbekämpning", "bidragsberoende", "fridens", "arbetsföra"]
    s_pca = ["budgetpropositionen", "arbetsmarknadspolitik", "samlingspartiet", "ungdomsarbetslösheten",
             "skattesänkningar"]
    s_norm = ["överläggningen", "moderatledda", "kd", "skattesänkningarna", "borgarna"]

    dict_word_lists = {"m_pca": m_pca, "m_norm": m_norm, "s_pca": s_pca, "s_norm": s_norm}
    kw = []
    if args.sorted_by is not None:
        kw = dict_word_lists[args.party[0]+"_"+args.sorted_by[0]]
    elif args.terms is not None:
        kw = args.terms
    else:
        parser.error("Provide either list of terms or choose sorted_by option.")

    if args.format[0] == "latex":
        print_latex(args.party[0], kw, x_val, y_val)
    elif args.format[0] == "plain":
        print_plain(args.party[0], kw, x_val, y_val)


if __name__ == "__main__":
    main()
