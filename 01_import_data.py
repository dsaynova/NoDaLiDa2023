import xml.etree.ElementTree as elementTree
from collections import defaultdict
import pickle


# {'s': 104842, 'm': 62160, 'v': 30934, 'fp': 29502, 'c': 29498, 'mp': 27836,
# 'kd': 24540, 'sd': 6981, 'kds': 2572, '': 2432, 'l': 2236, 'nyd': 1075, '-': 319,
# 'talmannen': 93, 'förste vice talmannen': 63, 'tredje vice talmannen': 58,
# 'andre vice talmannen': 56, 'hans majestät konungen': 3, 'ålderspresidenten': 1, 'tjänstgörande ålderspresidenten': 1}

def main():
    temp_text = []

    pol_spec = set()
    lemma_list = ['allians', 'moderat', 'borgerli', 'folkpart', 'socialdem', 'moderat', 'vänsterpar', 'centerpar',
                  'miljöpar', 'kristdemokrat', 'sverigedemokrat', 'liberal']
    party_mapping = {}
    for x in ['s', 'm', 'v', 'c', 'mp', 'kd', 'sd', 'l', 'nyd']:
        party_mapping["{0}".format(x)] = defaultdict(str)
    for event, elem in elementTree.iterparse("data/rd-anf.xml", events=("end",)):
        if event == "end":
            if elem.tag == 'w':
                # replace names
                if elem.attrib['pos'] == 'PM':
                    temp_text.append('&&&')
                # replace political references
                elif 'swefn' in list(elem.attrib.keys()) \
                        and elem.attrib['swefn'] == '|People_along_political_spectrum|':
                    pol_spec.add(elem.attrib['lemma'])
                    temp_text.append('###')
                elif any(item.lower() in elem.attrib['lemma'] for item in lemma_list):
                    pol_spec.add(elem.attrib['lemma'])
                    temp_text.append('###')
                else:
                    temp_text.append(elem.text)
                elem.clear()
            if elem.tag == 'text':
                party = elem.attrib['parti'].lower().replace('kds', 'kd').replace('fp', 'l')
                if party in party_mapping.keys():
                    party_mapping[party][elem.attrib['anforande_id']] = " ".join(temp_text)
                temp_text = []
                elem.clear()

    for key in party_mapping.keys():
        with open("data/text_nnnp_cased_{0}.pkl".format(key), "wb") as f:
            pickle.dump(party_mapping[key], f)


if __name__ == "__main__":
    main()
