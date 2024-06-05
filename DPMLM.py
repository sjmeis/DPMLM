import os
import torch
import nltk
import json
import string
import numpy as np
from collections import Counter
import wn
nltk.download('words', quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, logging
import importlib_resources as impresources

en = wn.Wordnet('oewn:2022') 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.set_verbosity_warning()

stop = set([x for x in stopwords.words("english")])

def nth_repl(s, sub, repl, n):
    s_split = s.split()
    i = 0
    try:
        find = s_split.index(sub)
        i += 1
    except ValueError:
        return s
    
    while i != n:
        try:
            find = s_split.index(sub, find + 1)
            i += 1
        except ValueError:
            break
    if i == n:
        return " ".join(s_split[:find] + [repl] + s_split[find+1:])
    return s

def nth_rem(s, sub, n):
    s_split = s.split()
    i = 0
    try:
        find = s_split.index(sub)
        i += 1
    except ValueError:
        return s
    
    while i != n:
        try:
            find = s_split.index(sub, find + 1)
            i += 1
        except ValueError:
            break
    if i == n:
        return " ".join(s_split[:find] + s_split[find+1:])
    return s

def sentence_enum(tokens):
    counts = Counter()
    n = []
    for t in tokens:
        counts[t] += 1
        n.append(counts[t])
    return n

def get_opposites():
	with open(impresources.files("DPMLM") / "data" / "opposites.json", 'r') as f:
		opposites = json.load(f)
	return opposites

def get_vocab():
	with open(impresources.files("DPMLM") / "data" / "vocab.txt", 'r') as f:
		vocab = set([x.strip() for x in f.readlines()])
	return vocab

def get_antonyms(word):
    ants = list()

    #Get antonyms from WordNet for this word and any of its synonyms.
    for ss in en.synsets(word):
        for sense in ss.senses():
            ants.extend([x.word().lemma() for x in sense.get_related("antonym")])

    #Get snyonyms of antonyms found in the previous step, thus expanding the list even more.
    syns = list()
    for word in list(set(ants)):
        for ss in en.synsets(word):
            syns.extend(ss.lemmas())

    return sorted(list(set(syns)))

'''
Gets pertainyms of the target word from the WordNet knowledge base.
* pertainyms = words pertaining to the target word (industrial -> pertainym is "industry")
'''
def get_pertainyms(word):
    perts = list()
    for ss in en.synsets(word):
        for sense in ss.senses():
            perts.extend([x.word().lemma() for x in sense.get_related("pertainym")])
    return sorted(list(set(perts)))
'''
Get hyponyms (new wn)
'''
def get_hyponyms(word):
    hypo = list()
    for ss in en.synsets(word):
        for sense in ss.senses():
            hypo.extend([x.word().lemma() for x in sense.get_related("hyponyms")])
    return sorted(list(set(hypo)))

'''
Get hypernyms (new wn)
'''
def get_hypernyms(word):
    hyper = list()
    for ss in en.synsets(word):
        for h in ss.hypernyms():
            hyper.extend([x.lemma() for x in h.words()])
    return sorted(list(set(hyper)))

'''
Gets derivationally related forms (e.g. begin -> 'beginner', 'beginning')
'''
def get_related_forms(word):
    forms = list()
    for ss in wn.synsets(word):
        for sense in ss.senses():
            forms.extend([x.word().lemma() for x in sense.get_related("derivation")]) 
    return sorted(list(set(forms)))

'''
General get nym
'''
def get_general_nym(word, nym):
    n = list()
    for ss in wn.synsets(word):
        for sense in ss.senses():
            n.extend([x.word().lemma() for x in sense.get_related(nym)]) 
    return sorted(list(set(n)))

'''
Gets antonyms, hypernyms, hyponyms, holonyms, meronyms, pertainyms, and derivationally related forms of a target word from WordNet.
* hypernym = a word whose meaning includes a group of other words ("animal" is a hypernym of "dog")
* hyponym = a word whose meaning is included in the meaning of another word ("bulldog" is a hyponym of "dog")
* a meronym denotes a part and a holonym denotes a whole: "week" is a holonym of "weekend", "eye" is a meronym of "face", and vice-versa
'''
def get_nyms(word, depth=-1):
    nym_list = ['antonyms', 'hypernyms', 'hyponyms', 'holonyms', 'meronyms', 
                'pertainyms', 'derivationally_related_forms']
    results = list()
    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word)

    def query_wordnet(getter):
        res = list()
        for ss in en.synsets(word):
            res_list = [item.lemmas() for item in ss.closure(getter)]
            res_list = [item.name() for sublist in res_list for item in sublist]
            res.extend(res_list)
        return res

    for nym in nym_list:
        if nym=='antonyms':
            results.append(get_antonyms(word))

        elif nym == "hypernyms":
            results.append(get_hypernyms(word))

        elif nym == "hyponyms":
            results.append(get_hyponyms(word))

        elif nym in ['holonyms', 'meronyms']:
            res = list()
            #Three different types of holonyms and meronyms as defined in WordNet
            for postfix in ["_member", "_part", "_portion", "_substance"]:
                res.extend(get_general_nym(word, "{}{}".format(nym[:4], postfix)))
            results.append(res)

        elif nym=='pertainyms':
            results.append(get_pertainyms(word))

        else:
            results.append(get_related_forms(word))

    results = map(set, results)
    nyms = dict(zip(nym_list, results))
    return nyms

#Converts a part-of-speech tag returned by NLTK to a POS tag from WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

#Function for clearing up duplicate words (capitalized, upper-case, etc.), stop words, and antonyms from the list of candidates.
def filter_words(target, words, scr, tkn, opp={}):
    dels = list()
    toks = tkn.tolist()
    nyms = get_nyms(target)
    lemmatizer = WordNetLemmatizer()

    if lemmatizer.lemmatize(target.lower()) in opp:
        opp_del = [x for x in words if lemmatizer.lemmatize(x.lower()) in opp[lemmatizer.lemmatize(target.lower())]]
        dels.extend(opp_del)

    for w in words:
        if w.lower() in words and w.upper() in words:
            dels.append(w.upper())
        if lemmatizer.lemmatize(w.lower()) in nyms['antonyms']:
            dels.append(w)

    dels = list(set(dels))
    for d in dels:
        del scr[words.index(d)]
        del toks[words.index(d)]
        words.remove(d)

    return words, scr, torch.tensor(toks)

#Calculates the similarity score
def similarity_score(original_output, subst_output, k):
    mask_idx = k
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    weights = torch.div(torch.stack(list(original_output[3])).squeeze().sum(0).sum(0), (12 * 12.0))

    suma = 0.0
    sent_len = original_output[2][2].shape[1]

    for token_idx in range(sent_len):     
        original_hidden = original_output[2]
        subst_hidden = subst_output[2]

        #Calculate the contextualized representation of the i-th word as a concatenation of RoBERTa's values in its last four layers
        context_original = torch.cat( tuple( [original_hidden[hs_idx][:, token_idx, :] for hs_idx in [1, 2, 3, 4]] ), dim=1)
        context_subst = torch.cat( tuple( [subst_hidden[hs_idx][:, token_idx, :] for hs_idx in [1, 2, 3, 4]] ), dim=1)
        suma += weights[mask_idx][token_idx] * cos_sim(context_original, context_subst)

    substitute_validation = suma
    return substitute_validation

#Calculates the proposal score
def proposal_score(original_score, subst_scores, device):
    subst_scores = torch.tensor(subst_scores).to(device)
    return np.log( torch.div(subst_scores , (1.0 - original_score)).cpu() )

class DPMLM():
    opposites = get_opposites()
    vocab = get_vocab()
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()
    tokenizer = None
    lm_model = None
    raw_model = None
    device = None
    nlp = None
    alpha = None

    def __init__(self, MODEL="roberta-base", SPACY="en_core_web_md", alpha=0.003):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.lm_model = AutoModelForMaskedLM.from_pretrained(MODEL)
        self.raw_model = AutoModel.from_pretrained(MODEL, output_hidden_states=True, output_attentions=True)
        self.alpha = alpha

        self.clip_min = -3.2093127
        self.clip_max = 16.304797887802124
        self.sensitivity = abs(self.clip_max - self.clip_min)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lm_model = self.lm_model.to(self.device)
        self.raw_model = self.raw_model.to(self.device)

    def load_transformers(self):
        return self.tokenizer, self.lm_model, self.raw_model

    #Calculates the proposal scores, substitute validation scores, and then the final score for each candidate word's fit as a substitution.
    def calc_scores(self, scr, sentences, original_output, original_score, mask_index):
        #Get representations of all substitute sentences
        _, _, raw_model = self.load_transformers()
        subst_output = raw_model(sentences)

        prop_score = proposal_score(original_score, scr, self.device)
        substitute_validation = similarity_score(original_output, subst_output, mask_index)

        final_score = substitute_validation.cpu() + self.alpha*prop_score
        
        return final_score, prop_score, substitute_validation

    def privatize(self, sentence, target, n=1, K=5, CONCAT=True, FILTER=True, POS=False, ENGLISH=False, epsilon=1, MS=None, TEMP=False):
        split_sent = nltk.word_tokenize(sentence)
        original_sent = ' '.join(split_sent)
        #orig_pos = [x.tag_ for x in self.nlp(original_sent)]

        # Masks the target word in the original sentence.
        if MS is None:
            masked_sent = ' '.join(split_sent)
        else:
            masked_sent = MS

        if isinstance(target, list):
            if n == 1:
                n = [1 for _ in range(len(target))]

            for t, nn in zip(target, n):
                masked_sent = nth_repl(masked_sent, t, self.tokenizer.mask_token, nn)
        else:
            masked_sent = nth_repl(masked_sent, target, self.tokenizer.mask_token, n)
            n = [n]

        #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
        if CONCAT == False:
            input_ids = self.tokenizer.encode(" "+masked_sent, add_special_tokens=True)
        else:
            input_ids = self.tokenizer.encode(" "+original_sent.replace("MASK", ""), " "+masked_sent, add_special_tokens=True)
        if isinstance(target, list):
            masked_position = np.where(np.array(input_ids) == self.tokenizer.mask_token_id)[0].tolist()
        else:
            masked_position = [input_ids.index(self.tokenizer.mask_token_id)]
            target = [target]

        original_output = self.raw_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))

        #Get the predictions of the Masked LM transformer.
        with torch.no_grad():
            output = self.lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))
        
        logits = output[0].squeeze().detach().cpu().numpy()

        predictions = {}
        for t, m, nn in zip(target, masked_position, n):
            current = "{}_{}".format(t, nn)

            #Get top guesses: their token IDs, scores, and words.
            mask_logits = logits[m].squeeze()
            if TEMP == True:
                mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
                mask_logits = mask_logits / (2 * self.sensitivity / epsilon)

                logits_idx = [i for i, x in enumerate(mask_logits)]
                scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
                scores = scores / scores.sum()
                chosen_idx = np.random.choice(logits_idx, p=scores.numpy())
                predictions[current] = (self.tokenizer.decode(chosen_idx).strip(), scores[chosen_idx])
                continue
            else:
                top_tokens = torch.topk(torch.from_numpy(mask_logits), k=K, dim=0)[1]
                scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)[top_tokens].tolist()
            words = [self.tokenizer.decode(i.item()).strip() for i in top_tokens]

            if FILTER == True:
                words, scores, top_tokens = filter_words(t, words, scores, top_tokens, self.opposites)

            if len(words) == 0:
                predictions[current] = [(t, 1)]
                continue


            assert len(words) == len(scores)

            if len(words) == 0:
                predictions[current] = [(t, 1)]
                continue

            original_score = torch.softmax(torch.from_numpy(mask_logits), dim=0)[m]
            sentences = list()

            for i in range(len(words)):
                subst_word = top_tokens[i]
                input_ids[m] = int(subst_word)
                sentences.append(list(input_ids))

            torch_sentences = torch.tensor(sentences).to(self.device)

            finals, _, _ = self.calc_scores(scores, torch_sentences, original_output, original_score, m)
            finals = map(lambda f : float(f), finals)

            zipped = dict(zip(words, finals))
            for i in range(len(words)):
                cand = words[i]
                if cand not in zipped:
                    continue
                
                # remove non-words
                if ENGLISH == True:
                    if cand not in self.vocab and self.lemmatizer.lemmatize(cand) not in self.vocab:
                        del zipped[cand]
                        continue

            zipped = dict(zipped)
            finish = list(sorted(zipped.items(), key=lambda item: item[1], reverse=True))[:K]
            predictions[current] = finish

        if TEMP == True:
            for p in predictions:
                predictions[p] = predictions[p][0]

        return predictions
    
    def dpmlm_rewrite(self, sentence, epsilon, REPLACE=False, FILTER=False, STOP=False, TEMP=True, POS=True, CONCAT=True):
        if isinstance(sentence, list):
            tokens = sentence
        else:
            tokens = nltk.word_tokenize(sentence)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))] #epsilon #/ num_tokens
        n = sentence_enum(tokens)
        replace = []
        new_tokens = [str(x) for x in tokens]

        perturbed = 0
        total = 0
        for i, (t, nn, eps) in enumerate(zip(tokens, n, word_eps)):
            if i >= len(tokens):
                break

            if (STOP == False and t in stop) or t in string.punctuation:
                total += 1
                if tokens[i][0].isupper() == True:
                    replace.append(t.capitalize())
                else:
                    replace.append(t)
                continue

            if REPLACE == True:
                new_s = " ".join(new_tokens)
                new_n = sentence_enum(new_tokens)
                res = self.privatize(sentence, t, n=new_n[i], ENGLISH=True, FILTER=FILTER, epsilon=eps, MS=new_s, TEMP=TEMP, POS=POS, CONCAT=CONCAT)
                r = res[t+"_{}".format(new_n[i])]
                new_tokens[i] = r
            else:
                res = self.privatize(sentence, t, n=nn, ENGLISH=True, FILTER=FILTER, epsilon=eps, TEMP=TEMP, POS=POS, CONCAT=CONCAT)
                r = res[t+"_{}".format(nn)]

            if tokens[i][0].isupper() == True:
                replace.append(r.capitalize())
            else:
                replace.append(r.lower())

            if r != t:
                perturbed += 1
            total += 1

        return self.detokenizer.detokenize(replace), perturbed, total
    
    def dpmlm_rewrite_plus(self, sentence, epsilon, FILTER=False, TEMP=True, POS=True, CONCAT=True, ADD_PROB=0.15, DEL_PROB=0.05):
        if isinstance(sentence, list):
            tokens = sentence
        else:
            tokens = nltk.word_tokenize(sentence)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))] #epsilon #/ num_tokens
        n = sentence_enum(tokens)
        replace = []
        new_tokens = [str(x) for x in tokens]

        perturbed = 0
        total = 0
        deleted = 0
        added = 0

        for i, (t, nn, eps) in enumerate(zip(tokens, n, word_eps)):
            if t in string.punctuation:
                total += 1
                replace.append(t)
                continue

            if i == len(tokens) - 1:
                DELETE = 1
            else:
                DELETE = np.random.rand()
            if DELETE >= DEL_PROB:
                new_s = " ".join(new_tokens)
                new_n = sentence_enum(new_tokens)
                res = self.privatize(sentence, t, n=new_n[i+added-deleted], ENGLISH=True, FILTER=FILTER, epsilon=eps, MS=new_s, TEMP=TEMP, POS=POS, CONCAT=CONCAT)
                r = res[t+"_{}".format(new_n[i+added-deleted])]
                if i+added-deleted > len(new_tokens) - 1:
                    new_tokens.insert(i+added-deleted, r)
                else:
                    new_tokens[i+added-deleted] = r
                replace.append(r)

                if r != t:
                    perturbed += 1
                total += 1
            else:
                new_n = sentence_enum(new_tokens)
                temp = nth_rem(" ".join(new_tokens), t, new_n[i+added-deleted])
                new_tokens = [str(x) for x in temp.split()]
                deleted += 1
                continue

            ADD = np.random.rand()
            if ADD <= ADD_PROB:
                tokens_copy = new_tokens.copy()
                tokens_copy.insert(i+1+added-deleted, "MASK")
                new_s = " ".join(tokens_copy)
                new_n = sentence_enum(new_tokens)
                res = self.privatize(sentence, "MASK", n=1, ENGLISH=True, FILTER=FILTER, epsilon=eps, MS=new_s, TEMP=TEMP, POS=POS, CONCAT=CONCAT)
                r = res["MASK_1"]
                new_tokens.insert(i+1+added-deleted, r)
                replace.append(r)
                added += 1

        return self.detokenizer.detokenize(replace), perturbed, total, added, deleted