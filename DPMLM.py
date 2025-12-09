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
import gc

torch.set_float32_matmul_precision('medium')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.set_verbosity_warning()

stop = set([x for x in stopwords.words("english")])

def nth_repl(s, sub, repl, n):
    s_split = nltk.word_tokenize(s)
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

    def __init__(self, MODEL="FacebookAI/roberta-base", SPACY="en_core_web_md", alpha=0.003):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.lm_model = AutoModelForMaskedLM.from_pretrained(MODEL)
        self.raw_model = AutoModel.from_pretrained(MODEL, output_hidden_states=True, output_attentions=True)
        self.alpha = alpha

        # old for roberta-base
        self.clip_min = -3.2093127
        self.clip_max = 16.304797887802124

        # new for ModernBert
        # self.clip_min = -1.207156
        # self.clip_max = 14.831477403640747
        # self.max_idx = max_idx

        self.sensitivity = abs(self.clip_max - self.clip_min)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lm_model = self.lm_model.to(self.device)
        self.raw_model = self.raw_model.to(self.device)

    def load_transformers(self):
        return self.tokenizer, self.lm_model, self.raw_model

    def privatize(self, sentence, target, n, start_index, CONCAT=True, epsilon=1):
        split_sent = nltk.word_tokenize(sentence)
        original_sent = ' '.join(split_sent)

        # Masks the target word in the original sentence.
        masked_sent = ' '.join(split_sent)
        masked_sent = nth_repl(masked_sent, target, self.tokenizer.mask_token, n)
        n = [n]

        encoded = self.tokenizer.encode(masked_sent, add_special_tokens=False)
        lower, upper = self.sliding_window(encoded, start_index, int((self.tokenizer.model_max_length-32)/2))
        masked_sent = self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=False)

        #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
        if CONCAT == False:
            input_ids = self.tokenizer.encode(" "+masked_sent, add_special_tokens=True, truncation=True)
        else:
            input_ids = self.tokenizer.encode(" "+original_sent.replace("MASK", ""), " "+masked_sent, add_special_tokens=True, truncation="only_first")
        masked_position = [input_ids.index(self.tokenizer.mask_token_id)]
        target = [target]

        #original_output = self.raw_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))

        #Get the predictions of the Masked LM transformer.
        with torch.no_grad():
            output = self.lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))
        
        logits = output[0].squeeze().detach().cpu().numpy()

        predictions = {}
        for t, m, nn in zip(target, masked_position, n):
            current = "{}_{}".format(t, nn)

            #Get top guesses: their token IDs, scores, and words.
            mask_logits = logits[m].squeeze()
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / (2 * self.sensitivity / epsilon)

            logits_idx = [i for i, x in enumerate(mask_logits)]
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            chosen_idx = np.random.choice(logits_idx, p=scores.numpy())
            predictions[current] = (self.tokenizer.decode(chosen_idx).strip(), scores[chosen_idx])
        
        for p in predictions:
            predictions[p] = predictions[p][0]

        return predictions
    
    def privatize_batch(self, sentences, targets, n, epsilon, CONCAT=True, STOP=False, batch_size=16):
        predictions = {}
        outputs = []
        masked_position = []

        new_sentences = []
        new_targets = []
        new_n = []
        for t, s, x in zip(targets, sentences, n):
            if (STOP == False and t in stop) or t in string.punctuation:
                predictions["{}_{}".format(t, x)] = t
                continue
            new_sentences.append(s)
            new_targets.append(t)
            new_n.append(x)
        sentences = new_sentences
        targets = new_targets
        n = new_n

        split_size = int(np.ceil(len(sentences) / batch_size))

        start_index = 0
        begin = 0
        end = batch_size
        for idx in range(split_size):
            if end is not None:
                batch = sentences[begin:end]
                targets_batch = targets[begin:end]
                n_batch = n[begin:end]
            else:
                batch = sentences[begin:]
                targets_batch = targets[begin:]
                n_batch = n[begin:]

            begin += batch_size
            if idx == split_size - 2 or idx == split_size - 1:
                end = None
            else:
                end += batch_size

            split_sents = [nltk.word_tokenize(sentence) for sentence in batch]
            original_sents = [' '.join(split_sent) for split_sent in split_sents]

            # Masks the target word in the original sentence.
            masked_sents = [' '.join(split_sent) for split_sent in split_sents]

            for i, (t, nn) in enumerate(zip(targets_batch, n_batch)):
                temp = nth_repl(masked_sents[i], t, self.tokenizer.mask_token, nn)
                encoded = self.tokenizer.encode(temp, add_special_tokens=False)
                lower, upper = self.sliding_window(encoded, start_index, int((self.tokenizer.model_max_length-32)/2))
                masked_sent = self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=False)
                masked_sents[i] = masked_sent
                start_index += 1

            #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
            if CONCAT == False:
                inputs = self.tokenizer(
                    masked_sents,
                    max_length=self.tokenizer.model_max_length-32,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True
                )
            else:
                original_sents = [" "+x.replace("MASK", "") for x in original_sents]
                masked_sents = [" "+x for x in masked_sents]

                inputs = self.tokenizer(
                    text=original_sents,
                    text_pair=masked_sents,
                    max_length=self.tokenizer.model_max_length-32,
                    return_tensors="pt",
                    padding=True,
                    truncation="only_first",
                    add_special_tokens=True
                )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            masked_position.extend((inputs['input_ids'] == self.tokenizer.mask_token_id).tolist())

            #Get the predictions of the Masked LM transformer.
            with torch.no_grad():
                outputs.extend(self.lm_model(**inputs).logits)

            del inputs
                    
        #predictions = {}
        for i in range(len(outputs)):
            current = "{}_{}".format(targets[i], n[i])

            mask_logits = outputs[i][masked_position[i]].squeeze().detach().cpu().numpy()
            if len(mask_logits) == 0:
                predictions[current] = targets[i] # (targets[i], 0)
                continue
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / (2 * self.sensitivity / epsilon[i])

            logits_idx = [j for j, x in enumerate(mask_logits)]
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            chosen_idx = np.random.choice(logits_idx, p=scores.numpy())
            predictions[current] = self.tokenizer.decode(chosen_idx).strip() #(self.tokenizer.decode(chosen_idx).strip(), scores[chosen_idx])

        # for p in predictions:
        #     predictions[p] = predictions[p][0]

        del outputs
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        return predictions
    
    def sliding_window(self, tokens, target_idx, max_len):
        length = len(tokens)
        lower = max(0, int(target_idx-(max_len/2)))
        remaining = max_len - (target_idx - lower)
        upper = min(length, target_idx+remaining)
        return lower, upper

    def dpmlm_rewrite(self, sentence, epsilon, REPLACE=False, STOP=False, CONCAT=True):
        sentence = " ".join(sentence.split("\n"))
        tokens = nltk.word_tokenize(sentence)
        encoded = self.tokenizer.encode(sentence, add_special_tokens=False)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))]
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
                lower, upper = self.sliding_window(new_tokens, i, int((self.tokenizer.model_max_length-32)/2))
                new_s = " ".join(new_tokens[lower:upper])
                new_n = sentence_enum(new_tokens[lower:upper])
                t_sentence = self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=True)
                res = self.privatize(t_sentence, t, n=new_n[i], ENGLISH=True, FILTER=FILTER, epsilon=eps, MS=new_s, TEMP=TEMP, POS=POS, CONCAT=CONCAT)
                r = res[t+"_{}".format(new_n[i])]
                new_tokens[i] = r
            else:
                res = self.privatize(sentence, t, nn, i, epsilon=eps, CONCAT=CONCAT)
                r = res[t+"_{}".format(nn)]

            if tokens[i][0].isupper() == True:
                replace.append(r.capitalize())
            else:
                replace.append(r.lower())

            if r != t:
                perturbed += 1
            total += 1

        return self.detokenizer.detokenize(replace), perturbed, total
    
    def dpmlm_rewrite_batch(self, sentence, epsilon, REPLACE=False, STOP=False, CONCAT=True, batch_size=16):
        sentence = " ".join(sentence.split("\n"))
        tokens = nltk.word_tokenize(sentence)
        encoded = self.tokenizer.encode(sentence, add_special_tokens=False)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))]

        n = sentence_enum(tokens)
        batch = []
        for i in range(len(tokens)):
            # lower, upper = self.sliding_window(tokens, i, int((self.tokenizer.model_max_length-32)/2))
            # batch.append(self.tokenizer.decode(encoded[lower:upper], skip_special_tokens=True))
            batch.append(sentence)
        res = self.privatize_batch(batch, tokens, n=n, epsilon=word_eps, CONCAT=CONCAT, STOP=STOP, batch_size=batch_size)

        replace = []
        #for i, r in enumerate(res):
        for i, (t, x) in enumerate(zip(tokens, n)):
            r = "{}_{}".format(t, x)
            if tokens[i][0].isupper() == True:
                replace.append(res[r].capitalize())
            else:
                replace.append(res[r].lower())

        return self.detokenizer.detokenize(replace)
    
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