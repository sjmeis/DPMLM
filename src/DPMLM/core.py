import os
import torch
import nltk
import string
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, logging, pipeline
import textspan

from presidio_analyzer import AnalyzerEngine
#from presidio_anonymizer import AnonymizerEngine

torch.set_float32_matmul_precision('medium')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.set_verbosity_warning()

stop = set(stopwords.words("english"))

def replace_at_index(token_list, index, replacement):
    '''Replacement function for to-be-deprecated nth_repl.'''
    new_tokens = list(token_list)
    if 0 <= index < len(new_tokens):
        new_tokens[index] = replacement
    return new_tokens

def remove_at_index(token_list, index):
    '''Replacement function for to-be-deprecated nth_rem.'''
    new_tokens = list(token_list)
    if 0 <= index < len(new_tokens):
        new_tokens.pop(index)
    return new_tokens

# def nth_repl(s, sub, repl, n):
#     s_split = nltk.word_tokenize(s)
#     i = 0
#     try:
#         find = s_split.index(sub)
#         i += 1
#     except ValueError:
#         return s

#     while i != n:
#         try:
#             find = s_split.index(sub, find + 1)
#             i += 1
#         except ValueError:
#             break
#     if i == n:
#         return " ".join(s_split[:find] + [repl] + s_split[find+1:])
#     return s

# def nth_rem(s, sub, n):
#     s_split = s.split()
#     i = 0
#     try:
#         find = s_split.index(sub)
#         i += 1
#     except ValueError:
#         return s
    
#     while i != n:
#         try:
#             find = s_split.index(sub, find + 1)
#             i += 1
#         except ValueError:
#             break
#     if i == n:
#         return " ".join(s_split[:find] + s_split[find+1:])
#     return s

# def sentence_enum(tokens):
#     counts = Counter()
#     n = []
#     for t in tokens:
#         counts[t] += 1
#         n.append(counts[t])
#     return n

# def get_opposites():
# 	with open(impresources.files("DPMLM") / "data" / "opposites.json", 'r') as f:
# 		opposites = json.load(f)
# 	return opposites

# def get_vocab():
# 	with open(impresources.files("DPMLM") / "data" / "vocab.txt", 'r') as f:
# 		vocab = set([x.strip() for x in f.readlines()])
# 	return vocab

class DPMLM():
    # opposites = get_opposites()
    # vocab = get_vocab()
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()
    tokenizer = None
    lm_model = None
    raw_model = None
    device = None
    nlp = None
    alpha = None

    def __init__(self, MODEL="FacebookAI/roberta-base", alpha=0.003, IPI=False, IPI_model=None, PII=False, calibration=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.max_context = (self.tokenizer.model_max_length // 2) - 32

        self.lm_model = AutoModelForMaskedLM.from_pretrained(MODEL)
        self.alpha = alpha

        if calibration:
            self.clip_min = calibration.get('clip_min')
            self.clip_max = calibration.get('clip_max')
            self.sensitivity = calibration.get('sensitivity')
        else:
            # default is for roberta-base
            self.clip_min = -3.2093127
            self.clip_max = 16.304797887802124
            self.sensitivity = abs(self.clip_max - self.clip_min)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm_model = self.lm_model.to(self.device)
        self.lm_model.eval()
        torch.set_grad_enabled(False)

        if IPI == True:
            self.ipi_pipe = pipeline("ner", model=IPI_model, device=self.device)

        if PII == True:
            self.analyzer = AnalyzerEngine()

    def load_transformers(self):
        return self.tokenizer, self.lm_model

    def privatize(self, token_list, target, start_index, CONCAT=True, epsilon=1):
        max_word_context = self.max_context
        lower_w, upper_w = self.sliding_window(token_list, start_index, max_word_context)
        chunk_tokens = token_list[lower_w:upper_w]
        relative_index = start_index - lower_w

        # Masks the target word in the original sentence.
        masked_tokens = replace_at_index(chunk_tokens, relative_index, self.tokenizer.mask_token)
        masked_sent = self.detokenizer.detokenize(masked_tokens)

        #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
        if CONCAT == False:
            input_ids = self.tokenizer.encode(" "+masked_sent, add_special_tokens=True, truncation=True, max_length=self.tokenizer.model_max_length)
        else:
            original_sent_clean = self.detokenizer.detokenize(chunk_tokens)
            input_ids = self.tokenizer.encode(" " + original_sent_clean, " " + masked_sent, add_special_tokens=True, truncation="only_first", max_length=self.tokenizer.model_max_length)

        try:
            masked_position = input_ids.index(self.tokenizer.mask_token_id)
        except ValueError:
            return {"{}_{}".format(target, start_index): target}

        #Get the predictions of the Masked LM transformer.
        with torch.no_grad():
            output = self.lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))
            mask_logits = output[0, masked_position].detach().cpu().numpy()

        #Get top guesses: their token IDs, scores, and words.
        mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
        mask_logits = mask_logits / (2 * self.sensitivity / epsilon)

        # use for stability, equivalent to softmax
        shifted_logits = mask_logits - np.max(mask_logits)
        scores = np.exp(shifted_logits)
        scores = scores / scores.sum()
        chosen_idx = np.random.choice(len(scores), p=scores)
        prediction = self.tokenizer.decode(chosen_idx).strip()

        return {"{}_{}".format(target, start_index): prediction}
    
    def privatize_batch(self, tokens, indices, epsilon, CONCAT=True, batch_size=16):
        predictions = {}

        for k in range(0, len(indices), batch_size):
            batch_indices = indices[k : k + batch_size]
            batch_eps = epsilon[k : k + batch_size]
            
            batch_input_ids = []
            batch_mask_positions = []
            
            for idx in batch_indices:
                masked_tokens = replace_at_index(tokens, idx, self.tokenizer.mask_token)
                
                max_word_context = self.max_context 
                lower_w, upper_w = self.sliding_window(tokens, idx, max_word_context)
                
                chunk_tokens = tokens[lower_w:upper_w]
                masked_chunk_tokens = masked_tokens[lower_w:upper_w]
                rel_idx = idx - lower_w

                masked_chunk_tokens = replace_at_index(chunk_tokens, rel_idx, self.tokenizer.mask_token)
                
                clean_chunk_sent = self.detokenizer.detokenize(chunk_tokens)
                masked_chunk_sent = self.detokenizer.detokenize(masked_chunk_tokens)
                
                if CONCAT == False:
                    input_ids = self.tokenizer.encode(" " + masked_chunk_sent, add_special_tokens=True, truncation=True, max_length=self.tokenizer.model_max_length)
                else:
                    input_ids = self.tokenizer.encode(" " + clean_chunk_sent, " " + masked_chunk_sent, add_special_tokens=True, truncation="only_first", max_length=self.tokenizer.model_max_length)
                
                try:
                    m_pos = input_ids.index(self.tokenizer.mask_token_id)
                except ValueError:
                    m_pos = 0 # fallback
                    
                batch_input_ids.append(input_ids)
                batch_mask_positions.append(m_pos)

            inputs = self.tokenizer.pad({"input_ids": batch_input_ids}, padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                logits_output = self.lm_model(**inputs).logits # [batch, seq, vocab]

            for i, (idx, eps) in enumerate(zip(batch_indices, batch_eps)):
                target_word = tokens[idx]
                m_pos = batch_mask_positions[i]

                mask_logits = logits_output[i, m_pos].cpu().numpy()
                mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
                mask_logits = mask_logits / (2 * self.sensitivity / eps)
                
                shifted = mask_logits - np.max(mask_logits)
                scores = np.exp(shifted)
                scores /= scores.sum()
                
                chosen_idx = np.random.choice(len(scores), p=scores)
                pred = self.tokenizer.decode(chosen_idx).strip()
                
                predictions["{}_{}".format(target_word, idx)] = pred

        return predictions
    
    def sliding_window(self, tokens, target_idx, max_len):
        length = len(tokens)
        if length <= max_len:
            return 0, length

        half_window = max_len // 2
        lower = target_idx - half_window
        upper = target_idx + half_window

        if lower < 0:
            upper -= lower
            lower = 0
        elif upper > length:
            lower -= (upper - length)
            upper = length

        lower = max(0, lower)
        upper = min(length, upper)

        return int(lower), int(upper)

    def dpmlm_rewrite(self, sentence, epsilon, REPLACE=False, STOP=False, CONCAT=True, IPI=False, PII=False):
        sentence = " ".join(sentence.split("\n"))
        tokens = nltk.word_tokenize(sentence)

        if PII == True:
            sentence = sentence.replace("<", "").replace(">", "")
            results = self.analyzer.analyze(text=sentence, language="en")
            pii_spans = [(x.start, x.end) for x in results]
            pii_types = [x.entity_type for x in results]
            rep_spans = []
            offset = 0
            for s, t in zip(pii_spans, pii_types):
                rep = "<" + t + ">"
                rep_len = s[1] - s[0]
                sentence = sentence[:s[0]+offset] + rep + sentence[s[1]+offset:]
                offset = offset - rep_len + len(rep)
                rep_spans.append((len(sentence[:s[0]]), len(sentence[:s[0]])+len(rep)))
            tokens = nltk.word_tokenize(sentence)
            orig_spans = [x[0] for x in textspan.get_original_spans(tokens, sentence)]
            pii_mask = []
            started = False
            for t, s in zip(tokens, orig_spans):
                if t == "<":
                    pii_mask.append(True)
                    started = True
                elif t == ">":
                    pii_mask.append(True)
                    started = False
                else:
                    if started == True:
                        pii_mask.append(True)
                    else:
                        pii_mask.append(False)
        else:
            pii_mask = None

        if IPI == True:
            res = self.ipi_pipe(sentence)
            ipi_entities = [x["entity"] for x in res]
            ipi_spans = [(x["start"], x["end"]) for x in res]
            orig_spans = [x[0] for x in textspan.get_original_spans(tokens, sentence)]
            ipi_mask = [False if x in ipi_spans else True for x in orig_spans]
        else:
            ipi_mask = None

        if ipi_mask is not None and pii_mask is not None:
            assert len(pii_mask) == len(ipi_mask)
            all_mask = []
            for x, y in zip(pii_mask, ipi_mask):
                if x is True or y is True:
                    all_mask.append(True)
                else:
                    all_mask.append(False)
        elif ipi_mask is not None:
            all_mask = ipi_mask
        elif pii_mask is not None:
            all_mask = pii_mask
        else:
            all_mask = None

        word_eps = epsilon if isinstance(epsilon, list) else [epsilon] * len(tokens)
        replace = []
        working_tokens = list(tokens)

        perturbed = 0
        total = 0
        for i, (t, eps) in enumerate(zip(tokens, word_eps)):
            # if i >= len(tokens):
            #     break

            # if IPI or PII, skip non-IPI/PII tokens
            if all_mask is not None and all_mask[i] == True:
                total += 1
                if tokens[i].isupper() == True:
                    replace.append(t)
                elif tokens[i][0].isupper() == True:
                    replace.append(t.capitalize())
                else:
                    replace.append(t)
                continue

            if (STOP == False and t in stop) or t in string.punctuation:
                total += 1
                if tokens[i][0].isupper() == True:
                    replace.append(t.capitalize())
                else:
                    replace.append(t)
                continue

            res = self.privatize(working_tokens, t, i, CONCAT=CONCAT, epsilon=eps)
            r = res.get(f"{t}_{i}", t)
            
            if REPLACE:
                working_tokens[i] = r

            if tokens[i][0].isupper() == True:
                replace.append(r.capitalize())
            else:
                replace.append(r.lower())

            if r != t:
                perturbed += 1
            total += 1

        if IPI == True:
            return self.detokenizer.detokenize(replace), perturbed, total, ipi_entities
        else:
            return self.detokenizer.detokenize(replace), perturbed, total
    
    def dpmlm_rewrite_batch(self, sentence, epsilon, STOP=False, CONCAT=True, batch_size=16, IPI=False, PII=False):
        sentence = " ".join(sentence.split("\n"))
        tokens = nltk.word_tokenize(sentence)

        if PII == True:
            sentence = sentence.replace("<", "").replace(">", "")
            results = self.analyzer.analyze(text=sentence, language="en")
            pii_spans = [(x.start, x.end) for x in results]
            pii_types = [x.entity_type for x in results]
            rep_spans = []
            offset = 0
            for s, t in zip(pii_spans, pii_types):
                rep = "<" + t + ">"
                rep_len = s[1] - s[0]
                sentence = sentence[:s[0]+offset] + rep + sentence[s[1]+offset:]
                offset = offset - rep_len + len(rep)
                rep_spans.append((len(sentence[:s[0]]), len(sentence[:s[0]])+len(rep)))
            tokens = nltk.word_tokenize(sentence)
            orig_spans = [x[0] for x in textspan.get_original_spans(tokens, sentence)]
            pii_mask = []
            started = False
            for t, s in zip(tokens, orig_spans):
                if t == "<":
                    pii_mask.append(True)
                    started = True
                elif t == ">":
                    pii_mask.append(True)
                    started = False
                else:
                    if started == True:
                        pii_mask.append(True)
                    else:
                        pii_mask.append(False)
        else:
            pii_mask = None

        if IPI == True:
            res = self.ipi_pipe(sentence)
            ipi_entities = [x["entity"] for x in res]
            ipi_spans = [(x["start"], x["end"]) for x in res]
            orig_spans = [x[0] for x in textspan.get_original_spans(tokens, sentence)]
            ipi_mask = [False if x in ipi_spans else True for x in orig_spans]
        else:
            ipi_mask = None

        if ipi_mask is not None and pii_mask is not None:
            assert len(pii_mask) == len(ipi_mask)
            all_mask = []
            for x, y in zip(pii_mask, ipi_mask):
                if x is True or y is True:
                    all_mask.append(True)
                else:
                    all_mask.append(False)
        elif ipi_mask is not None:
            all_mask = ipi_mask
        elif pii_mask is not None:
            all_mask = pii_mask
        else:
            all_mask = None

        indices_to_process = []
        word_eps = epsilon if isinstance(epsilon, list) else [epsilon] * len(tokens)

        for i, t in enumerate(tokens):
            if (all_mask and all_mask[i]) or \
            (not STOP and t.lower() in self.stop) or \
            (t in string.punctuation):
                continue
            indices_to_process.append(i)

        res = self.privatize_batch(tokens, indices_to_process, epsilon=word_eps, CONCAT=CONCAT, batch_size=batch_size)

        replace = []
        perturbed = 0
        total = 0
        for i, t in enumerate(tokens):
            key = "{}_{}".format(t, i)
            if key in res:
                r = res[key]
                
                # maintain capitalization
                if t[0].isupper():
                    rep_token = r.capitalize()
                else:
                    rep_token = r.lower()
                
                replace.append(rep_token)
                if rep_token != t:
                    perturbed += 1
            else:
                # token was skipped
                replace.append(t)
            
            total += 1

        if IPI == True:
            return self.detokenizer.detokenize(replace), perturbed, total, ipi_entities
        else:
            return self.detokenizer.detokenize(replace)
    
    def dpmlm_rewrite_plus(self, sentence, epsilon, CONCAT=True, ADD_PROB=0.15, DEL_PROB=0.05):
        if isinstance(sentence, list):
            tokens = sentence
        else:
            tokens = nltk.word_tokenize(sentence)

        if isinstance(epsilon, list):
            word_eps = epsilon
        else:
            word_eps = [epsilon for _ in range(len(tokens))] # epsilon / num_tokens
        replace = []
        working_tokens = [str(x) for x in tokens]

        perturbed = 0
        total = 0
        deleted = 0
        added = 0
        cursor = 0

        for i, (t, eps) in enumerate(zip(tokens, word_eps)):
            if t in string.punctuation:
                total += 1
                replace.append(t)
                cursor += 1
                continue

            if i == len(tokens) - 1:
                DELETE = 1
            else:
                DELETE = np.random.rand()


            if DELETE < DEL_PROB:
                working_tokens = remove_at_index(working_tokens, cursor)
                deleted += 1
                continue

            res = self.privatize(working_tokens, t, n=1, start_index=cursor, CONCAT=CONCAT, epsilon=eps)
            r = res["{}_{}".format(t, cursor)]
            
            working_tokens[cursor] = r
            replace.append(r)
            
            if r != t:
                perturbed += 1
            total += 1

            ADD = np.random.rand()
            if ADD <= ADD_PROB:
                working_tokens.insert(cursor + 1, self.tokenizer.mask_token)
            
                res_add = self.privatize(working_tokens, self.tokenizer.mask_token, start_index=cursor + 1, CONCAT=CONCAT, epsilon=eps)
                r_add = res_add["{}_{}".format(self.tokenizer.mask_token, cursor + 1)]
                
                # Swap mask for the predicted word
                working_tokens[cursor + 1] = r_add
                replace.append(r_add)
                
                added += 1
                cursor += 1

            cursor += 1

        return self.detokenizer.detokenize(replace), perturbed, total, added, deleted