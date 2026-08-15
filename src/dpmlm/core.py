import torch
import nltk
import string
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, logging, pipeline

from presidio_analyzer import AnalyzerEngine

torch.backends.cuda.matmul.fp32_precision = 'tf32'

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

class DPMLM():
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()
    nltk_tokenizer = TreebankWordTokenizer()
    tokenizer = None
    lm_model = None
    raw_model = None
    device = None
    nlp = None
    alpha = None

    def __init__(self, MODEL="FacebookAI/roberta-base", alpha=0.003, IPI=False, IPI_model=None, PII=False, calibration=None, hybrid=False, hybrid_budget=100):
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

        if hybrid == True:
            self.hybrid = True
            self.hybrid_budget = hybrid_budget
        else:
            self.hybrid = False

    def load_transformers(self):
        return self.tokenizer, self.lm_model

    def privatize(self, token_list, target, start_index, CONCAT=True, epsilon=1):
        masked_tokens = replace_at_index(token_list, start_index, self.tokenizer.mask_token)
        masked_sent_full = self.detokenizer.detokenize(masked_tokens)
        
        # Encode to sub-token space without adding special tokens yet
        encoded_masked = self.tokenizer.encode(masked_sent_full, add_special_tokens=False)

        try:
            mask_id_idx = encoded_masked.index(self.tokenizer.mask_token_id)
        except ValueError:
            mask_id_idx = len(encoded_masked) // 2

        # Calculate exact sub-token budget per chunk
        # If CONCAT=True, allocate half budget (minus room for <s>, </s>)
        budget = (self.tokenizer.model_max_length - 8) // 2 if CONCAT else (self.tokenizer.model_max_length - 4)

        # Slide window in sub-token space (!)
        lower, upper = self.sliding_window(encoded_masked, mask_id_idx, budget)
        windowed_masked_ids = encoded_masked[lower:upper]
        masked_chunk_sent = self.tokenizer.decode(windowed_masked_ids, skip_special_tokens=False)

        #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
        if CONCAT == False:
            input_ids = self.tokenizer.encode(" "+masked_chunk_sent, add_special_tokens=True, truncation=True, max_length=self.tokenizer.model_max_length)
        else:
            clean_sent_full = self.detokenizer.detokenize(token_list)
            encoded_clean = self.tokenizer.encode(clean_sent_full, add_special_tokens=False)
            
            # Extract identical sub-token window from clean text
            windowed_clean_ids = encoded_clean[lower:upper]
            clean_chunk_sent = self.tokenizer.decode(windowed_clean_ids, skip_special_tokens=False)
            input_ids = self.tokenizer.encode(" " + clean_chunk_sent, " " + masked_chunk_sent, add_special_tokens=True, truncation="longest_first", max_length=self.tokenizer.model_max_length)

        try:
            masked_position = input_ids.index(self.tokenizer.mask_token_id)
        except ValueError:
            return {"{}_{}".format(target, start_index): target}

        # Get the predictions of the Masked LM transformer.
        with torch.no_grad():
            output = self.lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(self.device))
            mask_logits = output.logits[0, masked_position].squeeze().cpu().numpy()

        # Get top guesses: token IDs, scores, and words.
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

        clean_sent_full = self.detokenizer.detokenize(tokens)
        encoded_clean = self.tokenizer.encode(clean_sent_full, add_special_tokens=False) if CONCAT else None
        budget = (self.tokenizer.model_max_length - 8) // 2 if CONCAT else (self.tokenizer.model_max_length - 4)

        for k in range(0, len(indices), batch_size):
            batch_indices = indices[k : k + batch_size]
            batch_eps = epsilon[k : k + batch_size]
            
            batch_input_ids = []
            batch_mask_positions = []
            
            for idx in batch_indices:
                masked_tokens = replace_at_index(tokens, idx, self.tokenizer.mask_token)
                masked_sent_full = self.detokenizer.detokenize(masked_tokens)

                # Encode full masked text to sub-token IDs
                encoded_masked = self.tokenizer.encode(masked_sent_full, add_special_tokens=False)
                
                try:
                    m_sub_idx = encoded_masked.index(self.tokenizer.mask_token_id)
                except ValueError:
                    m_sub_idx = len(encoded_masked) // 2
                
                lower, upper = self.sliding_window(encoded_masked, m_sub_idx, budget)
                
                masked_chunk_sent = self.tokenizer.decode(encoded_masked[lower:upper], skip_special_tokens=False)
                
                if CONCAT == False:
                    input_ids = self.tokenizer.encode(" " + masked_chunk_sent, add_special_tokens=True, truncation=True, max_length=self.tokenizer.model_max_length)
                else:
                    clean_chunk_sent = self.tokenizer.decode(encoded_clean[lower:upper], skip_special_tokens=False)
                    input_ids = self.tokenizer.encode(" " + clean_chunk_sent, " " + masked_chunk_sent, add_special_tokens=True, truncation="longest_first", max_length=self.tokenizer.model_max_length)
                
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

        pii_mask = []
        if PII == True:
            sentence = sentence.replace("<", "").replace(">", "")
            results = self.analyzer.analyze(text=sentence, language="en")
            
            sorted_results = sorted(results, key=lambda x: x.start, reverse=True)
            
            placeholder_char_ranges = []
            
            for x in sorted_results:
                rep = "<" + x.entity_type + ">"
                sentence = sentence[:x.start] + rep + sentence[x.end:]
                placeholder_char_ranges.append((x.start, x.start + len(rep)))
                
            tokens = nltk.word_tokenize(sentence)
            
            token_spans = list(self.nltk_tokenizer.span_tokenize(sentence))
            
            pii_mask = [False] * len(tokens)
            for i, (tok_start, tok_end) in enumerate(token_spans):
                for p_start, p_end in placeholder_char_ranges:
                    if max(tok_start, p_start) < min(tok_end, p_end):
                        pii_mask[i] = True
                        break
        else:
            tokens = nltk.word_tokenize(sentence)
            token_spans = list(self.nltk_tokenizer.span_tokenize(sentence))
            pii_mask = [False] * len(tokens)

        ipi_mask = [False] * len(tokens)
        ipi_entities = []
        if IPI == True:
            res = self.ipi_pipe(sentence)
            ipi_entities = [x["entity"] for x in res]
            ipi_spans = [(x["start"], x["end"]) for x in res]
            
            for i, (tok_start, tok_end) in enumerate(token_spans):
                for ipi_start, ipi_end in ipi_spans:
                    if max(tok_start, ipi_start) < min(tok_end, ipi_end):
                        ipi_mask[i] = True
                        break
        else:
            ipi_mask = [True] * len(tokens)

        all_mask = [False] * len(tokens)
        for i in range(len(tokens)):
            if PII and pii_mask[i]:
                all_mask[i] = True
            elif IPI and not ipi_mask[i]:
                all_mask[i] = True

        word_eps = epsilon if isinstance(epsilon, list) else [epsilon] * len(tokens)
        replace = []
        working_tokens = list(tokens)

        perturbed = 0
        total = 0
        for i, (t, eps) in enumerate(zip(tokens, word_eps)):
            # if IPI or PII, skip non-IPI/PII tokens
            skip = False
            if all_mask is not None and all_mask[i] == True:
                if self.hybrid == True:
                    skip = True
                    pass
                else:
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

            if skip == True:
                res = self.privatize(working_tokens, t, i, CONCAT=CONCAT, epsilon=self.hybrid_budget)
            else:
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

        pii_mask = []
        if PII == True:
            sentence = sentence.replace("<", "").replace(">", "")
            results = self.analyzer.analyze(text=sentence, language="en")
            
            sorted_results = sorted(results, key=lambda x: x.start, reverse=True)
            
            placeholder_char_ranges = []
            
            for x in sorted_results:
                rep = "<" + x.entity_type + ">"
                sentence = sentence[:x.start] + rep + sentence[x.end:]
                placeholder_char_ranges.append((x.start, x.start + len(rep)))
                
            tokens = nltk.word_tokenize(sentence)
            
            token_spans = list(self.nltk_tokenizer.span_tokenize(sentence))
            
            pii_mask = [False] * len(tokens)
            for i, (tok_start, tok_end) in enumerate(token_spans):
                for p_start, p_end in placeholder_char_ranges:
                    if max(tok_start, p_start) < min(tok_end, p_end):
                        pii_mask[i] = True
                        break
        else:
            tokens = nltk.word_tokenize(sentence)
            token_spans = list(self.nltk_tokenizer.span_tokenize(sentence))
            pii_mask = [False] * len(tokens)

        ipi_mask = [False] * len(tokens)
        ipi_entities = []
        if IPI == True:
            res = self.ipi_pipe(sentence)
            ipi_entities = [x["entity"] for x in res]
            ipi_spans = [(x["start"], x["end"]) for x in res]
            
            for i, (tok_start, tok_end) in enumerate(token_spans):
                for ipi_start, ipi_end in ipi_spans:
                    if max(tok_start, ipi_start) < min(tok_end, ipi_end):
                        ipi_mask[i] = True
                        break
        else:
            ipi_mask = [True] * len(tokens)

        all_mask = [False] * len(tokens)
        for i in range(len(tokens)):
            if PII and pii_mask[i]:
                all_mask[i] = True
            elif IPI and not ipi_mask[i]:
                all_mask[i] = True

        indices_to_process = []
        temp_eps = epsilon if isinstance(epsilon, list) else [epsilon] * len(tokens)
        word_eps = []

        for i, t in enumerate(tokens):
            if (not STOP and t.lower() in stop) or (t in string.punctuation):
                continue
            elif (all_mask and all_mask[i]):
                if self.hybrid == True:
                    indices_to_process.append(i)
                    word_eps.append(self.hybrid_budget)
            else:
                indices_to_process.append(i)
                word_eps.append(temp_eps[i])

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