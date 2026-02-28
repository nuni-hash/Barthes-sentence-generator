import random
import re
import nltk
from collections import defaultdict, Counter


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

START = "<START>"
END = "<END>"

class MiniRecombiner:
    def __init__(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            raw = f.read()

        self.sentences = nltk.sent_tokenize(raw)
        self.words_by_pos = defaultdict(Counter)
        self._train()

    def _train(self):
        for sent in self.sentences:
            toks = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(toks)
            for word, pos in tagged:
                if re.match(r"^\W+$", word):
                    continue
                self.words_by_pos[pos][word] += 1

        #  aggregated categories
        self.cats = {
            "DET": sum_counters(self.words_by_pos, ("DT",)),
            "ADJ": sum_counters(self.words_by_pos, ("JJ", "JJR", "JJS")),
            "NOUN_SG": sum_counters(self.words_by_pos, ("NN", "NNP")),
            "NOUN_PL": sum_counters(self.words_by_pos, ("NNS", "NNPS")),
            "PRON": sum_counters(self.words_by_pos, ("PRP",)),
            "VB": sum_counters(self.words_by_pos, ("VB",)),
            "VBP": sum_counters(self.words_by_pos, ("VBP",)),
            "VBZ": sum_counters(self.words_by_pos, ("VBZ",)),
            "VBD": sum_counters(self.words_by_pos, ("VBD",)),
            "PREP": sum_counters(self.words_by_pos, ("IN",)),
            "ADV": sum_counters(self.words_by_pos, ("RB", "RBR", "RBS")),
        }

    def _sample_from_counter(self, counter):
        if not counter:
            return ""
        words, weights = zip(*counter.items())
        return random.choices(words, weights=weights, k=1)[0]

    def _sample_cat(self, cat):
        return self._sample_from_counter(self.cats.get(cat, Counter()))

    def _is_plural_from_choice(self, word, cat):
        
        if cat == "NOUN_PL":
            return True
        if cat == "NOUN_SG":
            return False
        if cat == "PRON":
            return word.lower() in ("we", "they", "you")
        return False

    def _conjugate_3sg(self, verb):
        # Very naive conjugation
        if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
            return verb[:-1] + "ies"
        if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
            return verb + "es"
        return verb + "s"
    
    def _correct_indefinite_article(self, det, next_word):
        if not det or det.lower() not in ("a", "an") or not next_word:
            return det
        # simple vowel check (naive but effective for this toy model)
        if next_word[0].lower() in "aeiou":
            return "an"
        else:
            return "a"
        
    def _choose_verb_for_subject(self, subj_plural):
        
        if subj_plural:
            
            v = self._sample_cat("VBP") or self._sample_cat("VB")
            return v
        else:
            
            v = self._sample_cat("VBZ") or self._sample_cat("VB")
            if v:
                
                if v in self.cats.get("VBZ", {}):
                    return v
                if v in self.cats.get("VB", {}):
                    return self._conjugate_3sg(v)
            return v or ""

    def _build_np(self, for_subject=False):
        # Decide pronoun vs determiner + noun
        if for_subject and random.random() < 0.25 and self.cats["PRON"]:
            pron = self._sample_cat("PRON")
            return pron, self._is_plural_from_choice(pron, "PRON")

        det = ""
        if random.random() < 0.75 and self.cats["DET"]:
            det = self._sample_cat("DET")

        adj = ""
        if random.random() < 0.4 and self.cats["ADJ"]:
            adj = self._sample_cat("ADJ")

        if random.random() < 0.25 and self.cats["NOUN_PL"]:
            noun = self._sample_cat("NOUN_PL")
            plural = True
        else:
            noun = self._sample_cat("NOUN_SG") or self._sample_cat("NOUN_PL")
            plural = False if noun in self.cats.get("NOUN_SG", {}) else True
        if noun == "":
            noun = self._sample_cat("NOUN_SG") or self._sample_cat("NOUN_PL") or "thing"

        
        next_word = adj if adj else noun
        det = self._correct_indefinite_article(det, next_word)

        parts = " ".join(p for p in (det, adj, noun) if p)
        return parts, plural

    def _build_pp(self):
        if random.random() > 0.35 or not self.cats["PREP"]:
            return ""
        prep = self._sample_cat("PREP")
        obj, _ = self._build_np(for_subject=False)
        return f"{prep} {obj}" if obj else prep

    def generate_sentence(self):
        
        subj, subj_plural = self._build_np(for_subject=True)
        verb = self._choose_verb_for_subject(subj_plural)
        if not verb:
            verb = self._sample_cat("VB") or "be"

        # optionally object
        obj = ""
        if random.random() < 0.6:
            obj, _ = self._build_np(for_subject=False)

        pp = self._build_pp()

        # assemble
        pieces = [p for p in (subj, verb, obj, pp) if p]
       
        if len(" ".join(pieces).split()) > 12:
            
            if pp:
                pieces.remove(pp)
            if len(" ".join(pieces).split()) > 10 and obj:
                pieces.remove(obj)

        if not pieces:
            return ""

        sentence = " ".join(pieces)
        sentence = re.sub(r"\s+([,.;:?!])", r"\1", sentence)
        sentence = sentence[0].upper() + sentence[1:]
        if not re.search(r"[.!?]$", sentence):
            sentence += "."
        return sentence

def sum_counters(src, keys):
    out = Counter()
    for k in keys:
        out.update(src.get(k, Counter()))
    return out

if __name__ == "__main__":
    model = MiniRecombiner("Barthes.txt")
    for _ in range(12):
        print(model.generate_sentence())