import random
import re
import nltk
from collections import defaultdict, Counter

# Downloads (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

START = "<START>"
END = "<END>"

class MiniRecombiner:
    def __init__(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # basic cleanup (keep punctuation attached for tokenization)
        self.sentences = nltk.sent_tokenize(raw)
        self.pos_sequences = []
        self.words_by_pos = defaultdict(Counter)
        self.pos_transitions = defaultdict(Counter)

        self._train()

    def _train(self):
        for sent in self.sentences:
            toks = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(toks)

            # Build words_by_pos
            for word, pos in tagged:
                if re.match(r"^\W+$", word):  # punctuation
                    continue
                self.words_by_pos[pos][word] += 1

            # Build POS transitions per sentence with START/END
            seq = [START] + [pos for (_, pos) in tagged] + [END]
            for a, b in zip(seq, seq[1:]):
                self.pos_transitions[a][b] += 1

    def _sample_next_pos(self, current_pos):
        trans = self.pos_transitions.get(current_pos)
        if not trans:
            return END
        choices, weights = zip(*trans.items())
        return random.choices(choices, weights=weights, k=1)[0]

    def _sample_word_for_pos(self, pos, prefer_plural=None):
        # prefer_plural: None/default, True -> prefer plural verb (VBP), False -> prefer 3rd-sing (VBZ)
        counter = self.words_by_pos.get(pos)
        if not counter:
            # fallback: sample from any POS
            all_pos = list(self.words_by_pos.keys())
            if not all_pos:
                return ""
            pos = random.choice(all_pos)
            counter = self.words_by_pos[pos]
        words, weights = zip(*counter.items())
        return random.choices(words, weights=weights, k=1)[0]

    def _is_plural_subject(self, token, pos):
        # Heuristic: NNS = plural noun; PRP 'we','they' plural; 'you' ambiguous -> treat plural
        if pos == "NNS":
            return True
        if pos == "NN":
            return False
        if pos == "PRP":
            lower = token.lower()
            if lower in ("we", "they", "you"):
                return True
            return False
        return False

    def generate_sentence(self, max_len=20):
        pos = START
        pos_sequence = []
        tokens = []
        last_non_punct = ("", "")
        subj_info = None  # (token, pos, is_plural) — updated when we hit candidate subject

        while True:
            next_pos = self._sample_next_pos(pos)
            if next_pos == END or len(pos_sequence) >= max_len:
                break
            pos_sequence.append(next_pos)

            # Simple heuristic: decide if this POS is a noun/pronoun that can be a subject
            if subj_info is None and next_pos in ("NN", "NNS", "PRP", "NNP"):
                # pick a candidate subject word (sample)
                subj_word = self._sample_word_for_pos(next_pos)
                is_plural = self._is_plural_subject(subj_word, next_pos)
                subj_info = (subj_word, next_pos, is_plural)
                tokens.append(subj_word)
                last_non_punct = (subj_word, next_pos)
                pos = next_pos
                continue

            # If next_pos is a verb (present forms), try to match subject plurality
            if next_pos in ("VBZ", "VBP"):
                prefer_plural = None
                if subj_info is not None:
                    prefer_plural = subj_info[2]
                    # VBZ is 3rd-sing, so if prefer_plural True, prefer VBP over VBZ
                    if prefer_plural:
                        # if chosen POS is VBZ, try to instead pick VBP if available via transitions
                        if next_pos == "VBZ" and "VBP" in self.pos_transitions.get(pos, {}):
                            next_pos = "VBP"
                    else:
                        if next_pos == "VBP" and "VBZ" in self.pos_transitions.get(pos, {}):
                            next_pos = "VBZ"

            # sample a word for next_pos
            word = self._sample_word_for_pos(next_pos)
            tokens.append(word)
            last_non_punct = (word, next_pos)
            pos = next_pos

        # post-process tokens into a clean sentence
        sentence = []
        for i, tok in enumerate(tokens):
            # attach punctuation if token is punctuation-like
            if re.match(r"^[.,;:?!'\"-]+$", tok):
                if sentence:
                    sentence[-1] = sentence[-1] + tok
                else:
                    sentence.append(tok)
            else:
                sentence.append(tok)

        if not sentence:
            return ""

        # Capitalize first word, ensure ending punctuation
        sentence[0] = sentence[0].capitalize()
        out = " ".join(sentence)
        if not re.search(r"[.!?]$", out):
            out = out + "."
        return out

if __name__ == "__main__":
    model = MiniRecombiner("Barthes.txt")
    for _ in range(10):
        print(model.generate_sentence())