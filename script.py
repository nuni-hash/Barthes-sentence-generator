import random
import nltk
from collections import defaultdict, Counter
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

# Falls noch nicht geschehen (einmalig ausführen)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


class MiniLanguageModel:
    def __init__(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.tokens = nltk.word_tokenize(self.text)
        self.tagged = nltk.pos_tag(self.tokens)

        self.words_by_pos = defaultdict(list)
        self.pos_transitions = defaultdict(Counter)

        self._train()

    def _train(self):
        # Wörter nach POS sammeln
        for word, pos in self.tagged:
            if word.isalpha():
                self.words_by_pos[pos].append(word.lower())

        # POS-Übergänge lernen
        for i in range(len(self.tagged) - 1):
            pos1 = self.tagged[i][1]
            pos2 = self.tagged[i + 1][1]
            self.pos_transitions[pos1][pos2] += 1

    def _next_pos(self, current_pos):
        transitions = self.pos_transitions[current_pos]
        if not transitions:
            return random.choice(list(self.words_by_pos.keys()))
        return random.choices(
            list(transitions.keys()),
            weights=list(transitions.values())
        )[0]

    def generate_sentence(self, max_length=12):
        sentence = []

        # typische englische Start-POS
        possible_starts = [p for p in self.words_by_pos if p in ("DT", "PRP", "NN")]
        current_pos = random.choice(possible_starts)

        for _ in range(max_length):
            if current_pos not in self.words_by_pos:
                break

            word = random.choice(self.words_by_pos[current_pos])
            sentence.append(word)

            current_pos = self._next_pos(current_pos)

        return " ".join(sentence).capitalize() + "."

model = MiniLanguageModel("Barthes.txt")

for _ in range(10):
    print(model.generate_sentence())
