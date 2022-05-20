"""Python file with vocabulary class and utils."""
# https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html

import re

PAD_INDEX = 0 # Used for padding short sentences
SOS_INDEX = 1 # Start-of-sentence token
EOS_INDEX = 2 # End-of-sentence token

class Vocabulary:

    def __init__(self):
        self.word2index = {
            "<PAD>": PAD_INDEX,
            "<SOS>": SOS_INDEX,
            "<EOS>": EOS_INDEX,
        }
        self.index2word = {
            PAD_INDEX: "<PAD>",
            SOS_INDEX: "<SOS>",
            EOS_INDEX: "<EOS>",
        }
        self.word2count = {
            "<PAD>": 0,
            "<SOS>": 0,
            "<EOS>": 0,
        }

        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


# https://github.com/pschwllr/MolecularTransformer
def smi_tokenizer(smi):
    """Tokenize a SMILES molecule or reaction."""
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(['<SOS>'] + tokens + ['<EOS>'])
