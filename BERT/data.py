from library import *

class Dataset:

    def __init__(self, batch_size, max_length, max_predict, data_path):
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_predict = max_predict
        self.vocab_size = 0
        self.number_dict = {}
        self.word_dict = {}
        self.data_path = data_path
        self.col_label = 'review'

    def remove_html_tags(self, sentence):
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        return sentence

    def remove_punctuation(self, sentence):
        words = sentence.split()
        filtered_sentence = ''
        table = str.maketrans('', '', string.punctuation)
        for word in words:
            word = word.translate(table)
            filtered_sentence += word + ' '

        return filtered_sentence

    def remove_emoji(self, sentence):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', sentence)

    def clean_sentence(self, sentences):
        clean_sentences = ""
        for sentence in sentences:
            sentence = sentence.lower()         #1 convert to original text to lower
            sentence = self.remove_html_tags(sentence)
            sentence = self.remove_emoji(sentence)
            sentence = self.remove_punctuation(sentence)
            clean_sentences += sentence+"\n"
        return clean_sentences

    def data_preprocessing(self, sentences):
        sentences = self.clean_sentence(sentences)

        return sentences


    def load_dataset(self, data_name):
        print(" ")
        print("Load dataset ... ")
        df = pd.read_csv(self.data_path)
        sentences = df[data_name].to_list()[0:500]

        # cleaning and convert labels
        sentences = self.data_preprocessing(sentences)

        return sentences

    def build_data(self, text):
        sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
        word_list = list(set(" ".join(sentences).split()))
        word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(word_list):
            word_dict[w] = i + 4
        number_dict = {i: w for i, w in enumerate(word_dict)}
        vocab_size = len(word_dict)
        self.vocab_size = vocab_size
        self.number_dict = number_dict
        self.word_dict = word_dict
        token_list = list()
        for sentence in sentences:
            arr = [word_dict[s] for s in sentence.split()]
            token_list.append(arr)
        maxlen = self.max_length # maximum of length
        batch_size = self.batch_size
        max_pred = self.max_predict  # max tokens of prediction
        batch = []
        positive = negative = 0
        # we will end until positive == negative = batch_size/2
        while positive != batch_size/2 or negative != batch_size/2:
            tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences))
            tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]
            # We get batch_size / 2 because for each loop we get 2 sentences
            input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
            if len(input_ids) > self.max_length:
                continue
            # 1 mean "word_dict['[CLS]']", "word_dict['[SEP]']", "word_dict['[SEP]']"
            # => input_ids = segment_ids
            segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
            #MASK LM
            n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                            if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
            shuffle(cand_maked_pos)
            # we will replace and mask
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                if random() < 0.8:  # 80%
                    input_ids[pos] = word_dict['[MASK]'] # make mask
                elif random() < 0.5:  # 50%
                    index = randint(0, vocab_size - 1) # random index in vocabulary
                    input_ids[pos] = word_dict[number_dict[index]] # replace

            # Zero Paddings
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)

        #     # Zero Padding (100% - 15%) tokens
            if max_pred > n_pred:
                n_pad = max_pred - n_pred
                masked_tokens.extend([0] * n_pad)
                masked_pos.extend([0] * n_pad)
            # we only put when positive < batch_size/2
            if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
                positive += 1
            elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
                negative += 1
        return batch

    def build_dataset(self):
        text = self.load_dataset(self.col_label)
        batch = self.build_data(text)
        train_dataset = map(lambda batch: tf.convert_to_tensor(batch, tf.int64), zip(*batch))
        return train_dataset


    def build_test_dataset(self):
        text = (
            'how are you.\n'
            'Nice to meet you.\n'
            'How are you today?\n'
            'My baseball team won the competition\n'
            'Oh Congratulations\n'
            'Thanks you.'
        )
        batch = self.build_data(text)[0]
        train_dataset = map(lambda batch: tf.convert_to_tensor(batch, tf.int64), zip(batch))
        return train_dataset