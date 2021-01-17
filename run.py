from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array, AutoRegressiveDecoder
import numpy as np

# load model
config_path = './bert_model/bert_config.json'
dict_path = './bert_model/vocab.txt'
trump_ckpt = './bert_model/trump.ckpt'
biden_ckpt = './bert_model/biden.ckpt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
trump_model = build_transformer_model(
    config_path=config_path, checkpoint_path=trump_ckpt, with_mlm=True, model='bert', application='unilm'
)
biden_model = build_transformer_model(
    config_path=config_path, checkpoint_path=biden_ckpt, with_mlm=True, model='bert', application='unilm'
)

# generate argument
class TrumpGenerator():
    def __init__(self, end_id, maxlen):
        self.end_id = end_id
        self.maxlen = maxlen
    
    def generate(self, context):
        token_ids, segment_ids = tokenizer.encode(context)
        context_len = len(token_ids)
        segment_id = segment_ids[-1] + 1
        sentence = ''
        words = []
        gen_tokens = []
        for i in range(self.maxlen):
            token_ids.append(tokenizer._token_dict['[MASK]'])
            segment_ids.append(segment_id)
            tokens, segments = to_array([token_ids], [segment_ids])
            probas = trump_model.predict([tokens, segments])[0]
            # token = probas[context_len + i].argmax()
            ids = np.argsort(probas[context_len + i])[::-1]
            for token in ids:
                if token not in gen_tokens:
                    gen_tokens.append(token)
                    break
            words.append(tokenizer.decode([token]))
            token_ids[context_len + i] = token
            if token in self.end_id:
                sentence = ' '.join(words)
                return sentence
        sentence = ' '.join(words)
        sentence += '.'
        return sentence

class BidenGenerator():
    def __init__(self, end_id, maxlen):
        self.end_id = end_id
        self.maxlen = maxlen
    
    def generate(self, context):
        token_ids, segment_ids = tokenizer.encode(context)
        context_len = len(token_ids)
        segment_id = segment_ids[-1] + 1
        sentence = ''
        words = []
        gen_tokens = []
        for i in range(self.maxlen):
            token_ids.append(tokenizer._token_dict['[MASK]'])
            segment_ids.append(segment_id)
            tokens, segments = to_array([token_ids], [segment_ids])
            probas = biden_model.predict([tokens, segments])[0]
            # token = probas[context_len + i].argmax()
            ids = np.argsort(probas[context_len + i])[::-1]
            for token in ids:
                if token not in gen_tokens:
                    gen_tokens.append(token)
                    break
            words.append(tokenizer.decode([token]))
            token_ids[context_len + i] = token
            if token in self.end_id:
                sentence = ' '.join(words)
                return sentence
        sentence = ' '.join(words)
        sentence += '.'
        return sentence

trump = TrumpGenerator(
    [tokenizer._token_dict['.'], tokenizer._token_dict['!'], tokenizer._token_dict['?']], 120
)
biden = BidenGenerator(
    [tokenizer._token_dict['.'], tokenizer._token_dict['!'], tokenizer._token_dict['?']], 120
)


context = 'america'
trump_tweet = trump.generate(context)
print("Trump: " + trump_tweet)
context += trump_tweet

biden_tweet = biden.generate(context)
print("Biden: " + biden_tweet)
