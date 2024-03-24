import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import re

class GPT2_next_word_pred:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def predict(self,starting_sentence):
        if type(starting_sentence) != str: 
            raise ValueError('Prediction was not passed string format, please pass predictor string format.')
        if starting_sentence in [""," "]:
            starting_sentence = "  "
        t = self.tokenizer
        if starting_sentence[-1] == " ":
            starting_sentence = starting_sentence[:-1]
        encoded_text = t(starting_sentence, return_tensors="pt")

        #1. step to get the logits of the next token
        with torch.inference_mode():
            outputs =  self.model(**encoded_text)

        next_token_logits = outputs.logits[0, -1, :]
        p = re.compile("[^a-z0-9']")
        # often returns punctuation for this version we only want words so take top 20 just in case and then filter down to have only words using regex
        topk_next_tokens= torch.topk(next_token_logits, 20, sorted = True)
        top_five_words = [p.sub('',t.decode(idx).lower()) for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values) if len(p.sub('', t.decode(idx))) >0][:5]
        # print(f'{starting_sentence}: Next word: {top_five_words}')
        return top_five_words