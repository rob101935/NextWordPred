from next_word_pred import predict_gpt2 
import pytest

# def test_inputTransformations():
#     # created Tokenized NGram
#     predictor = predict.LSTM_next_word_pred()
#     tokenNgrams = predictor.inputTransformations([''],predictor.tokenizer) 
#     expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     assert all([a == b for a, b in zip(tokenNgrams[0], expected)])
#     tokenNgrams = predictor.inputTransformations(['a'],predictor.tokenizer) 
#     expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]
#     assert all([a == b for a, b in zip(tokenNgrams[0], expected)])
def test_singleWord():
    
    predictor = predict_gpt2.GPT2_next_word_pred() 
    
    assert predictor.predict("where") == ['the', 'and', 'is', 'of', 'a']
    assert predictor.predict("where ") == ['the', 'and', 'is', 'of', 'a']
def test_multiWord():
    
    predictor = predict_gpt2.GPT2_next_word_pred() 
    
    assert predictor.predict("I will be back ") == ['in', 'soon', 'to', 'with', 'on']
    assert predictor.predict("I will be back") == ['in', 'soon', 'to', 'with', 'on']

def test_EmptyString():
    
    predictor = predict_gpt2.GPT2_next_word_pred() 
    
    assert predictor.predict("") == ['the', 'of', '1', '2', 'is']

def test_valError():
    predictor = predict_gpt2.GPT2_next_word_pred() 
    with pytest.raises(ValueError):
            predictor.predict([""])