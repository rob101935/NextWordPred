from next_word_pred import predict 
import pytest

def test_inputTransformations():
    # created Tokenized NGram
    predictor = predict.LSTM_next_word_pred()
    tokenNgrams = predictor.inputTransformations([''],predictor.tokenizer) 
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert all([a == b for a, b in zip(tokenNgrams[0], expected)])
    tokenNgrams = predictor.inputTransformations(['a'],predictor.tokenizer) 
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]
    assert all([a == b for a, b in zip(tokenNgrams[0], expected)])
def test_singleWord():
    
    predictor = predict.LSTM_next_word_pred() 
    
    assert predictor.predict("where") == ['are', 'do', 'you', 'is', 'will']

def test_EmptyString():
    
    predictor = predict.LSTM_next_word_pred() 
    
    assert predictor.predict("") == ['a', 'going', 'not', 'you', 'better']

def test_valError():
    predictor = predict.LSTM_next_word_pred()
    with pytest.raises(ValueError):
            predictor.predict([""])