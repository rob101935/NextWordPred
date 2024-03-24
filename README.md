
## General approach
I decided to use NGram technique using a bidirectional LSTM to encode the sequence information and used Glove to embed the semantic meaning of the words in the dictionary into vectors in the network. Teh sizes of the network were all modelled from a 100d glove embedding layer. Assuming this would be used in a smartphone the size of the model is important. These would be a reasonable sized network and fit into 8gb of VRAM using CUDA which used around 6gb during prediction. The stored model itself and tokenizer is 42Mb in storage. This is a major advantage over using the state of the art LLM for example as a model such as mistral 7b with GGUF quantization with 4bit quantization would take up just under 4.5Gb in storage and a similar amount of RAM in runtime.

This model assumes that it will be triggered in a smartphone keyboard app or similar use case, when the user presses space on the preceding word, then string the sentence fragment will be passed to the model for next word prediction. It assumes it only wants to return words (not punctuation) and only in lower case.

Full Process of developing the model in Python is avaliable in the Model Development Jupyter Notebook. The finished prediction and model files are packaged up in the next_word_pred subdirectory.

References
https://www.analyticsvidhya.com/blog/2023/07/next-word-prediction-with-bidirectional-lstm/
https://www.analyticsvidhya.com/blog/2021/08/predict-the-next-word-of-your-text-using-long-short-term-memory-lstm/
https://medium.com/linagoralabs/next-word-prediction-a-complete-guide-d2e69a7a09e6


## Issues & Performance:
It became apparent during training that the model would massively overfit if allowed to run more than 10 epochs despite having drop out layers in the model. Thus an early stopping callback was implemented to prevent the model from running when the validation losses were increasing. The model ran for 8 epochs with a Crossentropy validation loss of 6.2668. Testing on a set of the last 100 records the test accuracy was found to be slightly less at 5.3297, giving a perplexity of 206 vs our word vocab list of 7299. 

When considering a smart phone keyboard whole word completion situation usually more than one word is suggested. I implemented the code to return the top 5 most likely suggestions by using top_k rather than argmax/softmax on the category probability output of the network. Looking at the accuracy of the model when considering suggesting 5 rather than one word, the accuracy of 31% is achieved. meaning one third of the time the model would give a correct suggestion within the first 5.



## Usage 

Requires Python Version 3.9.18
Package requirements are listed in the requirements.txt file in the root of the project
CUDA version 11.7 using in Windows Environment on 8Gb VRAM GPU


example usage:
```from next_word_pred import predict 
predictor = predict.LSTM_next_word_pred() 
    
predictor.predict("where ")
```
will return the top 5 most likely next words in a list of strings:
['are', 'do', 'you', 'is', 'will']




## Running Test cases using Pytest
I implemented a few test cases using pytest to test various functionalities.

Testing using Pytest
from the root directory of the project run 
`pytest tests\`


## follow up with transformers
I wanted to assess the transfer learning available from using a purpose built model such as GPT or bert to perform the same task. I also trialed using LLMs such as Mistral 7b, but it proved troublesome to fit into a reasonable VRAM buffer. Also for this task we would only access the base model's prediction power (and really benefit from the added complexity of instruction tuning/HLRF parts of the model development). So I selected GPT2 as it is much more compact and easier to run than later GPT or LLM models it is also better in general than bert at next token prediction and bert is trained using pre and post snippets of the sentence in a bi-directional manner as opposed to GPT which has a next token prediction training basis. 

I downloaded the model from the hugging face repository via the transformer package and used it to predict the top 5 word tokens and assessed its performance on the same test N-gram set that was used to assess the LSTM model. The pre-trained model as downloaded achieved 43.75% accuracy without any fine-tuning, which is just over 10% better than the original LSTM. This model shows promise for further exploration via fine-tuning and could be expanded to handle prediction of punctuation and casing due to the flexibility of its tokenizer vs the LSTM's tokenizer. You can see more details of the code in the model development in the GPT2_Model_development.ipynb notebook. The model was downloaded from hugging face in runtime by the transformer package and has not been altered so has not been saved.

I implemented it into a class within the next_word_predict package and wrote similar test cases to the LSTM.

example usage:
```from next_word_pred import predict 
predictor = predict_gpt2.GPT2_next_word_pred() 
    
predictor.predict("I will be back")
```

will return the list of word candidates 
`['in', 'soon', 'to', 'with', 'on']`