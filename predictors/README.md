# Predictors

There are several predictors that we have developed for various testing of performance:
- [Convolutional](convolutional-predictor)
- [Fully connected](#fully-connected)
- [LSTM](#lstm)
- [Autoencoder](#autoencoder)

## Convolutional
It is a convolutional neural network with batch normalization and max pooling, and two fully connected layer at the end for 
classification. It uses a cross entropy loss. It is the best performing predictor at the moment for all classes of agents.

To run it:
```
python run_pred.py --model_class conv
```

## Fully connected
A five layers, fully connected network. It has a higher number of parameters than its convolutional counterpart, and it 
does not achieve great results in accuracy. As the convolutional predictor, it uses a cross-entropy loss.

To run it:
```
python run_pred.py --model_class dense
```

## LSTM
A LSTM network, that tries to take advantage of the sequentiality of the inputs. Differently from the previous two, an entire
episode is loaded instead of a single observation.

To run it:
```
python run_pred.py --model_class lstm
```

## Autoencoder
We use a sparse autoencoder to learn an encoded vector from the observations. First we train extensively the autoencoder by 
running
```
python run_pred.py --model_class autoencoder
```
When the autoencoder is trained to convergence, we move the model to the directory of `encoder_pred` and we load only the 
encoder, disregarding the decoder. Instead, we train a predictor that tries to classify the encoded vector. The perform comance
of this method are not as satisfying as training the complete network. 

To train the predictor, run:
```
python run_pred.py --model_class encoder_pred
```
