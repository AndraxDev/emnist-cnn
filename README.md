# MNIST-like datasets classifier
Convolutional neural network to recognize symbols using a MNIST-like datasets

## Purpose
Recognize handwritten letters and digits

## Accuracy

```
92%
```

## Requirements

```
PyTorch, TorchVision, Numpy (1.23.1), matplotlib.pyplot
```

## Requirements for datasets

28x28 grayscaled images

## Tested datasets

- <a href = "https://en.wikipedia.org/wiki/MNIST_database">MNIST</a>
- <a href = "https://www.nist.gov/itl/products-and-services/emnist-dataset">EMNIST</a>

## Run

```
python3 cnn.py
```

## Runtime options

```python
# Comment or uncomment necessary functions
train(model, model_path)
test(model_path)
test_all(model_path)
```

> **Warning**
>
> Training NN or testing  all images (function test_all()) may take very long time.


## Output example
![Screenshot_11](https://user-images.githubusercontent.com/43646136/213566800-8456364a-79e6-41ca-932a-5a33b61bfa02.png)

![Screenshot_12](https://user-images.githubusercontent.com/43646136/213566825-073a97c0-3e36-44cc-b53f-21681d28fced.png)
