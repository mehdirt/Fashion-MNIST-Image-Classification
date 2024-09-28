# Image Classification on Fashion-MNIST Dataset using PyTorch
A deep learning project to classify images from the Fashion-MNIST dataset using a feedforward neural network implemented in PyTorch. The project also includes visualization tools and demonstrates handling custom datasets.

## Description
This project was created to explore and understand the fundamentals of PyTorch through the implementation of a simple neural network for image classification. The model is trained on the Fashion-MNIST dataset, which consists of 60,000 grayscale images of clothing items (28x28 pixels) categorized into 10 distinct classes. The project also supports custom datasets using a CustomImageDataset class.

After training for 25 epochs, the model achieved an accuracy of 87.1% on the test set. The project also visualizes the dataset samples and evaluates model performance using a confusion matrix.

## Dataset
**Fashion-MNIST:** A dataset of clothing images consisting of:
**Training set:** 60,000 examples
**Test set:** 10,000 examples
Each image is a **28x28 grayscale** image, and each label belongs to one of 10 classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
0. Bag
10. Ankle boot

## Model Architecture
- Input Layer: 784 neurons (28x28 pixels flattened)
- Hidden Layer 1: 128 neurons (fully connected)
- ReLU Activation
- Hidden Layer 2: 64 neurons (fully connected)
- ReLU Activation
- Output Layer: 10 neurons (for 10 clothing categories)
- Activation: Softmax function for classification

## Hyperparameters
- Learning Rate: 0.01
- Batch Size: 64
- Epochs: 25
- Optimizer: SGD (Stochastic Gradient Descent)
- Loss Function: Cross-Entropy Loss

## Dependencies
- Python 3.x
- PyTorch
- Torchvision
- Scikit-learn (for confusion matrix)

## Results
- Accuracy: The model achieved 87.1% accuracy after 25 epochs.
- The plot below shows how the loss function decreases during training:
- Confusion matrix visualizing model predictions:

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact Information
Feel free to reach out via the following platforms:

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.hopp.bio/mehdirt)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahdirafati680@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdi-rafati-97420a197/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mehdirt)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mahdirafati)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/itsmehdirt)
