<h1>Neural Network for Digit Recognition</h1>
This is a fun beginner project creating a basic neural network from scratch, by only using Math and NumPy.
The Neural Network is trained on the popular MNIST dataset which contains gray-scale images of hand-drawn digits, with each 784 pixels (28x28).
The training model utilizes a simple architecture with ReLU and softmax activation functions, predicting the correct digit based on the pixel values and continuously improving its accuracy through iterative training.<br><br>


![image](https://github.com/Salchegger/Simple-Neural-Network-for-Handwritten-Digit-Recognition/assets/167821529/8a15e151-f150-46d4-b5f6-b5a2392946fe)
<br>

<h1>Key Finding: The accuracy of the handwritten digit prediction on the training data (83.75%) is lower than on the data tested (84.75%)!
This indicates that there're meaningful differences in the datasets (e.g. the test set is probably less diverse and therefore easier for the model to predict).</h1>

<h2>Table of Contents</h2>
<ul>
  <li>Introduction</li>
  <li>Data Source</li>
  <li>Architecture</li>
  <li>Algorithm</li>
  <li>Results</li>
</ul>

<h2>Introduction</h2>
Have you ever wondered how machines can read hand-drawn digits super-fast and pretty accurately?<br>
Well me neither, till like a week ago when my boyfriend explained some math to me and I looked at both of our handwritten numbers and started to think about how well or even if a machine could read both of our handwritings correctly. - His' obviously being worse than mine! ;) <br>
Somehow, I couldn't stop thinking about it and I asked my boyfriend (who is a data scientist) if there are any easy beginner-friendly machine learning projects that I could wrap my head around to understand how a computer can be trained to recognize different digits. <br>
Of course, "easy" was not the best word to describe a first-ever attempt to write a code like that, but still being stubborn as I am sometimes, I tried to solve the suggested problem of creating a neural network to predict hand-drawn digits.<br><br>
I started by watching a <a href="https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1667s">YouTube video</a> to get familiar with the math and code that I needed. Then it was time to open my JupyterLab notebook and begin writing down the code trying to understand how each line of code and function interferes with another, adding a ton of sidenotes so I wouldn't get confused. Unfortunately, the initial code presented some error issues (even after making sure that I corrected the ones from the video itself). This made it definitely more complicated than expected but very realistic as I understand now that writing a code is not always as easy as it might seem. <br>
After several hours of work tho, I was able to debug the code and suddenly it ran smoothly, making me very happy and proud of myself.
I hope by uploading this file, I can help others to try out their first projects in machine learning and have fun with coding as well!<br> 

 
<h2>Data Source</h2>
I found both of my MNIST datasets on Kaggle, each consisting of hand-drawn gray-scale images ranging from 0-9. 
Both sets are extremely clean and don't need any modification. Here is more information about the data, taken directly from <a href="https://www.kaggle.com"> Kaggle </a>.<br> <br>

> The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine. <br>Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. <br>The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.<br> <br>
Train Data:
https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv <br>
Test Data:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv


<h2>Architecture</h2>
The neural network's architecture includes an input layer, two hidden layers, and one output layer. <br>
The input layer consists of 784 neurons corresponding to each pixel in the image. Each of the two hidden layers contains 10 neurons, both using the ReLU activation function, while the output layer employs the softmax activation function.<br> <br>

<ol>
  <li><strong>Rectified Linear Unit</strong></li>
    The ReLU activation function is used in the hidden layers to introduce non-linearity to the network by replacing negative values with zero, effectively removing linearity constraints     and enabling the network to capture complex patterns in the data more effectively.<br><br>
  <li><strong>Softmax</strong></li>
  The softmax function transforms the raw output values from the previous layer into a probability distribution over multiple classes. This allows us to interpret the output as the         likelihood of each class being the correct prediction. Softmax ensures that the sum of all probabilities adds up to 1, making it suitable for multi-class classification tasks, like       digit recognition.
</ol>

<h2>Algorithm</h2>
The model is trained using gradient descent optimization. During training, the network parameters (weights and biases) are initialized randomly and updated iteratively to minimize the 
loss function, which measures the difference between predicted and actual digit labels. Backpropagation is utilized to compute gradients efficiently, allowing for the adjustment of 
parameters to improve model performance.


<h2>Results</h2>
After training, the model's performance is evaluated on a new, separate <a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv">test dataset</a>. The accuracy of the neural network on the new data is 84.75%, which is pretty good for such a basic model.<br> <br>

> As previously mentioned, the accuracy of the predicted initial training data is 83.75% which is a little lower than the accuracy on the test data set. This is not very common but
suggests that both datasets are to some degree different from another. The training data might be more complex and diverse whereas the test data might present a lack in those features.

<br>
In conclusion this hands-on beginner project was a lot of fun for me and I'm inspired by the insights that I have gained and the progress I've made in understanding neural networks and machine learning principles, to keep practicing and dive deeper into advanced concepts in order to be able to tackle more complex machine learning challenges in the future.<br> <br>

Thanks for checking out my project!
