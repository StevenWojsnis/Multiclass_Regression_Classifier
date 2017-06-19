# Multiclass_Regression_Classifier
Uses softmax regression to perform multiclass classification. Written by Steven Wojsnis


A multiclass classifier built from scratch. Uses multinomial regression (softmax regression) for classification. This classifier is
tested on various datasets, included the iris dataset, the MNIST dataset (digits), and the CIFAR-10 dataset. (Note that the CIFAR-10
dataset must be downloaded from the above link and the file paths must be changed in the code, if you wish to test on CIFAR-10).

This project was written in python using numpy. SKlearn was used for PCA to reduce the dimensionality (only really necessary for the
CIFAR-10 dataset, as it contains a relatively large number of features). It should be noted that PCA is currently only used for CIFAR-10,
and by default the dimensionality of a CIFAR-10 sample is reduced to 100.

The model can be trained with two different methods, for the sake of comparison. These two methods are gradient descent and an
implementation of a quasi-Newthon method, called the bfgs method. This bfgs method was taken from the scipy.optimize library.

This project takes advantage of numpy's precompiled code for iteration. This is often referred to as "vectorization." In short, this
means that regular python loops weren't used for iteration, as the runtime would be too long. Instead, all iterative calculations
were performed using matrix operations, such as numpy's dot product. This dramatically decreases runtime.

Different runtime options (such as model-training method and dataset are selected via command line arguments / runtime inputs. See the 
Softmax_regression_readme.txt file for instructions on how to run the program and choose these different options.

After training the model the program then tests the model on a section of the dataset. The accuracy of that test is shown as output.
Additionally, plots showing the calculated loss over the training process are generated and shown.

