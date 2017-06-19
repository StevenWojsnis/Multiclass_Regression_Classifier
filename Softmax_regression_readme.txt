###################

# ReadMe Document #

###################


*NOTE* Softmax_Derivation.pdf shows the derivation of the softmax function, while Loss_Function_Derivation.pdf
shows the derivation of the loss function as a whole, which uses the softmax derivation.

Instructions on how to run:

To run this program with cifar-10, you need to download the data from the link provided in this project description..
You then must change the paths in the code to reflect the paths to the various batches on your machine.
(This doesn't need to be done if testing iris or mnist)

To run the program in general, you must navigate to wherever the program is located, and then run the following
command:

python Softmax_regression.py 0
	OR
python Softmax_regression.py 1

(0 corresponds to gradient descent, 1 corresponds to quasi-newton)
You will then be asked to enter either a 0, 1, or 2 to choose your dataset. Upon selection, the training will begin.
