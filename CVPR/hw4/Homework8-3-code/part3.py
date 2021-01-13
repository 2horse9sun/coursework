import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F  # useful stateless functions

from part1 import device, dtype, loader_train, print_every, loader_val


'''
------------------------------------------------------------------------------------------------------------------------
Part III. PyTorch Module API
Barebone PyTorch requires that we track all the parameter tensors by hand. This is fine for small networks 
with a few tensors, but it would be extremely inconvenient and error-prone to track tens or hundreds of 
tensors in larger networks.

PyTorch provides the nn.Module API for you to define arbitrary network architectures, while tracking every 
learnable parameters for you. In Part II, we implemented SGD ourselves. PyTorch also provides the torch.optim 
package that implements all the common optimizers, such as RMSProp, Adagrad, and Adam. It even supports 
approximate second-order methods like L-BFGS! You can refer to the doc for the exact specifications of each optimizer.

To use the Module API, follow the steps below:

1. Subclass nn.Module. Give your network class an intuitive name like TwoLayerFC.

2. In the constructor __init__(), define all the layers you need as class attributes. Layer objects 
like nn.Linear and nn.Conv2d are themselves nn.Module subclasses and contain learnable parameters, 
so that you don't have to instantiate the raw tensors yourself. nn.Module will track these internal 
parameters for you. Refer to the doc to learn more about the dozens of builtin layers. Warning: don't 
forget to call the super().__init__() first!

3. In the forward() method, define the connectivity of your network. You should use the attributes 
defined in __init__ as function calls that take tensor as input and output the "transformed" tensor. 
Do not create any new layers with learnable parameters in forward()! All of them must be declared
 upfront in __init__.

After you define your Module subclass, you can instantiate it as an object and call it just like 
the NN forward function in part II.

Module API: Two-Layer Network
Here is a concrete example of a 2-layer fully connected network:
'''

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # forward always defines connectivity
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores


def test_TwoLayerFC():
    input_size = 50
    x = torch.zeros((64, input_size), dtype=dtype)  # minibatch size 64, feature dimension 50
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]

test_TwoLayerFC()

'''
Module API: Three-Layer ConvNet
It's your turn to implement a 3-layer ConvNet followed by a fully connected layer.
 The network architecture should be the same as in Part II:

1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2
2. ReLU
3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
4. ReLU
5. Fully-connected layer to num_classes classes
You should initialize the weight matrices of the model using the Kaiming normal 
initialization method.

HINT: http://pytorch.org/docs/stable/nn.html#conv2d

After you implement the three-layer ConvNet, the test_ThreeLayerConvNet function will 
run your implementation; it should print (64, 10) for the shape of the output scores.
'''


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        self.conv_1 = nn.Conv2d(in_channel, channel_1, (5,5), padding=2)
        nn.init.kaiming_normal_(self.conv_1.weight)
        self.conv_2 = nn.Conv2d(channel_1, channel_2, (3,3), padding=1)
        nn.init.kaiming_normal_(self.conv_2.weight)
        self.fc = nn.Linear(channel_2*32*32,num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = flatten(x)
        scores = self.fc(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores


def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


test_ThreeLayerConvNet()

'''
Module API: Check Accuracy
Given the validation or test set, we can check the classification accuracy of a neural network.

This version is slightly different from the one in part II. You don't manually pass in the parameters anymore.
'''

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


'''
Module API: Training Loop
We also use a slightly different training loop. Rather than updating the values of the weights ourselves,
we use an Optimizer object from the torch.optim package, which abstract the notion of an optimization
algorithm and provides implementations of most of the algorithms commonly used to optimize neural networks.
'''


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()

'''
Module API: Train a Two-Layer Network
Now we are ready to run the training loop. In contrast to part II, we don't explicitly allocate parameter tensors anymore.

Simply pass the input size, hidden layer size, and number of classes (i.e. output size) to the constructor of TwoLayerFC.

You also need to define an optimizer that tracks all the learnable parameters inside TwoLayerFC.

You don't need to tune any hyperparameters, but you should see model accuracies above 40% after training for one epoch.
'''

hidden_layer_size = 4000
learning_rate = 1e-2
model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_part34(model, optimizer)

'''
Module API: Train a Three-Layer ConvNet
You should now use the Module API to train a three-layer ConvNet on CIFAR. 
This should look very similar to training the two-layer network! You don't need to 
tune any hyperparameters, but you should achieve above above 45% after training for one epoch.

You should train the model using stochastic gradient descent without momentum.
'''

learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None
################################################################################
# TODO: Instantiate your ThreeLayerConvNet model and a corresponding optimizer #
################################################################################
model = nn.Sequential(
    nn.Conv2d(3,channel_1,(5,5),padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1,channel_2,(3,3),padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2*32*32,10)
)
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,nesterov=True)
################################################################################
#                                 END OF YOUR CODE
################################################################################

train_part34(model, optimizer)











