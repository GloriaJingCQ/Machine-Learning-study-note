- Explanation of gradient, and why it's important:
We have initial input a,b. We have final input g. We want to know the gradient(derivatives), basicaly means we want to know how the change of a,b will influence g.
The function sensitivity regarding to the slightly change if a and b (Go back to the definition of derivative instead of solving derive derivative functions). How the function responds to the slightly change of our input, ie x
(f(x+h) - f(x))/h    (the change of my function)/(the change of my input)

- BackPropagation: 
We're going through layers by layers to reversely compute the the gradient regarding
output and the intermediate nodes. The derivative of L(output) with respect to each layer input(intermediate nodes).

? In neural net, the loss function is the output L.
Chain Rule:
We're not really interested in the derivative of L with respect to original data, since orignal data are fixed. We're more interested in the weights, because we're going to change them based on gradient.
You know that d has direct impact on L. You also know c has direct impact on d. You should feel somehow there is a thing between c and L. That thing is what we want to get now.
Multiplication of local derivatives based on chain rule

Loss Function:
- Use a single value to somehow measure the whole performance of the neural network. -> We use loss function to do so! The more off, the larger loss we will have


Implementation of neural network:
- initialization of neuron class:




- Some notes:
- Pytorch tensor also has .data/ .grad. If you do o.data.item(), you get the number getting rid of the tensor.
