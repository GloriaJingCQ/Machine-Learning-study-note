Study note after watching Andrej's video: https://www.youtube.com/watch?v=VMj-3S1tku0&t=6666s. Amazing video!

1. **Explanation of gradient, and why it's important**:
- We have initial input a,b. We have final input g. We want to know the gradient(derivatives), basicaly means we want to know how the change of a,b will influence g.
The function sensitivity regarding to the slightly change if a and b (Go back to the definition of derivative instead of solving derive derivative functions). How the function responds to the slightly change of our input, ie x
(f(x+h) - f(x))/h    (the change of my function)/(the change of my input)

2. **BackPropagation Intuiton**: 
- We're going through layers by layers to reversely compute the the gradient regarding
output and the intermediate nodes. The derivative of L(output) with respect to each layer input(intermediate nodes).

- Chain Rule:
We're not really interested in the derivative of L with respect to original data, since orignal data are fixed. We're more interested in the weights, because we're going to change them based on gradient.
You know that d has direct impact on L. You also know c has direct impact on d. You should feel somehow there is a thing between c and L. That thing is what we want to get now.
Multiplication of local derivatives based on chain rule

- Loss Function:
Use a single value to somehow measure the whole performance of the neural network. -> We use loss function to do so! The more off, the larger loss we will have


3. **Implementation of Neural Network**:
   
   (Initialization first(i.e. n = ...), having w and b prepared. Then do n(x), to return neuron value or a layer(a set of neuron values))
- **Initialization of Neuron Class**:
  ![image](https://github.com/GloriaJingCQ/CPSC-340-note/assets/87431812/67f6fe96-1836-43fd-aa28-60119d67ca6d)
  
  Note:
  - By initializing one neuron instance, we have a set(list) of w (#d) and a bias value for constructing the neuron. The parameter initialization takes in is nin, which is just how many input or for an example x_i, it's dimentionality. 
  - The value of the neuron is sum(w*x) + bias
  - The call function of a neuron takes an input x (one example with d dimensions, a vector), and pairwise mutiply w. The output is a neuron value.
  - n is the neuron instance, n(x) means you did the call function of the neuron instance

- **Initialization of Layer Class**
  ![image](https://github.com/GloriaJingCQ/CPSC-340-note/assets/87431812/c1fb92bb-1721-4c26-8b5d-dead277ce641)

  Note:
  - To initialize a layer, it takes in nin(number of input, d), nout(number of output, how many neurons you wanna have). The initialization is basically gives you a list of nueron initialization(a list of w(w is also a list) and b to construct future neurons). 
  - call function basically for every neuron initialization(different w vector and b), call it with same input x vector. Returns a list of neuron values.
  - the relationship between nin and x: nin is the number of elements in call function's input x vector.

- **Initialization of MLP(multiple layers)**
  ![image](https://github.com/GloriaJingCQ/CPSC-340-note/assets/87431812/b36ee09e-4575-4827-809e-cc8ffdb15aed)

  Note:
  - We take two parameters to initialize. nin: the number of input(original input layer), nouts(a list. i.e. [4,4,1], means this network we're building has 3 output layers(4 neurons one layer, 4..., 1 neurons one layer)).
  - The initialization gives a list of layer initialization(a set of w list and b). each layer list takes 2 parameters, #input, #output. i.e. MLP(3, [4,4,1]), layer initialization list is [layer(3,4), layer(4,4), layer(4,1)], layer(3,4) gives us a 4 w vectors and 4 b, each w vector has 3 elements.
  - The call function, we do for layer call function each layer, remember that layer's call function gives us a list of neuron values. We use the output list of neuron values as next layer's input vector x, using next layer's initialization(the right w and b).
  - Returns the final layer(a set of neuron values)

Some notes:
- Pytorch tensor also has .data/ .grad. If you do o.data.item(), you get the number getting rid of the tensor.
