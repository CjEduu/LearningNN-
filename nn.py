""" This repo is just for learning purposes, I do not intend to make anything new.
    Just trying to make from scratch the things im learning to understand them deeper
    TO DO: Add more cost_funcs"""


import activations as acts
import cost_funcs as cfncs
import matrix_math as mat
import matplotlib.pyplot as plt


class Neuron(object):
    def __init__(self,n_weights:int,rand:bool = True,low:float = None,top:float = None)->None:
        """ z = ws * outputs_previous_layer + bias
            a = act_func(z), ex: sigmoid(z)
        """
        self.ws = mat.Matrix(n_weights,1,[mat.rand_float(low,top) for _ in range(n_weights)] if rand else [0 for _ in range(n_weights)])
        self.bias = mat.rand_float(low,top)
        self.cost_func = None
        
    def think(self,input:mat.Matrix)->float:
        """think...xd"""
        z = mat.mat_dot(input,self.ws).data[0] + self.bias
        return z
    
    def __repr__(self)->str:
        ret = f"   WS    BS = {self.bias}\n"
        ret += f"{self.ws}\n"
        return ret 
      
class Layer(object):
    def __init__(self,neurons:list[Neuron],act_func)->None:
        self.neurons:list[Neuron] = neurons
        self.act_func = act_func
        self.zs:mat.Matrix = None
        self.acts:mat.Matrix = None
    
class NN(object):
    def __init__(self,topology:list[int],rand:bool = True, low:float = None, top:float = None)->None:
        """topology should be an array of unsigned ints describing the number of neurons of input,hidden and output, in that order
        activation refers to the activation function, available: sigmoid,ReLu,TanH.
        weights and biases are generated with random values between low and top, random behaviour on by default
        """
        
        self.expected_input_length = topology[0]
        self.layers:list[Layer] = list()
        self.act_func = None
        
        # initializate the nn
        for i in range(1,len(topology)):
            neurons:list[Neuron] = list()
            for _ in range(topology[i]):
                neurons.append(Neuron(topology[i-1],rand,low,top))
            layer = Layer(neurons,self.act_func)
            self.layers.append(layer)
            
    
    def cost(self,input:mat.Matrix,expected:mat.Matrix)->mat.Matrix:
        return self.cost_func(input,expected)
        
    def forward(self,input:mat.Matrix)->mat.Matrix:
        assert(input.cols == self.expected_input_length)
        
        current_act = input
        for layer in self.layers:
            layer.zs = mat.Matrix(1,len(layer.neurons),[])
            
            for nidx,neuron in enumerate(layer.neurons):
                layer.zs.data[nidx] = neuron.think(current_act)

            layer.acts = acts.sigmoid(layer.zs)    
            current_act = layer.acts
            
        return current_act

    def train(self, epochs: int, rate: float, train_in: mat.Matrix, train_exp: mat.Matrix,plt_values:list,cost_func='sqr_err',act_func='sigmoid') -> None:
        
        self.cost_func = getattr(cfncs,cost_func)
        cost_func_der = cost_func + '_der'
        self.cost_func_der = getattr(cfncs,cost_func_der) 
        
        self.act_func = getattr(acts,act_func)
        act_func_der = act_func + '_der'
        self.act_func_der = getattr(acts,act_func_der)  
        
        
        for epoch in range(epochs):
            total_cost = 0
            for i in range(train_in.rows):
                # Forward pass
                input_sample = mat.mat_row(train_in,i)
                output = self.forward(input_sample)

                # Compute cost
                expected_output = mat.Matrix(1, train_exp.cols, train_exp.data[i*train_exp.cols:(i+1)*train_exp.cols])
                cost = self.cost(output, expected_output)
                total_cost += sum(cost.data)

                # Backward pass
                self.backpropagate(input_sample,expected_output, rate)
                
            plt_values.append(total_cost/train_in.rows)
            print(f"Epoch {epoch+1}/{epochs}, Average Cost: {total_cost / train_in.rows}")

    def backpropagate(self,input_training:mat.Matrix, expected_output: mat.Matrix, learning_rate: float) -> None:
        # Compute the error in the output layer
        output_layer = self.layers[-1]
        delta = mat.mat_hadamard(
            self.cost_func_der(output_layer.acts, expected_output),
            self.act_func_der(output_layer.zs)
        )
        # print(f"COST\n {self.cost_func_der(output_layer.acts, expected_output)}")
        # print(f"ACT \n{self.act_func_der(output_layer.zs)}")
        # print(f"DELTA OUT\n{delta}")
        
        # Backpropagate the error
        for l in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[l]

            # Compute gradients
            if l > 0:
                prev_layer_acts = self.layers[l-1].acts
            else:
                prev_layer_acts = input_training

            for n, neuron in enumerate(layer.neurons):
                # Update weights
                for w in range(neuron.ws.rows):
                    neuron.ws.data[w] -= learning_rate * delta.data[n] * prev_layer_acts.data[w]

                # Update bias
                neuron.bias -= learning_rate * delta.data[n]

            # Compute delta for the previous layer
            if l > 0:
                weights = mat.Matrix(len(layer.neurons), layer.neurons[0].ws.rows, [])
                for n, neuron in enumerate(layer.neurons):
                    weights.data[n*weights.cols:(n+1)*weights.cols] = neuron.ws.data

                weights_T = weights.T()
                delta_T = delta.T()

                # print(f"DELTA\n{delta}")
                # print(f"W\n{weights_T}")
                # print(f"ZS\n{self.layers[l-1].zs}")
                # print(f"DOT\n{mat.mat_dot(weights_T,delta_T)}")
                # Compute delta for the previous layer
                
                delta = mat.mat_hadamard(
                    mat.mat_dot(weights_T,delta_T).T(),
                    self.act_func_der(self.layers[l-1].zs)
                )
                         
    def __repr__(self)->str:
        ret = ""
        
        for i,layer in enumerate(self.layers):
            ret += "-"*20 + "\n" + f"Layer {i+1}:\n" + "-"*20 + "\n" + "\nWS\n" 
            for i in range(len(layer.neurons[0].ws.data)//layer.neurons[0].ws.cols):
                ret += '['
                for neuron in layer.neurons:
                    ret += f" {neuron.ws.data[i]:.2f} "
                ret += ']\n'

            ret += "\nBS\n["
            for neuron in layer.neurons:
                ret += f" {neuron.bias:.2f} "
            ret += "]\n"
            
            ret += "\nZS\n"
            ret += f"{layer.zs}"
            
            ret += "\nACTS\n"
            ret += f"{layer.acts}\n"
        ret += "-"*20
        return ret
    

def main()->None:
    
    #----DEFINE THE NN----------
    topology = [2,2,1]
    activation = 'tanh'
    cost_func = 'sqr_err'
    epochs = 10000
    learning_rate = 0.1
    #----------------------------
    fig, ax = plt.subplots(layout = 'constrained')
    fig.canvas.manager.set_window_title('training a XOR model')
    fig.suptitle("XOR")
    
    plt_values = []
    
    #Set labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cost")
    
    
    #Set scale
    ax.set_ylim(0,None)
    ax.set_xlim(0,epochs)
    
    
    ax.set_aspect('auto')
    
    # Non-necessary
    ax.set_xscale("linear")
    ax.set_yscale('linear')
    
    #NN------------------------------
    
    nn = NN(topology,activation)
    train_input = mat.Matrix(4,2,[0.0,0.0,
                                  1.0,0.0,
                                  0.0,1.0,
                                  1.0,1.0]) 
    train_expected = mat.Matrix(4,1,[0.0,
                                     1.0,
                                     1.0,
                                     1.0])
    
    nn.train(epochs,learning_rate,train_input,train_expected,plt_values,cost_func=cost_func,act_func=activation)
    print(nn)
    
    #------------------------------------
    
    ax.plot(range(epochs),plt_values,label = "epochs")
    plt.show()
         
    
if __name__ == "__main__":
    main()