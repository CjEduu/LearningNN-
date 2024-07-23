""" This repo is just for learning purposes, I do not intend to make anything new.
    Just trying to make from scratch the things im learning to understand them deeper
    TO DO: Add more cost_funcs"""


import activations as acts
import cost_funcs as cfncs
import matrix_math as mat
import random
import matplotlib.pyplot as plt

PLOT = True


class Perceptron(object):
    def __init__(self,n_weights:int,rand:bool = True,low:float = None,top:float = None)->None:
        """ z = ws * outputs_previous_layer + bias
            a = act_func(z), ex: sigmoid(z)
        """
        self.ws = mat.Matrix(n_weights,1,[mat.rand_float(low,top) for _ in range(n_weights)] if rand else [0 for _ in range(n_weights)])
        self.bias = mat.rand_float(low,top)
        
    def forward(self,input:mat.Matrix)->float:
        z = mat.mat_dot(input,self.ws).data[0] + self.bias
        return z
    
    def __repr__(self)->str:
        ret = f"   WS    BS = {self.bias}\n"
        ret += f"{self.ws}\n"
        return ret 
      
class Linear(object):
    def __init__(self,n_neurons,n_inputs,rand:bool = True,low=None,top = None)->None:
        self.expected_input_length = n_inputs
        self.neurons:list[Perceptron] = [Perceptron(n_inputs,rand,low,top) for _ in range(n_neurons)]
        self.zs:mat.Matrix = None
        
    def __repr__(self) -> str:
        ret = "LINEAR LAYER\n"
        for neuron in self.neurons:
            ret += 20*'-' + "\n"
            ret += f"{neuron.ws} | {neuron.bias} \n"

        return ret + '\n'
        
class MLP(object):
    def __init__(self,list_layers:list[Linear])->None:
        self.layers:list[Linear] = list_layers
    
    def cost(self,input:mat.Matrix,expected:mat.Matrix)->mat.Matrix:
        return self.cost_func(input,expected)
        
    def forward(self,input:mat.Matrix)->mat.Matrix:
        current = input
        for layer in self.layers:
            if isinstance(layer,acts.Activation):
                current = layer.forward(current)
            else:
                layer.zs = mat.Matrix(1,len(layer.neurons),[])
                for nidx,neuron in enumerate(layer.neurons):
                    layer.zs.data[nidx] = neuron.forward(current)
                current = layer.zs
                
        return current

    def train(self, epochs: int, rate: float, train_in: mat.Matrix, train_exp: mat.Matrix,plt_values:list,cost_func='sqr_err') -> None:
        self.cost_func = getattr(cfncs,cost_func)
        self.cost_func_der = getattr(cfncs,cost_func + "_der")
        
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
                
            if PLOT:plt_values.append(total_cost/train_in.rows)
            print(f"Epoch {epoch+1}/{epochs}, Average Cost: {total_cost / train_in.rows}")

    def backpropagate(self,input_training:mat.Matrix, expected_output: mat.Matrix, learning_rate: float) -> None:
        # Compute the error in the output layer
        output_layer = self.layers[-1]
        delta = mat.mat_hadamard(
            self.cost_func_der(output_layer.act, expected_output),
            output_layer.derivative(self.layers[-2].zs)
        )
        prev_layer_acts = None
        # print(f"COST\n {self.cost_func_der(output_layer.acts, expected_output)}")
        # print(f"ACT \n{self.act_func_der(output_layer.zs)}")
        # print(f"DELTA OUT\n{delta}")
        
        # Backpropagate the error
        
        
        for l in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[l]

            # Compute gradients
            if(isinstance(layer,acts.Activation)):
                if l != 1:
                    prev_layer_acts = self.layers[l-2].act
                else:
                    prev_layer_acts = input_training
                continue
            
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
                    self.layers[l+1].derivative((self.layers[l-2].zs))
                )
             
    def __repr__(self)->str:
        ret = ""
        for layer in self.layers:
            if isinstance(layer,acts.Activation):
                continue
            ret += f"{layer}"
        return ret
    

def test()->None:
    
    #----DEFINE THE MLP----------
    topology = [4,5,9]
    activation = 'sigmoid'
    cost_func = 'sqr_err'
    epochs = 10000
    learning_rate = 0.1
    plt_values = []
    
    #MLP------------------------------
    
    nn = MLP(topology)
    
    numeros = [random.randint(0,1) for _ in range(128)]
    paridad = []

    contador = 0
    for _ in range(16):
        unos = 0
        for _ in range(8):
            if numeros[contador] == 1:
                unos += 1 
            contador += 1
        paridad.append(int(unos%2==1))
     
    train_input = mat.Matrix(16,8,numeros) 
    train_expected = mat.Matrix(16,1,paridad)
    
    nn.train(epochs,learning_rate,train_input,train_expected,plt_values,cost_func=cost_func,act_func=activation)
    print(nn)
    
    print(nn.forward(mat.Matrix(1,8,[1,1,1,1,1,0,0,0])))
    
    #------------------------------------
    
    #----------------------------
    fig, ax = plt.subplots(layout = 'constrained')
    fig.canvas.manager.set_window_title('training a XOR model')
    fig.suptitle("XOR")
    
    
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
    
    ax.plot(range(epochs),plt_values,label = "epochs")
    plt.show()
         
         
def main():
    epochs = 10000
    rate = 0.1
    x = Linear(2,2)
    z = Linear(1,2)
    a1 = acts.Sigmoid()
    a3 = acts.Sigmoid()
    layers = [x,a1,z,a3]
    mlp = MLP(layers)
    
    training_input = mat.Matrix(4,2,[0,0,1,0,0,1,1,1])
    expected_output = mat.Matrix(4,1,[0,1,1,1])
    
    mlp.train(epochs,rate,training_input,expected_output,[])
    print(mlp)
    
    #Truth table
    for i in range(2):
        for j in range(2):
            print(f" {i} | {j} = {mlp.forward(mat.Matrix(1,2,[i,j]))}")
    
    
if __name__ == "__main__":
    main()