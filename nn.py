""" This repo is just for learning purposes, I do not intend to make anything new.
    Just trying to make from scratch the things im learning to understand them deeper"""

import activations as acts
import cost_funcs as cfncs
import matrix_math as mat
from random import random


DEBUG = False

def rand_float(low=None,top=None)->float:
    """If no args passed, generates between 0 and 1"""
    if low == None or top == None:
        rand = random()    
    else:
        rand = low + random() * (top - low)
    return rand

class Neuron(object):
    def __init__(self,n_weights:int,rand:bool = True,low:float = None,top:float = None)->None:
        """ z = ws * outputs_previous_layer + bias
            a = act_func(z), ex: sigmoid(z)
        """
        self.ws = mat.Matrix(n_weights,1,[rand_float(low,top) for _ in range(n_weights)] if rand else [0 for _ in range(n_weights)])
        self.bias = rand_float(low,top)
        self.z = None
        self.a = None
        
    def think(self,input:mat.Matrix,act_func)->float:
        """think...xd"""
        self.z = mat.mat_dot(input,self.ws).data[0] + self.bias
        self.a = act_func(self.z) 
        return self.a
    
    def __repr__(self)->str:
        ret = f"   WS    BS = {self.bias}\n"
        ret += f"{self.ws}\n"
        return ret 
      
    
    
class NN(object):
    def __init__(self,topology:list[int],rand:bool = True, low:float = None, top:float = None)->None:
        """topology should be an array of unsigned ints describing the number of neurons of input,hidden and output, in that order
        activation refers to the activation function, available: sigmoid.
        weights and biases are generated with random values between low and top, random behaviour on by default"""
        
        self.expected_input_length = topology[0]
        self.layers:list[list[Neuron]] = list()
        self.act_func = acts.sigmoid
        self.cost_func = cfncs.sqr_err # to do: have a module with more act_funcs, kinda like activations
        self.cost_func_der = cfncs.sqr_err_der
        self.act_func_der = acts.sigmoid_der
        
        # initializate the nn
        for i in range(1,len(topology)):
            layer:list[Neuron] = list()
            for _ in range(topology[i]):
                layer.append(Neuron(topology[i-1],rand,low,top))
            self.layers.append(layer)
    
    def cost(self,input:mat.Matrix,expected:mat.Matrix)->float:
        return self.cost_func(input,expected)

    
    def train(self,epochs:int,rate:float,train_in:mat.Matrix,train_exp:mat.Matrix,cost_func = cfncs.sqr_err,act_func = acts.sigmoid)->None:
        pass
        
    
    def forward(self,input:mat.Matrix)->mat.Matrix:
        assert(input.cols == self.expected_input_length)
        
        current_act = input
        for layer in self.layers:
            next_activation_data:list[float] = list()
            for neuron in layer:
                a = neuron.think(current_act,self.act_func)
                next_activation_data.append(a)
                
            current_act = mat.Matrix(1,len(layer),next_activation_data)
        return current_act
    
    def backprop(self,input:mat.Matrix,expected:mat.Matrix)->None:
        output = self.forward(input)
        
        # first cost derivative 
        actual_a = self.cost_func_der(output,expected)
        
        for layer in reversed(self.layers):
            for neuron in layer:
                pass
        
        
                
    def __repr__(self)->str:
        ret = ""
        
        for i,layer in enumerate(self.layers):
            ret += "-"*20 + "\n" + f"Layer {i+1}:\n" + "-"*20 + "\n" + "\nWS\n" 
            for i in range(len(layer[0].ws.data)//layer[0].ws.cols):
                ret += '['
                for neuron in layer:
                    ret += f" {neuron.ws.data[i]:.2f} "
                ret += ']\n'

            ret += "\nBS\n["
            for neuron in layer:
                ret += f" {neuron.bias:.2f} "
            ret += "]\n\n"
            
        return ret[:-1]
    

def main()->None:
    topology = [2,2,1]
    activation = "sigmoid"
    
    nn = NN(topology,activation)
    train_input = mat.Matrix(4,2,[0.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0]) 
    train_expected = mat.Matrix(4,1,[0.0,1.0,1.0,1.0])
    print(nn)
    print(nn.cost(nn.forward(mat.mat_row(train_input,3)),mat.mat_row(train_expected,2)))
if __name__ == "__main__":
    main()