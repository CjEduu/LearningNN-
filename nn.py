""" This repo is just for learning purposes, I do not intend to make anything new.
    Just trying to make from scratch the things im learning to understand them deeper"""

import activations as acts
import cost_funcs as cfncs
import matrix_math as mat
from random import random


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
        
    def think(self,input:mat.Matrix)->float:
        """think...xd"""
        z = mat.mat_dot(input,self.ws)
        return z.data[0] + self.bias
    
    def __repr__(self)->str:
        ret = f"   WS    BS = {self.bias}\n"
        ret += f"{self.ws}\n"
        return ret 
      
    
    
class NN(object):
    #usar kwargs?¿
    def __init__(self,topology:list[int],activation:str,rand:bool = True, low:float = None, top:float = None)->None:
        """topology should be an array of unsigned ints describing the number of neurons of input,hidden and output, in that order
        activation refers to the activation function, available: sigmoid.
        weights and biases are generated with random values between low and top, random behaviour on by default"""
        self.expected_input_length = topology[0]
        self.act_func = acts.sigmoid 
        self.layers:list[list[Neuron]] = list()
        self.cost_func = cfncs.avg_sqr_err  # to do: have a module with more act_funcs, kinda like activations
        
        # initializate the nn
        for i in range(1,len(topology)):
            layer:list[Neuron] = list()
            for _ in range(topology[i]):
                layer.append(Neuron(topology[i-1],rand,low,top))
            self.layers.append(layer)
    
    def cost(self,input:mat.Matrix,expected:mat.Matrix)->float:
        return self.cost_func(self,input,expected)
    
    def forward(self,input:mat.Matrix)->mat.Matrix:
        assert(input.cols == self.expected_input_length)
        current_act = input
        for layer in self.layers:
            next_activation_data:list[float] = list()
            for neuron in layer:
                z = neuron.think(current_act)
                a = self.act_func(z)
                next_activation_data.append(a)
            current_act = mat.Matrix(1,len(layer),next_activation_data)
        return current_act
                
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
    print(nn)
    
    train_input = mat.Matrix(3,2,[1.0,1.0,2.0,3.0,0.0,0.0]) 
    train_expected = mat.Matrix(3,1,[1.0,1.0,1.0])
    print(train_input)
    print(train_expected)
    print(f"Cost = {nn.cost(train_input,train_expected)}")
    
if __name__ == "__main__":
    main()