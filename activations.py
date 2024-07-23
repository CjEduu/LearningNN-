"""if input is a matrix, returns a element-wise-activation-applied matrix"""

import matrix_math as mat
from math import exp


def sigmoid(x:float|mat.Matrix)->float|mat.Matrix:
    if(isinstance(x,float)):
        return 1/(1+exp(-x))
    elif(isinstance(x,mat.Matrix)):
        res = mat.Matrix(x.rows,x.cols,[1/(1+exp(-a)) for a in x.data])
        return res
    else:
        print("Input should be float|matrix")
        return None
    
def sigmoid_der(x:float|mat.Matrix)->float|mat.Matrix:
    if(isinstance(x,float)):
        return sigmoid(x) * (1 - sigmoid(x))
    elif(isinstance(x,mat.Matrix)):
        res = mat.Matrix(x.rows,x.cols,[sigmoid(a) * (1 - sigmoid(a)) for a in x.data])
        return res
    else:
        print("Input should be float|matrix")
        return None

def tanh(x:float|mat.Matrix)->float|mat.Matrix:
    if(isinstance(x,float)):
        e = exp(x)
        _e = exp(-x)
        return (e -_e)/(e+_e)
    elif(isinstance(x,mat.Matrix)):
        data = []
        for a in x.data:
            e = exp(a)
            _e = exp(-a)
            data.append((e -_e)/(e+_e))
        res = mat.Matrix(x.rows,x.cols,data)
        return res
    else:
        print("Input should be float|matrix")
        return None
    
def tanh_der(x:float|mat.Matrix)->float|mat.Matrix:
    if(isinstance(x,float)):
        return 1-(tanh(x))**2
    elif(isinstance(x,mat.Matrix)):
        data = []
        for a in x.data:
            z = tanh(a)
            data.append( 1 - (z)**2)
        res = mat.Matrix(x.rows,x.cols,data)
        return res
    else:
        print("Input should be float|matrix")
        return None
    
def ReLu(x:float|mat.Matrix)->float|mat.Matrix:
    if(isinstance(x,float)):
        return max(0,x)
    elif(isinstance(x,mat.Matrix)):
        res = mat.Matrix(x.rows,x.cols,[max(0,a) for a in x.data])
        return res
    else:
        print("Input should be float|matrix")
        return None

def ReLu_der(x:float|mat.Matrix)->float|mat.Matrix:
    if x == 0:
        return 0
    if(isinstance(x,float)):
        return 1 if x>0 else 0
    elif(isinstance(x,mat.Matrix)):
        res = mat.Matrix(x.rows,x.cols,[1 if a>0 else 0 for a in x.data])
        return res
    else:
        print("Input should be float|matrix")
        return None
    

class Activation(object):
    def __init__(self):
        super().__init__()
    

class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.act:mat.Matrix = None
    
    def forward(self,x:mat.Matrix)->mat.Matrix:
        self.act = ReLu(x) 
        return self.act
    
    def derivative(self,x:mat.Matrix)->mat.Matrix:
        return ReLu_der(x)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.act:mat.Matrix = None
    
    def forward(self,x:mat.Matrix)->mat.Matrix:
        self.act = sigmoid(x) 
        return self.act
    
    def derivative(self,x:mat.Matrix)->mat.Matrix:
        return sigmoid_der(x)
    
class TanH(Activation):
    def __init__(self):
        super().__init__()
        self.act:mat.Matrix = None
    
    def forward(self,x:mat.Matrix)->mat.Matrix:
        self.act = tanh(x) 
        return self.act
    
    def derivative(self,x:mat.Matrix)->mat.Matrix:
        return tanh_der(x)

def main():
    x = TanH()
    y = Sigmoid()
    z = ReLU()
    print(isinstance(x,Activation))
    print(isinstance(y,Activation))
    print(isinstance(z,Activation))
    
if __name__ == "__main__":
    main()