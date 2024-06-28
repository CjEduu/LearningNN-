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



def main():
    x = -1.0
    z = 1.5
    t = 1.9
    y = mat.Matrix(1,3,[-1.0,1.5,1.9])
    
    print(ReLu_der(x))
    print(ReLu_der(z))
    print(ReLu_der(t))
    print(ReLu_der(y))
    
    
if __name__ == "__main__":
    main()