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



def main():
    x = 5.0
    z = 4.0
    t = 3.0
    y = mat.Matrix(1,3,[5.0,4.0,3.0])
    
    print(sigmoid_der(x))
    print(sigmoid_der(z))
    print(sigmoid_der(t))
    print(sigmoid_der(y))
    
    
if __name__ == "__main__":
    main()