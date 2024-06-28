"""Defining Matrix class and all the needed operation between matrices and scalars"""
from random import random

def rand_float(low=None,top=None)->float:
    """If no args passed, generates between 0 and 1"""
    if low == None or top == None:
        rand = random()    
    else:
        rand = low + random() * (top - low)
    return rand


class Matrix(object):
    def __init__(self,rows,cols,data:list[float]):
        if len(data) == 0:
            data = [0 for _ in range(rows*cols)]
        assert(rows*cols == len(data))
        data = [float(x) for x in data]
        self.rows = rows
        self.cols = cols
        self.data:list[float] = data
    
    def mat_in(self,i:int,j:int)->float:
        return self.data[i*self.cols + j]
    
    def __repr__(self)->str:
        ret = ""
        for i in range(self.rows):
            ret += '['
            for j in range(self.cols):
                ret += f" {self.mat_in(i,j):.2f} "
            ret += ']\n'
        ret = ret[:-1];
        return ret

    def T(self)->"Matrix":
        transposed_data = []
        for j in range(self.cols):
            for i in range(self.rows):
                transposed_data.append(self.mat_in(i, j))
        return Matrix(self.cols, self.rows, transposed_data)
    
    
def mat_sum(a:Matrix,b:Matrix)->Matrix:
    assert((a.cols == b.cols) and (a.rows == b.rows))
    return Matrix(a.rows,b.cols,[x+y for (x,y) in zip(a.data,b.data)])

def mat_dot(a:Matrix,b:Matrix)->Matrix:
    assert(a.cols == b.rows)
    x = a.rows
    y = b.cols
    data:list[float] = list()
    for i in range(a.rows):
        for j in range(b.cols):
            c = 0
            for k in range(a.cols):
                c += a.mat_in(i,k)*b.mat_in(k,j)
            data.append(c)
    return Matrix(x,y,data)


def mat_hadamard(a:Matrix,b:Matrix)->Matrix:
    assert(a.rows == b.rows and a.cols == b.cols)
    x = a.rows
    y = a.cols
    return Matrix(x,y,[x*y for (x,y) in zip(a.data,b.data)])

def mat_scalar(a:Matrix,b:float)->Matrix:
    data = [num*b for num in a.data]
    return Matrix(a.rows,a.cols,data)

def mat_row(mat:Matrix,j:int)->Matrix:
    return Matrix(1,mat.cols,mat.data[mat.cols*j:mat.cols*j + mat.cols])
    
def main():
    mat = Matrix(2,2,[1,0,1,0])
    mat2 = Matrix(2,2,[-1,2,4,-2])

    print("----------------------")
    print(mat)
    print("----------------------")
    print(mat2)
    print("----------------------")
    print(mat_dot(mat,mat2))
if __name__ == "__main__":
    main()
                