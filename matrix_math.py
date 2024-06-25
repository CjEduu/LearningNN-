"""Defining Matrix class and all the needed operation between matrices and scalars"""

class Matrix(object):
    def __init__(self,rows,cols,data:list[float]):
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


def mat_sum(a:Matrix,b:Matrix)->Matrix:
    assert((a.cols == b.cols) and (a.rows == b.rows))
    return Matrix(a.rows,b.cols,[x+y for (x,y) in zip(a.data,b.data)])

def mat_dot(a:Matrix,b:Matrix)->Matrix:
    assert(a.cols == b.rows)
    data:list[float] = list()
    for i in range(a.rows):
        for j in range(b.cols):
            c = 0
            for k in range(a.cols):
                c += a.mat_in(i,k)*b.mat_in(k,j)
            data.append(c)
    return Matrix(a.rows,b.cols,data)

def mat_scalar(a:Matrix,b:float)->Matrix:
    data = [num*b for num in a.data]
    return Matrix(a.rows,a.cols,data)

def mat_row(mat:Matrix,j:int)->Matrix:
    return Matrix(1,mat.cols,mat.data[mat.cols*j:mat.cols*j + mat.cols])
    
def main():
    mat = Matrix(2,2,[2,3,4,5])
    mat2 = Matrix(2,1,[1 for _ in range(2)])
    mat3 = mat_dot(mat,mat2)
    print(mat)
    mat4 = mat_row(mat,0)
    print(mat4)

if __name__ == "__main__":
    main()
                