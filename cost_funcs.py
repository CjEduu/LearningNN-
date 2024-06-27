"""ALL cost functions should accept JUST a training input and a training output"""


import matrix_math as mat


def sqr_err(returned:mat.Matrix,to:mat.Matrix)->mat.Matrix:
    """Returns the error between the 2 vectors"""
    m = returned.rows
    n = returned.cols
    assert(n == to.cols and m == to.rows)
    ret = mat.Matrix(m,n,[])
    data = list()
    for i in range(m):
        for j in range(n):
            data.append((returned.mat_in(i,j)-to.mat_in(i,j))**2)
    ret.data = data
    return ret


def sqr_err_der(returned:mat.Matrix,to:mat.Matrix)->float:
    m = returned.rows
    n = returned.cols
    assert(n == to.cols and m == to.rows)
    ret = mat.Matrix(m,n,[])
    data = list()
    for i in range(m):
        for j in range(n):
            data.append(2*(returned.mat_in(i,j)- to.mat_in(i,j)))
    ret.data = data
    return ret

