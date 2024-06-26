"""ALL cost functions JUST a training input and a training output"""


import matrix_math as mat


def sqr_err(returned:mat.Matrix,to:mat.Matrix)->float:
    n = returned.cols
    assert(n == to.cols)
    cost = 0.0
    for i in range(n):
        cost += (returned.data[i] - to.data[i])**2
    cost = cost/n
    return cost


def sqr_err_der(returned:mat.Matrix,to:mat.Matrix)->float:
    n = returned.cols
    assert(n == to.cols)
    cost = 0.0
    for i in range(n):
        cost += 2*(returned.data[i] - to.data[i])
    cost = cost/n
    return cost
