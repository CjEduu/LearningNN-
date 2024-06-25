"""ALL cost functions should accept a NN, a training input and a training output"""


import matrix_math as mat


def avg_sqr_err(nn,ti:mat.Matrix,to:mat.Matrix)->float:
    assert(ti.rows == to.rows)
    total = 0.0
    n_ti = ti.rows
    for j in range(ti.rows): # iterate over the train inputs
        input = mat.mat_row(ti,j)
        expected = mat.mat_row(to,j)
        returned = nn.forward(input)
        
        n = returned.cols
        assert(n == expected.cols)
        cost = 0.0
        for i in range(n):
            cost += (expected.data[i] - returned.data[i])**2
        cost = cost/n
        total += cost
    return total/n_ti