import matrix_math as mat
import activations as acts
import cost_funcs as cfncs


#28x28 image
# 1st layer: Kernel size = 3 Pooling Stride = 2 =>  13x13 matrix (5 cneurons)
# 2nd layer: Kernel size = 2 Pooling stride = 3 => 4x4 matrix ( 5 cneurons)
# 3rd layer: Kernel size = 3 Pooling Stride = 2 => 1x1 Matrix ( 4 cneurons)
# MLP ------------------------
# 4rd layer: Input size = 4
# 5th layer: Size = 4
# 6th layer: Size = 9


class ConvNeuron(object):
    def __init__(self,filter_height:int,filter_width:int,rand:bool = True,low:float = None,top:float = None)->None:
        """Now they aren't weights they are filters(kernels)"""
        self.filter = mat.Matrix(filter_height,filter_width,[mat.rand_float(low,top) for _ in range(filter_height*filter_width)] if rand else [0 for _ in range(filter_height*filter_width)])
        self.bias = mat.rand_float(low,top)
        
    def convolve(self,input:mat.Matrix)->mat.Matrix:
        output_height = input.rows  - self.filter.rows + 1
        output_width = input.cols  - self.filter.cols + 1 
        output_data = []
        
        for i in range(output_height):
            for j in range(output_width):
                # print(f"{i,j,i+self.filter.rows-1,j+self.filter.cols-1}")
                result = mat.mat_inner_dot(mat.mat_submatrix(input,i,j,i+self.filter.rows - 1 ,j+self.filter.cols - 1),self.filter)
                output_data.append(result)
                
        output:mat.Matrix = mat.Matrix(output_height,output_width,output_data) # move it to the end after 
        return output
    
    def __repr__(self)->str:
        ret = f"KERNEL          BIAS = {self.bias}\n"
        ret += f"{self.filter}\n"
        return ret 

class ConvLayer(object):
    def __init__(self,neurons:list[ConvNeuron],act_func)->None:
        self.convNeurons:list[ConvNeuron] = neurons
        self.act_func = getattr(acts,act_func)
        self.zs:list[mat.Matrix] = []
        self.acts:list[mat.Matrix] = []
        
    def forward(self,volume:list[mat.Matrix])->list[mat.Matrix]:
        """Accepts a volume and forwards one"""
        for j,neuron in enumerate(self.convNeurons):
            for i,channel in enumerate(volume):
                if i == 0:
                    self.zs.append(neuron.convolve(channel))
                else:
                    self.zs[j] = mat.mat_sum(self.zs[j],neuron.convolve(channel))
            self.zs[j] = mat.mat_sum(self.zs[j],mat.Matrix(self.zs[j].rows,self.zs[j].cols,[neuron.bias for _ in range(self.zs[j].rows*self.zs[j].cols)]))
            self.acts.append(self.act_func(self.zs[j]))
        return self.acts
        
class PoolingLayer(object):
    def __init__(self,stride:int):
        """Make sure stride is divisor of the input volume matrix sizes"""
        self.stride = stride
        
    def forward(self,volume:list[mat.Matrix])->list[mat.Matrix]:
        output_volume:list[mat.Matrix] = list()
        for channel in volume:
            
            output_channel_width = channel.cols // self.stride
            output_channel_height = channel.rows // self.stride
            output_channel_data = []
            
            for i in range(output_channel_height):
                for j in range(output_channel_width):
                    i_ = i*self.stride
                    j_ = j*self.stride
                    # print(f"{i_,j_,i_+self.stride-1,j_+self.stride-1}")
                    output_channel_data.append(max(mat.mat_submatrix(channel,i_,j_,i_+self.stride - 1,j_+self.stride - 1).data))
            output_volume.append(mat.Matrix(output_channel_height,output_channel_width,output_channel_data))
        
        return output_volume

class CNN(object):
    def __init__(self,layers):
        self.layers:list = layers
        self.cost_func = None
        
    def forward(self,volume:list[mat.Matrix])->mat.Matrix:
        current = volume
        for layer in self.layers:
            current = layer.forward(current)
        return current
    
    def cost(self,forwarded:mat.Matrix,expected:mat.Matrix)->float:
        return self.cost_func(forwarded,expected)
    
    def train(self, epochs: int, rate: float, train_in: list[list[mat.Matrix]], train_exp: mat.Matrix,plt_values:list,cost_func='sqr_err',act_func='sigmoid') -> None:
        self.cost_func = getattr(cfncs,cost_func)
        cost_func_der = cost_func + '_der'
        self.cost_func_der = getattr(cfncs,cost_func_der) 
        
        self.act_func = getattr(acts,act_func)
        act_func_der = act_func + '_der'
        self.act_func_der = getattr(acts,act_func_der)
        
        for epoch in range(epochs):
            total_cost = 0
            for i in range(len(train_in)):
                # Forward pass
                input_sample = train_in[i]
                output = self.forward(input_sample)

                # Compute cost
                expected_output = mat.Matrix(1, train_exp.cols, train_exp.data[i*train_exp.cols:(i+1)*train_exp.cols])
                cost = self.cost(output, expected_output)
                total_cost += sum(cost.data)

                # Backward pass
                self.backpropagate(input_sample,expected_output, rate)
    
    def backpropagate(self,input_training:mat.Matrix, expected_output: mat.Matrix, learning_rate: float) -> None:
        # Compute the error in the output layer
        output_layer = self.layers[-1]
        delta = mat.mat_hadamard(
            self.cost_func_der(output_layer.acts, expected_output),
            self.act_func_der(output_layer.zs)
        )
        # print(f"COST\n {self.cost_func_der(output_layer.acts, expected_output)}")
        # print(f"ACT \n{self.act_func_der(output_layer.zs)}")
        # print(f"DELTA OUT\n{delta}")
        
        # Backpropagate the error
        for l in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[l]

            # Compute gradients
            if l > 0:
                prev_layer_acts = self.layers[l-1].acts
            else:
                prev_layer_acts = input_training

            for n, neuron in enumerate(layer.neurons):
                # Update weights
                for w in range(neuron.ws.rows):
                    neuron.ws.data[w] -= learning_rate * delta.data[n] * prev_layer_acts.data[w]

                # Update bias
                neuron.bias -= learning_rate * delta.data[n]

            # Compute delta for the previous layer
            if l > 0:
                weights = mat.Matrix(len(layer.neurons), layer.neurons[0].ws.rows, [])
                for n, neuron in enumerate(layer.neurons):
                    weights.data[n*weights.cols:(n+1)*weights.cols] = neuron.ws.data

                weights_T = weights.T()
                delta_T = delta.T()

                # print(f"DELTA\n{delta}")
                # print(f"W\n{weights_T}")
                # print(f"ZS\n{self.layers[l-1].zs}")
                # print(f"DOT\n{mat.mat_dot(weights_T,delta_T)}")
                # Compute delta for the previous layer
                
                delta = mat.mat_hadamard(
                    mat.mat_dot(weights_T,delta_T).T(),
                    self.act_func_der(self.layers[l-1].zs)
                )

def main():
    #Create an artificial volume
    volume = [ mat.Matrix(8,8,[mat.rand_float() for _ in range(64)]) for _ in range(5)]
    print("VOLUME")
    for hola in volume:
        print(hola)
        print("------------------")
    
    #create an artificial Convoloution layer and forward it
    list_neurons = [ConvNeuron(3,3) for _ in range(4)]
    print("NEURONS")
    for neuron in list_neurons:
        print(neuron)
    layer = ConvLayer(list_neurons,"ReLu")
    forwarded_volume = layer.forward(volume)
    
    print("FORWARDED")
    for patch in forwarded_volume:
        print(patch)
        print("----------------")
        
    #Pool the output volume from the input forward
    
    print("POOLED")
    pool_layer = PoolingLayer(3)
    pooled_volume = pool_layer.pool(forwarded_volume)
    for pool in pooled_volume:
        print(pool)
        print("---------------------------")
    
    
        
        
        
if __name__ == "__main__":
    main()