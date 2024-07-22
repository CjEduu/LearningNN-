import matrix_math as mat
import activations as acts


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
            self.acts.append(self.act_func(self.zs[j]))
        return self.acts
        
class PoolingLayer(object):
    def __init__(self,stride:int):
        """Make sure stride is divisor of the input volume matrix sizes"""
        self.stride = stride
        
    def pool(self,volume:list[mat.Matrix])->list[mat.Matrix]:
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