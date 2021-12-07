
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
rng = np.random.default_rng()


def k_cap(input, cap_size):
        """

        Given a vector input it returns the highest cap_size 
        entries from cap_zie
        """
        output = np.zeros_like(input)
        if len(input.shape) == 1:
            idx = np.argsort(input)[-cap_size:]
            output[idx] = 1
        else:
            idx = np.argsort(input, axis=-1)[:, -cap_size:]
            np.put_along_axis(output, idx, 1, axis=-1)
        return output

class brain_region: 
    """
    Creates a brain region from assembly calculus 
    
    """

    def __init__(self, n_neurons , n_in, cap_size, id: int ) -> None:
        """
        Creates a brain region that takes 
        """
        self.id=id
        self.n_neurons=n_neurons
        self._n_in=n_in
        self.cap_size=cap_size
        mask = np.zeros((self.n_neurons, self.n_neurons), dtype=bool) # NXN array of zeros 
        W = np.zeros((self.n_neurons, self.n_neurons)) 
        mask_a = np.zeros((self._n_in, self.n_neurons), dtype=bool) # image to N matrix 
        A = np.zeros((self._n_in, self.n_neurons))
        mask = (rng.random((self.n_neurons, self.n_neurons)) < sparsity) & np.logical_not(np.eye(n_neurons, dtype=bool)) # Creating matrix from N to B  with no entries in the diagonal 
        W = np.ones((self.n_neurons, self.n_neurons)) * mask 
        W /= W.sum(axis=0) # Transition probabiliy matrix 
        mask_a = rng.random((self._n_in, self.n_neurons)) < sparsity 
        A = np.ones((self._n_in, self.n_neurons)) * mask_a
        A /= A.sum(axis=0)
        W = np.ones_like(W) * mask
        A = np.ones_like(A) * mask_a
        W /= W.sum(axis=0, keepdims=True)
        A /= A.sum(axis=0, keepdims=True)
        self._W=W
        self._A=A
        self.mask=mask
        self.mask_a=mask_a
        self.act_h = np.zeros(self.n_neurons)
        self.bias = np.zeros(self.n_neurons)
        self.b = -1
          
        self.classify_act_h=np.zeros(self.n_neurons)


    def next(self, input, initial=False, final=False ):
        """
        It computes the activation output of the input going 
        through this brain region. 
        
        """
        if initial:
            self.act_h = np.zeros(self.n_neurons)        

        act_h_new = k_cap(self.act_h @ self._W + input @ self._A + self.bias, self.cap_size)   # output a NXN array for the neurons that are activated. The first part is from self activiation and second from inoput
        self._A[(input > 0)[:, np.newaxis] & (act_h_new > 0)[np.newaxis, :]] *= 1 + beta
        self._W[(self.act_h > 0)[:, np.newaxis] & (act_h_new > 0)[np.newaxis, :]] *= 1 + beta
        self.act_h = act_h_new
        if final: 
            self.reinforce_bias()
        print("Shape of act_h:"+str(self.act_h.shape))
        return self.act_h.copy()

    def reinforce_bias(self):
        """
        This function is meant to be called at the end of each round to renormalize the transition matrices 
        
        """
        self.bias[self.act_h>0]+=-1 # after all rounds the activated neurons have a smaller bias so they more likely to fire 
        self._A /= self._A.sum(axis=0, keepdims=True)
        self._W /= self._W.sum(axis=0, keepdims=True)



    def classify(self,input,  n_examples, initial=False, ):
        if initial:
            self.classify_act_h=np.zeros((n_examples , self.n_neurons))
        
        self.classify_act_h=k_cap(self.classify_act_h @ self._W + input @ self._A + self.bias, self.cap_size)
        return self.classify_act_h

        


class assembly_network: 
    """
    This class is meant to implement the assembly calculus structure
    This generalizes for multiple inputs and brain regions
    
    """

    def __init__(self, number_of_inputs: int , sparsity:int, layers: list, beta: float) -> None:

        """
        Initializes the structure of the Brain Region. It takes the number of inputs and then a list for layers that should contain tuples of the form (neurons, cap_size).

        
        """
        self.n_in = number_of_inputs  # Vector of 28X28 pixels 
         # List with pairs of tuples (n, c) where n is the number of neurons and c is the size of the cap 
        self.create_layers(layers) # Creates all the structure for the brain regions 
        self.sparsity = sparsity
        self.beta =beta  
    

    def create_layers(self, layers)-> None: 
        """ 
        Creates brain regions according to the list from layers 

        The layers list should contain tuples of the form (number of neurons, cap size)
        
        """
        
        self.layers=[]
        temp=self.n_in+0
        for k, (neurons, cap_size) in enumerate(layers):
            self.layers.append(brain_region(neurons, temp, cap_size, k))
            temp=neurons+0
    
        
    def next(self, input: np.array, initial=False, final=False ):
        """
        During the training process, it puts the input 
        through the network and it runs it through all the layers 




        """
    
        temp=input
        print(self.layers)
        for k , brain_region_k in enumerate(self.layers): 
            new_temp=brain_region_k.next(temp, initial=initial, final=final)
            temp=new_temp
            
        return temp 

    def classify(self,input, initial=False ):
        temp=input
        
        for brain_region in self.layers: 
            print("temp shape"+str(temp.shape))
            temp=brain_region.classify(temp, input.shape[0], initial)
        return temp 

class classification_mnist:

    def __init__(self, kernels: list ,train_path: str, test_path: str, number_of_inputs: int , sparsity:int, layers: list, beta: float  ) :
        """

        Creates a MNIST recognition architecture based on assembly calculus
        """
        
        self.cap_size=layers[-1][1]
        self.n_neurons= layers[-1][0]   
        self.n_in=number_of_inputs
        self.assembly_network=assembly_network(number_of_inputs, sparsity, layers, beta , )
        self.get_files( train_path, test_path)
        self.create_training_data(kernels)
        self.create_testing_data(kernels)
        

    def create_training_data(self ,kernels= [np.ones((1, 3, 3))] ):
        """

        Creates a data set with n_examples from the files obtained 
        by get_files 

        """
        self.train_examples = [] 
        for kernel in kernels:
            self.train_examples.append(np.zeros((10, self.n_examples, 784)))
            for i in range(10):
            #Does the convulution between a all 1's 3X3 kernel and each of the images 
                self.train_examples[-1][i] = k_cap(convolve(self.train_imgs[self.train_labels == i][:self.n_examples].reshape(-1, 28, 28), kernel, mode='same').reshape(-1, 28 * 28), self.cap_size)
    
    def create_testing_data(self ,kernels= [np.ones((1, 3, 3))] ):
        """

        Creates a data set with n_examples from the files obtained 
        by get_files 

        """
        self.test_examples = [] 
        for kernel in kernels:
            self.test_examples.append(  np.zeros((10, self.n_examples, 784) ))
            for i in range(10):
            #Does the convulution between a all 1's 3X3 kernel and each of the images 
                self.test_examples[-1][i] = k_cap(convolve(self.test_imgs[self.test_labels == i][:self.n_examples].reshape(-1, 28, 28), kernel, mode='same').reshape(-1, 28 * 28), self.cap_size)
    
    


    def get_files(self, train_path: str, test_path: str)-> None:
        """
        Given two paths it retrieves the data structure encoded in those paths. traun_path should be the path of the training data 
        and test_path should be the path for test data. 
        Assumes a csv format on nthe data on the paths 
        """
        test_data = np.loadtxt(test_path, delimiter=',')

        train_data = np.loadtxt(train_path,  delimiter=',')
        self.train_imgs = train_data[:, 1:]
        self.train_imgs.shape
        self.test_imgs = test_data[:, 1:]
        self.train_labels = train_data[:, 0]
        self.test_labels = test_data[:, 0]

    def train_model(self, n_rounds)-> np.array:
        """
        Given the number of rounds (images that will be shown to the model)

        The program runs and trains the edge weights for the network.
        """
        self.activations = np.zeros((10, n_rounds, self.n_neurons))
        for i in range(10): # iterations for each of the labels 
            for j in range(n_rounds):  # for each of the rounds 
                input = self.train_examples[0][i, j] # image inputs
                act_h= self.assembly_network.next(input, initial=(j==0), final= (j==n_rounds-1) ) # output a NXN array for the neurons that are activated. The first part is from self activiation and second from inoput
                self.activations[i, j] = act_h
        return self.activations

    def classify(self, n_rounds,  test=True )-> dict:
        """
        When called, this function runs one batch of data through 
        the whole network and then returns a dictionary with succes rates

        """
        if test:
            examples=self.test_examples[0]
            
        else:
            examples=self.train_examples[0]
        self.n_examples=examples.shape[1]
        #### RUNS THROUGH NETWORK    
        outputs = np.zeros((10, n_rounds+1, self.n_examples, self.n_neurons))
        for i in np.arange(10):
            for j in range(n_rounds):
                outputs[i, j+1] = self.assembly_network.classify(examples[i], initial= (j==0)) # run each one network for n_rounds and save the neurons active at each step
        #### STARTS CLASSIFICATION 
        c = np.zeros((10, self.n_neurons))
        for i in range(10):
            c[i, outputs[i, 1].sum(axis=0).argsort()[-self.cap_size:]] = 1

        predictions = (outputs[:, 1] @ c.T).argmax(axis=-1)
        acc = (predictions == np.arange(10)[:, np.newaxis]).sum(axis=-1) / self.n_examples
        return acc

n_in = 784  # Vector of 28X28 pixels 
cap_size = 200 # Size of the cap 
sparsity = 0.1
n_rounds = 10
n_examples=800
beta = 1e0
train_path="./data/mnist/mnist_train.csv"
layers=[ (2000,200)]# number of neurons in network  with respective cap_size
test_path="./data/mnist/mnist_test.csv"
kernels=[np.ones((1, 3, 3))]
classify_two=classification_mnist(kernels,train_path,test_path, n_in , sparsity, layers, beta)
classify_two.train_model( 5)
print(classify_two.classify( 5, test=False))





    
