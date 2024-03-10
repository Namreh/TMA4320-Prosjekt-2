import numpy as np
from utils import onehot

class Layer:
    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        #counter for hvilken iterasjon vi er på
        self.j = 0
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError

    def step_Adam(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.j += 1 #oppdaterer hvilken iterasjon vi er på
        #For å unngå to løkker, og samtidig oppdatere j for hver gang
        for param in self.params:
            G = self.params[param]['d']
            M = self.params[param]['m']
            V = self.params[param]['v']

            self.params[param]['m'] = beta1 * M + (1 - beta1) * G
            self.params[param]['v'] = beta2 * V + (1 - beta2) * (np.square(G)) #Kan ta vel ta G**2?

            M_hat = M / (1 - beta1**self.j)
            V_hat = V / (1 - beta2**self.j)

            self.params[param]['w'] -= alpha * (np.divide(M_hat, (np.sqrt(V_hat) + epsilon)))

        return self.params[param]['w']
    
    def step_gd(self,alpha):
    
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']
    '''
    def train_with_Adam(self, L, alpha, beta1, beta2, epsilon, D, niter):
        for j in range(niter):
            for x, y in D:
                output, params = self.forward(x)
                loss = L(output, y)
                grad = self.backward(loss, params)

                # Gjør Adam-oppdateringer med grad og lagets parametre
                self.step_Adam(grad, alpha, beta1, beta2, epsilon, j)
                   '''


class Attention(Layer):

    def __init__(self, d, k):
        #definerer parametermatrisene Wq Wk som tilfeldige variabler
        self.Wq = np.random.randn(k,d)*0.1
        self.Wk = np.random.randn(k,d)*0.1

        #definerer nødvendige variabler
        self.d = d
        self.k = k

        #definerer nødvendige lag innad i attention laget
        self.softmax = Softmax()
        self.Wo = LinearLayer(k, d)
        self.Wv = LinearLayer(d, k)

        self.params = {
            'wq':{
                'w' : self.Wq,
                'd' : np.zeros_like(self.Wq),
                'm' : np.zeros_like(self.Wq),
                'v' : np.zeros_like(self.Wq)
            },
            'wk':{
                'w' : self.Wk,
                'd' : np.zeros_like(self.Wk),
                'm' : np.zeros_like(self.Wk),
                'v' : np.zeros_like(self.Wk)
            },
        }

        super().__init__() #bruker dette for å forsikre oss at alle layers kjører init til Layer
        return


    def forward(self,z):
        self.z = z
        
        #Utfører operasjonen z^T @ W_Q^T @ W_K @ z
        B = np.einsum('bij,ki,kl,blm->bjm', self.z, self.Wq, self.Wk, self.z, optimize=True)

        #setter nedre triangularen til B til -inf
        i1, i2 = np.tril_indices(B.shape[1],-1)
        B[:,i1,i2] -= np.inf
        
        #Utfører softmax
        self.A = self.softmax.forward(B)

        #Ligning 20
        return self.z  + self.Wo.forward(self.Wv.forward(np.einsum('bij,bjk->bik',self.z,self.A, optimize=True)))


    def backward(self,grad):
        self.grad = grad
        b = grad.shape[0]

        self.Wk = self.params['wk']['w']
        self.Wq = self.params['wq']['w']

        g_ov = self.Wv.backward(self.Wo.backward(self.grad))
        g_s = self.softmax.backward(np.einsum('bij,bik->bjk', self.z, g_ov, optimize=True))

        #oppdaterer gradient for parameterene ifølge ligninger 22-25 
        #tar snittet av de ulike batchene
        self.Wo.params['w']['d'] = (np.einsum('ij,bjk,bkl,bml->im', self.Wv.params['w']['w'], self.z, self.A, self.grad, optimize=True)/b).T
        self.Wv.params['w']['d'] = np.einsum('ij,bjk,blk,bml->im', self.Wo.params['w']['w'].T, self.grad, self.A, self.z, optimize=True)/b
        self.params['wk']['d'] = np.einsum('ij,bjk,bkl,bml->im', self.Wq,self.z,g_s,self.z, optimize=True)/b
        self.params['wq']['d'] = np.einsum('ij,bjk,blk,bml->im', self.Wk,self.z,g_s,self.z, optimize=True)/b


        return self.grad + np.einsum('bij,bkj->bik', g_ov, self.A, optimize=True) + np.einsum('ij,ik,bkl,blm->bjm', self.Wk, self.Wq, self.z, g_s, optimize=True) + np.einsum('ij,ik,bkl,bml->bjm', self.Wq, self.Wk, self.z, g_s, optimize=True)
    
    #definerer egen step_gd funksjon
    def step_gd(self, alpha):
        self.Wo.step_gd(alpha)
        self.Wv.step_gd(alpha)

        #kjører originale step_gd funksjonen fra layers
        super().step_gd(alpha)

    def step_Adam(self, alpha, beta1, beta2, epsilon):
        #kaller på step adam for linear layers i laget
        self.Wo.step_Adam(alpha, beta1, beta2, epsilon)
        self.Wv.step_Adam(alpha, beta1, beta2, epsilon)

        #kjører originale step_adam funksjonen fra layers
        super().step_Adam(alpha, beta1,beta2,epsilon)



class Softmax(Layer):

    def __init__(self):
        super().__init__()
        return

    
    def forward(self,z):

        self.P = np.exp(z-z.max(axis = 1, keepdims = True))
        self.Q = np.sum(self.P, axis = 1, keepdims = True)
        
        self.z_l = np.divide(self.P, (self.Q + 10e-8))
        self.z = z
        return self.z_l


    def backward(self,g_l):
        
        S = np.divide(self.P,(np.multiply(self.Q,self.Q)+10e-8))
        delLdelZ = np.multiply(g_l,self.z_l) - np.multiply(np.sum((np.multiply(g_l,S)), axis = 1, keepdims = True),self.P)
        self.delLdelZ = delLdelZ
        return delLdelZ



class CrossEntropy(Layer):

    def __init__(self):

        super().__init__()
        return

        

    def forward(self,Z,y):

        self.y = y
        self.Z = Z


        #Definerer størrelser på dimensjoner
        self.b = Z.shape[0]
        self.m = Z.shape[1]
        self.n = y.shape[1]

        #fjerner de unødvendige dataene
        self.Y_hat = np.copy(self.Z[:,:,-self.n:])

        #Definerer ones = (b,m) andre= (b,m,n)
        self.p = np.einsum('bm,bmn->bn', np.ones((self.b,self.m)), np.multiply(self.Y_hat,onehot(y,self.m)), optimize=True)
        self.q = -np.log(self.p)

        self.L = (1/(self.b*self.n))*np.sum(self.q)
        
        return self.L


    def backward(self):

        self.n = self.Z.shape[-1]
        self.new_Y = np.zeros_like(self.Z)
        
        self.new_Y[:,:,-self.n:] = onehot(self.y, self.m)

        self.grad_Z = -(1/(self.n*self.b))*(np.divide(self.new_Y,(self.Z + 10e-8)))

        return self.grad_Z
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w),
                            'm' : np.zeros_like(self.w),
                            'v' : np.zeros_like(self.w)
                            }
                        }
        
        super().__init__()


    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('ij,bjk->bik',self.params['w']['w'],x, optimize=True)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x, optimize=True)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad, optimize=True)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        super().__init__()
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{
            'w':self.w,
            'd':None,
            'm' : np.zeros_like(self.w),
            'v' : np.zeros_like(self.w)}}
        super().__init__()

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'][:,:grad.shape[-1]] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)
    
    def step_Adam(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        self.embed.step_Adam(alpha,beta1,beta2,epsilon)
        
        super().step_Adam(alpha, beta1, beta2, epsilon)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)
        super().__init__()


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)
    def step_Adam(self, alpha, beta1, beta2, epsilon):
        #kaller på step adam for linear layers i laget
        self.l1.step_Adam(alpha, beta1, beta2, epsilon)
        self.l2.step_Adam(alpha, beta1, beta2, epsilon)