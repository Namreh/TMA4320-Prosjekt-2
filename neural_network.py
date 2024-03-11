from layers import *

class NeuralNetwork():
    """
    Neural network class that takes a list of layers
    and performs forward and backward pass, as well
    as gradient descent step.
    """

    def __init__(self,layers):
        #layers is a list where each element is of the Layer class
        self.layers = layers
    
    def forward(self,x):
        #Recursively perform forward pass from initial input x
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self,grad):
        """
        Recursively perform backward pass 
        from grad : derivative of the loss wrt 
        the final output from the forward pass.
        """

        #reversed yields the layers in reversed order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def step_gd(self,alpha):
        """
        Perform a gradient descent step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            #Check if layer is of class a class that has parameters
            if isinstance(layer,(LinearLayer,EmbedPosition,FeedForward,Attention)):
                layer.step_gd(alpha)
        return
    
    def step_Adam(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Perform a gradient descent step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            #Check if layer is of class a class that has parameters
            if isinstance(layer,(LinearLayer,EmbedPosition,FeedForward,Attention)):
                layer.step_Adam(alpha, beta1, beta2, epsilon)
        return
    
    def predict(self, x_test, m, output_length):
        predictions = []
        for i in range(x_test.shape[0]):
            x = x_test[i]
            for n in range(output_length):
                X = onehot(x, m)
                Z = self.forward(X)
                z = np.argmax(Z, axis=1)
                x = np.append(x, z[:,-1:], axis=1)
            predictions.append(x[:,-output_length:])
        return np.array(predictions)
    
    def countCorrect_sort(self, y_hat, y):
        batches = y_hat.shape[0]
        samples = y_hat[0].shape[0]

        counter = 0
        total = samples*batches

        for b in range(batches):
            for i in range(samples):
                if np.sum(y_hat[b,i] - y[b,i]) == 0:
                    counter += 1
        print("Antall rette prediksjoner:", counter)
        print("Totalt antall prediksjoner:", total)
        print("Prosentvis riktige predikasjoner:", (counter/total)*100, "%")
        return

    
    def countCorrect_add(self, y_hat, y):
        batches = y_hat.shape[0]
        samples = y_hat[0].shape[0]

        counter = 0
        total = samples*batches
        #Ittererer gjennom prediksjoner og forventede svar og teller antall rette prediksjoner
        for b in range(batches):
            for i in range(samples):
                #Tar ut og sammenlikner koresponderende siffer i prediksjon og treningsdata
                if np.sum(y_hat[b,i] - np.flip(y[b,i])) == 0:
                        counter += 1
        print("Antall rette prediksjoner:", counter)
        print("Totalt antall prediksjoner:", total)
        print("Prosentvis riktige predikasjoner:", round((counter/total)*100, 6), "%")
        return 