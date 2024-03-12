import numpy as np
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt

def createAllSamples():
    #Lager x og y arrays med dimensjon (100, 100, 4) og (100, 100, 3) med alle de mulige tallparene
    x_values = np.zeros((100, 100, 4), dtype=int)
    y_values = np.zeros((100, 100, 3), dtype=int)

    for i in range(100):
        for j in range(100):
            x_values[i, j] = [i // 10, i % 10, j // 10, j % 10]  
            sum_tall = i + j
            if sum_tall < 100: #legger til 0 før summen for å sikre en lengde på 3
                y_values[i, j] = [0, sum_tall // 10, sum_tall % 10]  
            else:
                y_values[i, j] = [sum_tall // 100, sum_tall // 10 % 10, sum_tall % 10] 
                
    return x_values, y_values

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

def test_Adam(nn: NeuralNetwork, x_data, y_data, n_iters, step_size, m, start, stop):
    n_batches = x_data.shape[0]
    mean_losses = np.zeros(n_iters)
    for j in range(n_iters):
        losses = []
        for i in range(n_batches):
            x = x_data[i]
            y = y_data[i][:,start:stop]
            #Forward pass
            X = onehot(x,m)
            Z = nn.forward(X)
            #Backward pass
            losses.append(loss.forward(Z,y))
            dLdZ = loss.backward()
            nn.backward(dLdZ)
            nn.step_Adam(step_size)
        mean_loss = np.mean(losses)
        print("Iterasjon ", str(j+1), " L = ",mean_loss, "")
        mean_losses[j] = mean_loss
    #Plotter loss-funksjonen per itterasjon i en logaritmisk skala
    plt.plot(np.arange(0,n_iters), np.log(mean_losses))
    plt.xlabel("Iterasjonnummer")
    plt.ylabel("Logaritmen av loss-funksjon")
    plt.title("Minimering ved antall iterasjoner")
    plt.show()
    #Returnerer gjennomsnittet over objektfunksjonen over batchene
    return mean_losses


def onehot(x,m):
    """
    Input:
    - x : np.array of integers with shape (b,n)
             b is the batch size and 
             n is the number of elements in the sequence
    - m : integer, number of elements in the vocabulary 
                such that x[i,j] <= m-1 for all i,j

    Output:     
    - x_one_hot : np.array of one-hot encoded integers with shape (b,m,n)

                    x[i,j,k] = 1 if x[i,j] = k, else 0 
                    for all i,j
    """

    b,n = x.shape

    #Making sure that x is an array of integers
    x = x.astype(int)
    x_one_hot = np.zeros((b,m,n))
    x_one_hot[np.arange(b)[:,None],x,np.arange(n)[None,:]] = 1
    return x_one_hot

