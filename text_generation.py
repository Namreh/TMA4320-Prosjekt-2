from utils import onehot
from data_generators import text_to_training_data
import numpy as np


def generate(net,start_idx,m,n_max,n_gen):
    
    #We will concatenate all generated integers (idx) in total_seq_idx
    total_seq_idx = start_idx

    n_total = total_seq_idx.shape[-1]
    slice = 0

    x_idx = start_idx

    while n_total < n_gen:
        n_idx = x_idx.shape[-1]
        X = onehot(x_idx,m)

        #probability distribution over m characters
        Z = net.forward(X)

        #selecting the last column of Z (distribution over final character)
        hat_Y = Z[0,:,-1]

        #sampling from the multinomial distribution
        #we do this instead of argmax to introduce some randomness
        #avoiding getting stuck in a loop
        y_idx = np.argwhere(np.random.multinomial(1, hat_Y.T)==1)

        if n_idx+1 > n_max:
            slice = 1

        #we add the new hat_y to the existing sequence
        #but we make sure that we only keep the last n_max elements
        x_idx = np.concatenate([x_idx[:,slice:],y_idx],axis=1)

        #we concatenate the new sequence to the total sequence
        total_seq_idx = np.concatenate([total_seq_idx,y_idx],axis=1)

        n_total = total_seq_idx.shape[-1]

    return total_seq_idx


def convertToID(text, dict):
    output = []
    for c in text:
        output.append(dict[c])
    return np.array(output)

def convertToTxt(ids, dict):
    output = ""
    for i in ids:
        output += (dict[i])
    return output