import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_structs import MolData, Vocabulary
from utils import Variable, decrease_learning_rate
from tqdm import tqdm
from rdkit import Chem


class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))

class MultiLSTM(nn.Module):
    """ Implements a three layer LSTM cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiLSTM, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.lstm_1 = nn.LSTMCell(128, 512)
        self.lstm_2 = nn.LSTMCell(512, 512)
        self.lstm_3 = nn.LSTMCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.lstm_1(x, h[0])
        x = h_out[1] = self.lstm_2(x, h[1])
        x = h_out[2] = self.lstm_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))

class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

    def likelihood(self, target):
        """
            Retrieves the likelihood of a given sequence
            Args:
                target: (batch_size * sequence_lenght) A batch of sequences
            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        final_col = target[:,seq_length-1]
        final_col = torch.reshape(final_col,(batch_size,1))
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :]), 1)
        target = torch.cat((target, final_col),1)
        h = self.rnn.init_h(batch_size)  #[3,128,512]

        log_probs = Variable(torch.zeros(batch_size))  # 128
        entropy = Variable(torch.zeros(batch_size))    
        for step in range(seq_length+1):
            logits, h = self.rnn(x[:, step], h)    #logits:[128,50]
            log_prob = F.log_softmax(logits, dim=1)    #[128,50]
            prob = F.softmax(logits, dim=1)       #[128,50]
            log_probs += NLLLoss(log_prob, target[:, step])   #[128]
            entropy += -torch.sum((log_prob * prob), 1)    #[128]
#        log_probs = log_probs/(seq_length+1)
        return log_probs, entropy

    def sample(self, batch_size, max_length=140):
        """
            Sample a batch of sequences
            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences
            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        for step in range(max_length):
            logits, h = self.rnn(x, h)
            prob = F.softmax(logits,dim=1)
            log_prob = F.log_softmax(logits,dim=1)
            x = torch.multinomial(prob,num_samples=1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

    def fit(voc_path, mol_path, restore_path, max_save_path, last_save_path,
	    epoch_num, step_num, decay_step_num, smile_num, lr, weigth_decay):
  
        restore_from = restore_path # if not restore model print None
        # Read vocabulary from a file
        voc = Vocabulary(init_from_file = voc_path)

        # Create a Dataset from a SMILES file
        moldata = MolData(mol_path, voc)
        data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                              collate_fn=MolData.collate_fn)

        Prior = RNN(voc)

        # Can restore from a saved RNN
        if restore_from:
           Prior.rnn.load_state_dict(torch.load(restore_from))
 
        total_loss=[]
        total_valid = []
        max_valid_pro=0
        
        optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = lr)

        for epoch in range(1, epoch_num):
            for step, batch in tqdm(enumerate(data), total=len(data)):

                # Sample from DataLoader
                seqs = batch.long()

                # Calculate loss
                log_p, _ = Prior.likelihood(seqs)
                loss = - log_p.mean()

                # Calculate gradients and take a step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Every 300 steps we decrease learning rate and print some information
                if step!=0 and step % decay_step_num == 0:
                    decrease_learning_rate(optimizer, decrease_by = weigth_decay)
                if  step % step_num == 0:
                    tqdm.write("*" * 50)
                    tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss))
#                    print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss))
                    total_loss.append(float(loss))
                    seqs, likelihood, _ = Prior.sample(128)
                    valid = 0
#                    smiles=[]
#                    vali_smi=[]
                    for i, seq in enumerate(seqs.cpu().numpy()):
                        smile = voc.decode(seq)
#                        smiles.append(smile)
                        if Chem.MolFromSmiles(smile):
                            valid += 1
#                            vali_smi.append(smile)
                        if i < smile_num:
                            print(smile)
                    vali_pro=valid/len(seqs)
                    total_valid.append(float(vali_pro))
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("*" * 50 + "\n")

                    if vali_pro > max_valid_pro:
                        max_valid_pro = vali_pro
                        torch.save(Prior.rnn.state_dict(), max_save_path)

            # Save the Prior
            torch.save(Prior.rnn.state_dict(), last_save_path)

        print("total loss:", total_loss)
        print("total valid:", total_valid)
        return total_loss, total_valid


def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.
        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*
        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
