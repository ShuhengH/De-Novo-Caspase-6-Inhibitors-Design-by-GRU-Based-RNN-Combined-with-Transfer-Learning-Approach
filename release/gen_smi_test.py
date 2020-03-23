from data_structs import Vocabulary
from model_construct_test import RNN
import torch
import pandas as pd
from rdkit import Chem
import gc

def Generator(restore_path, voc_path, csv_num, gen_num, mol_num):

    restore_from = restore_path
    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_path)

    for n in range(0,csv_num):
        Prior = RNN(voc)
    
        if restore_from:
            Prior.rnn.load_state_dict(torch.load(restore_from))

        seqs, likelihood, _ = Prior.sample(gen_num)
    
        valid = 0
        smiles=[]
        val_smi=[]
        for i, seq in enumerate(seqs.cpu().numpy()):
            smile = voc.decode(seq)
            smiles.append(smile)
            if Chem.MolFromSmiles(smile):
                valid += 1
                val_smi.append(smile)
            if i < mol_num:
                print(smile)
    
        Val_s = pd.DataFrame(data=val_smi,columns=['smiles'])
        Val_s.to_csv('./model/data_gen_'+str(n)+'.csv',index=False)
        print(valid)
        gc.collect()  