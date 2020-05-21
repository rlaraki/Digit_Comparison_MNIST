import torch.nn as nn
from torch.optim import Adam

# Import models
from models.Siamese import *
from models.Auxiliary import *
from models.LeNet import *
from models.LinearNet import *
from models.ResNet import *
from .data_utils import *
from .metrics import *

from .dlc_practical_prologue import generate_pair_sets

import time

SIMPLE_MODELS = {
    "LeNet": le_net,
    "LinearNet": linear_net,
    "ResNet": res_net
}

def select_model(model_name, auxiliary_loss=False, weight_sharing=False, is_siamese=False, dropout=0):
    
    if weight_sharing and not is_siamese:
        raise Exception("You can't build weight sharing without siamese network")
    
    flatten = True if model_name == "LinearNet" else False
    model = SIMPLE_MODELS[model_name](is_siamese) if flatten else SIMPLE_MODELS[model_name](dropout, is_siamese) 
    
    if is_siamese:
        return siamese(model, flatten, weight_sharing), flatten
    
    if auxiliary_loss:
        return auxiliary_net(model, flatten), flatten
    
    return model, flatten   

def full_train_test(model_name, auxiliary_loss=False, is_siamese=False, weight_sharing=False, num_iter=10, num_epochs=25, data_size=1000, batch_size=10, aux_rate = 0.5, verbose=False):
    
    train_loss_matrix = [] 
    validation_loss_matrix = []
    
    accuracy_array = []
    time_array = []
    
    for it in range(1, num_iter + 1):
        if verbose:
            print("Iteration %d" % it)
        
        start_time = time.time()
        
        # Create model instance
        model, flatten = select_model(model_name, auxiliary_loss, weight_sharing, is_siamese)
        
        # Generate a train and validation set
        train_input, train_target, train_classes, val_input, val_target, val_classes = generate_pair_sets(data_size)
        
        
        tr_loader, val_loader = build_loader(train_input,train_classes, train_target, val_input, val_classes,val_target,  batch_size)
        tr_losses, val_losses = train_model(model, num_epochs, tr_loader, val_loader, is_siamese , weight_sharing , auxiliary_loss, flatten, verbose, aux_rate)
        
        #
        train_loss_matrix.append(tr_losses)
        validation_loss_matrix.append(val_losses)
        
        # Generate test set
        _,_,_, acc_input, acc_target, _ = generate_pair_sets(data_size)
        acc_loader = build_test_loader(acc_input, acc_target, batch_size)
        
        # Compute test accuracy
        acc = accuracy(model, acc_loader,is_siamese,auxiliary_loss, flatten )

        end_time = time.time()   
        
        accuracy_array.append(acc)
        time_array.append(end_time - start_time)
    
    # Plot performances metrics
    if verbose:
        plot_losses(train_loss_matrix, validation_loss_matrix, model_name)    
        plot_accuracy(accuracy_array, model_name)
          
    
    acc_mean, acc_std = extract_mean_std(accuracy_array)
    time_mean, time_std = extract_mean_std(time_array)
    
    print("Accuracy: %.3f +/- %.3f" %(acc_mean, acc_std))
    print("Iteration time:  %.3f +/- %.3f seconds" % (time_mean, time_std))
    


def train_model(model, num_epochs, train_loader, test_loader, split=True, weight_sharing=True,
                       auxiliary=False, flatten=True, verbose=False, auxiliary_f=1):
    # Create model parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Init the losses
    tr_losses = []
    te_losses = []

    for epoch in range(1, num_epochs + 1):
        epoch_tr_loss = 0.0
        epoch_te_loss = 0.0

        # Iterate over the train/ batches 
        for (tr_inputs, tr_classes, tr_labels), (te_inputs, te_classes, te_labels) in zip(train_loader, test_loader):
            
            if flatten and (not split):
                tr_inputs = tr_inputs.view(-1, tr_inputs.shape[1]*tr_inputs.shape[2] * tr_inputs.shape[3])
                te_inputs = te_inputs.view(-1, te_inputs.shape[1]*te_inputs.shape[2] * te_inputs.shape[3])
                

            optimizer.zero_grad()

            # Compute prediction
            if split or auxiliary:
                tr_output, out1, out2 = model(tr_inputs)
                te_output, out1, out2 = model(te_inputs)

            else:
                tr_output = model(tr_inputs)
                te_output = model(te_inputs)

            # Compute loss
            tr_loss = criterion(tr_output, tr_labels.long())
            te_loss = criterion(te_output, te_labels.long())

            # Compute auxiliary loss if needed
            if auxiliary:
                cl_loss_tr1 = criterion(out1, tr_classes.long()[:, 0])
                cl_loss_tr2 = criterion(out2, tr_classes.long()[:, 1])
                total_loss = tr_loss + auxiliary_f * cl_loss_tr1 + auxiliary_f * cl_loss_tr2
            else:
                total_loss = tr_loss

            # Perform optimization step
            total_loss.backward()
            optimizer.step()

            # Add loss to epoch loss
            epoch_tr_loss += tr_loss.item()
            epoch_te_loss += te_loss.item()

        # Compute average epoch loss and add it to losses array
        epoch_tr_loss = epoch_tr_loss / len(train_loader)
        epoch_te_loss = epoch_te_loss / len(test_loader)
        tr_losses.append(epoch_tr_loss)
        te_losses.append(epoch_te_loss)
    return tr_losses, te_losses
