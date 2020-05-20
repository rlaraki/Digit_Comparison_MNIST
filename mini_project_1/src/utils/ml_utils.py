
import torch.nn as nn
from torch.optim import Adam     
from models.Aux import aux
from models.Siamese import siamese
            
def get_model(model_info, nb_channels, split, weight_sharing, auxiliary):


    if (split):
        args = model_info.get("args")
        if args is None:
            model = model_info.get('f1')()
        else:
            model = model_info.get('f1')(**args)
            model_1 = model
        args = {"m1": model_1, "ws": weight_sharing, "flatten":model_info.get('flatten')}
        model = siamese(**args)
    else:
        if (model_info.get("name") == "Residual Net"):
            args = model_info.get('args'+str(nb_channels))
            model = model_info.get('f1')(**args)
        else:
            args = model_info.get("args")    
            model =  model_info.get('f'+ str(nb_channels))(**args)
        if (auxiliary) & (nb_channels == 2):
            model_1 = model
            args = {"m1": model_1, "ws": weight_sharing, "flatten":model_info.get('flatten')}
            model = aux(**args)
            
            
    return model

# Train a model on given data
def train_model(model, device, num_epochs, train_loader, test_loader, flatten=False, verbose=False):
     # Create model parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
        
    # Init the losses
    tr_losses = []
    te_losses = []
        
    for epoch in range(1, num_epochs+1):
        epoch_tr_loss = 0.0
        epoch_te_loss = 0.0
            
        # Iterate over the train/ batches 
        for (tr_inputs, tr_labels), (te_inputs, te_labels) in zip(train_loader, test_loader):
                
            # Iterate over the two digits of the pair
            for digit in range(2):
                    
                # Load train example 
                tr_input = tr_inputs[:, digit, :, :].to(device)
                tr_label = tr_labels.t()[digit].to(device)
                    
                # Load test example
                te_input = te_inputs[:, digit, :, :].to(device)
                te_label = te_labels.t()[digit].to(device)
                    
                if flatten:
                    tr_input = tr_input.view( -1, tr_input.shape[1] * tr_input.shape[2])
                    te_input = te_input.view( -1, te_input.shape[1] * te_input.shape[2])
                else:
                    tr_input = tr_input.unsqueeze(1) # Add channel for convolutional model
                    te_input = te_input.unsqueeze(1)
                        
                optimizer.zero_grad()
                
                # Compute loss
                tr_output = model(tr_input)
                te_output = model(te_input)
            
                
                tr_loss = criterion(tr_output, tr_label.long())
                te_loss = criterion(te_output, te_label.long())
                    
                tr_loss.backward()
                optimizer.step()
                    
                # Add loss to epoch loss
                epoch_tr_loss += tr_loss.item()
                epoch_te_loss += te_loss.item()
                    
        # Compute average epoch loss and add it to losses array
        epoch_tr_loss = epoch_tr_loss/len(train_loader)
        epoch_te_loss = epoch_te_loss/len(test_loader)
        tr_losses.append(epoch_tr_loss)
        te_losses.append(epoch_te_loss)
    return tr_losses, te_losses


def train_model_two_ch(model, device, num_epochs, train_loader, test_loader,split = True, weight_sharing = True, auxiliary = False, flatten=True, verbose=False, auxiliary_f= 1):
     # Create model parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
        
    # Init the losses
    tr_losses = []
    te_losses = []
        
    for epoch in range(1, num_epochs+1):
        epoch_tr_loss = 0.0
        epoch_te_loss = 0.0
        
       
            
        # Iterate over the train/ batches 
        for(tr_inputs, tr_classes, tr_labels), (te_inputs, te_classes, te_labels) in zip(train_loader, test_loader):
                
           # Load train example 
            tr_input = tr_inputs.to(device)
            tr_label = tr_labels.to(device)
                
                # Load test example
            te_input = te_inputs.to(device)
            te_label = te_labels.to(device)
            
            if (auxiliary == True):    
                tr_classes = tr_classes.to(device)
                te_classes = te_classes.to(device)
                    
                    
            optimizer.zero_grad()
            
            
            if (split) or (auxiliary):
            # Compute loss
                tr_output,out1,out2 = model(tr_input)
                te_output,out1,out2 = model(te_input)
            
            else:
                tr_output = model(tr_input)
                te_output = model(te_input)
                    
                
            tr_loss = criterion(tr_output, tr_label.long())
            te_loss = criterion(te_output, te_label.long())
            
            if (auxiliary):
                cl_loss_tr1 = criterion(out1,tr_classes.long()[:,0])
                cl_loss_tr2 = criterion(out2,tr_classes.long()[:,1])
                total_loss = tr_loss + auxiliary_f*cl_loss_tr1 + auxiliary_f*cl_loss_tr2
            else:
                total_loss = tr_loss
                
                
            total_loss.backward()
            optimizer.step()
                    
            # Add loss to epoch loss
            epoch_tr_loss += tr_loss.item()
            epoch_te_loss += te_loss.item()
        
                    
        # Compute average epoch loss and add it to losses array
        epoch_tr_loss = epoch_tr_loss/len(train_loader)
        epoch_te_loss = epoch_te_loss/len(test_loader)
        tr_losses.append(epoch_tr_loss)
        te_losses.append(epoch_te_loss)
    return tr_losses, te_losses