#!/usr/bin/env python3
"""
File: nn_utils.py
Description: neural network utils 
"""
import torch



def train_model(model, num_epochs, train_loader,test_loader, flatten=True, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Copy all model parameters to the GPU
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    all_losses = []
    all_losses_t = []

    for epoch in range(num_epochs):
        total_loss_train = 0.0
        total_loss_test = 0.0
        for (inputs, labels), (inp, lab) in zip(train_loader, test_loader):
            for i in range(2):  # train on each element of the pair
                inputs_single = inputs[:, i, :, :].to(device)
                labels_single = labels.t()[i].to(device)
                inp_s = inp[:, i, :, :]
                lab_s = lab.t()[i]
                if flatten:
                    inputs_single = inputs_single.view(
                        -1, inputs_single.shape[1] * inputs_single.shape[2]
                    )
                    inp_s = inp_s.view(
                        -1, inp_s.shape[1] * inp_s.shape[2]
                    )
                else:
                    inp_s = inp_s.unsqueeze(1)
                    inputs_single = inputs_single.unsqueeze(1) # Add color channel 
                optimizer.zero_grad()
                outputs = model(inputs_single)
                output_t = model(inp_s)

                loss = criterion(outputs, labels_single.long())
                loss_t = criterion(output_t, lab_s.long())
                loss.backward()
                optimizer.step()

                total_loss_train += loss.item()
                total_loss_test += loss_t.item()
        avg_train = total_loss_train/len(train_loader)
        avg_test = total_loss_test/len(test_loader)
        all_losses.append(avg_train)
        all_losses_t.append(avg_test)
        if verbose:
           #print("Epoch %d, Batch %d, Loss=%.4f" % (epoch + 1, split + 1, total_loss_train / len(train_loader)))
            print("End of Epoch %d, Avg Loss=%.4f" % (epoch+1, avg_train))
       # if not(verbose):
           
        
    return all_losses, all_losses_t

