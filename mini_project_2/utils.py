import torch
import math

# Generate circle dataset
def generate_data(size=1000):
    # Create a random sample of points in [0, 1]^2.
    dataset = torch.Tensor(1000, 2).uniform_(0, 1)
    
    # Compute distance from radius
    distances = torch.pow(dataset - 0.5, 2).sum(1).sqrt()
    
    # Get labels using distance
    labels = torch.where(distances > (1 / math.sqrt((2 * math.pi))), torch.Tensor([0]), torch.Tensor([1]))
    
    # Convert labels to one hot encoding
    one_hot = torch.Tensor([[0., 1.] if l.item() == 1. else [1., 0.] for l in labels])
    
    return dataset, one_hot


# Train a given model 
def train(X_train, Y_train, model, optimizer, criterion, n_epochs=100, batch_size=10, verbose=False):

    for epoch in range(n_epochs): 
        loss = 0
        
        for batch in range(0, len(X_train), batch_size):

            # Build batch 
            X_batch = X_train.narrow(0, batch, batch_size)
            Y_batch = Y_train.narrow(0, batch, batch_size)
            
            # Forward pass
            output = model.forward(X_batch) # Model prediction
            loss += criterion.forward(output, Y_batch) # Compute the loss of the batch
            
            # Backward pass
            model.zero_grad()
            grad = criterion.backward() # Compute the gradient of the loss
            model.backward(grad) # Compute the model gradients using the loss gradient
            optimizer.step() # Update parameters using computed gradients
            
        if verbose:
            print("Epoch {}: Average loss={}".format((epoch + 1), loss/len(X_train)))
            
# Test a given model
def accuracy(model, X_test, Y_test):
    true = Y_test.max(1)[1] # Exctract labels from one hot
    pred = model.forward(X_test).max(1)[1] # Compute predictions
    count = (true == pred).sum().item() # Compare labels and predictions
    
    return count / len(X_test)
        