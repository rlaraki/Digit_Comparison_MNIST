# Copied from practical 5 solutions.


def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 250

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
######################################################################
            
def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

######################################################################

#####################################################################

######################################################################

def get_stats(skip_connections, batch_normalization, nb_samples = 100):

    model = ResNet(nb_residual_blocks = 30, nb_channels = 10,
                   kernel_size = 3, nb_classes = 10,
                   skip_connections = skip_connections, batch_normalization = batch_normalization)

    criterion = nn.CrossEntropyLoss()

    monitored_parameters = [ b.conv1.weight for b in model.resnet_blocks ]

    result = torch.empty(len(monitored_parameters), nb_samples)

    for n in range(nb_samples):
        output = model(train_input[n:n+1])
        loss = criterion(output, train_targets[n:n+1])
        model.zero_grad()
        loss.backward()
        for d, p in enumerate(monitored_parameters):
            result[d, n] = p.grad.norm()

    return result
