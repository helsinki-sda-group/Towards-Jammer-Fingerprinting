import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from blitz.modules import BayesianLinear, BayesianLSTM
from blitz.utils import variational_estimator
torch.manual_seed(42)

@variational_estimator
class SimpleBayesianNetwork(nn.Module):
    def __init__(self, window, jammers_nbr):
        super(SimpleBayesianNetwork, self).__init__()
        self.lstm1 = BayesianLSTM(window, 70, prior_sigma_1=1.25, bias=True, prior_pi=0.9, posterior_rho_init=-9.0)
        self.lstm2 = BayesianLSTM(70, 30, prior_sigma_1=1.25, bias=True, prior_pi=0.9, posterior_rho_init=-3.5)
        self.linear = BayesianLinear(30, jammers_nbr, prior_sigma_1=1.25, bias=True)

    def forward(self, inputs):
        output, _ = self.lstm1(inputs)
        output, _ = self.lstm2(output)
        output = output[:, -1, :]
        output = self.linear(output)
        return F.softmax(output, dim=1)

def train(train_loader, val_loader, model, criterion, optimizer, epochs, filepath):
    """
        Run one train epoch
    """
    # switch to train mode
    best_acc = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        for inputs, target in train_loader:
            optimizer.zero_grad()
            loss = model.sample_elbo(inputs=inputs,
                            labels=target,
                            criterion=criterion,
                            sample_nbr=7,
                            complexity_cost_weight=1/(100000))
        
            loss.backward()
            optimizer.step()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(f'Iteration: {epoch} | Accuracy of the network: {100 * correct / total} %')

        total_pred, correct_pred = 0, 0
        with torch.no_grad():
            model.eval()
            for inputs, target in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_pred += target.size(0)
                correct_pred += (predicted == target).sum().item()
            if correct_pred / total_pred > best_acc:
                print(f'Epoch {epoch}: Validation accuracy improved from {best_acc} to {correct_pred/total_pred}.')
                best_acc = correct_pred / total_pred
                torch.save(model.state_dict(), filepath)
                best_model = copy.deepcopy(model)
            print('Validation | Accuracy of the network: {} %'.format(str(100 * correct_pred / total_pred)))
    return best_model

def validate(val_loader, model):
    model.eval()
    total = 0
    correct = 0

    for inputs, target in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('Accuracy of the network: {} %'.format(str(100 * correct / total)))
