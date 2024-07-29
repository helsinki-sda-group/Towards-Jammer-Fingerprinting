import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.manual_seed(42)
 
class LSTMTrad(nn.Module):
    def __init__(self, window, jammers_nbr):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=window, hidden_size=256, num_layers=1, batch_first=True, bias=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bias=True)
        self.linear = nn.Linear(128, jammers_nbr)

    def forward(self, inputs):
        output, _ = self.lstm1(inputs)
        output, _ = self.lstm2(output)
        output = output[:, -1, :]
        output = self.linear(output)
        return F.softmax(output, dim=1)

def train_traditional(n_epochs, model, train_loader, val_loader, criterion, optimizer, filepath):
    best_acc = 0
    best_model = None
    for epoch in range(n_epochs):
        model.train()
        total = 0
        correct = 0
        for inputs, target in train_loader:
            y_pred = model(inputs)
            loss = criterion(y_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print('Epoch: {} | Accuracy of the network: {} %'.format(str(epoch) ,str(100 * correct / total)))
        model.eval()
        total_pred = 0
        correct_pred = 0
        with torch.no_grad():
            model.eval()
            for inputs, target in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_pred += target.size(0)
                correct_pred += (predicted == target).sum().item()
            if correct_pred / total_pred > best_acc:
                print(f'Epoch {epoch}: Validation accuracy improved from {best_acc} to {best_acc}.')
                best_acc = correct_pred / total_pred 
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), filepath)
        print('Validation | Accuracy of the network: {} %'.format(str(100 * correct_pred / total_pred)))
    return best_model

def validate_traditional(val_loader, model):
    model.eval()
    total = 0
    correct = 0
    all_predicts = []
    for inputs, target in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predicts.append(predicted)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('Accuracy of the network: {} %'.format(str(100 * correct / total)))
    return all_predicts
