import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class FindFlaws_simple(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()

    def pert(self, x, y, c, learning_rate, steps):
        x, y = x.to(self.device), y.to(self.device)
        delta = nn.Parameter(torch.zeros_like(x))
        optimizer = optim.Adam([delta], lr=learning_rate)

        for id_step in range(steps):
            optimizer.zero_grad()
            adv1 = torch.clamp(x+delta, 0, 1)
            adv2 = torch.clamp(x-delta, 0, 1)
            label1 = self.model(adv1)
            label2 = self.model(adv2)
            loss = -self.criterion(label1, y) - self.criterion(label2, y) + c * (delta**2).sum()
            loss.backward()
            optimizer.step()

            _, predicted1 = label1.max(1)
            _, predicted2 = label2.max(1)
            y1, y0, y2 = predicted1.item(), y.item(), predicted2.item()
            # print("step:{}/{}, {}, {}, {}".format(id_step, steps, y1, y0, y2))
            if (y1!=y0) and (y2!=y0):
                break
        return delta, torch.clamp(x+delta, 0, 1), torch.clamp(x-delta, 0, 1)

