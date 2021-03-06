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
        return delta.detach(), torch.clamp(x+delta, 0, 1), torch.clamp(x-delta, 0, 1)

    def l2(self, x1, x2):
        return ((x1-x2)**2).mean(-1)

    def pert2(self, x, y, c1, c2, learning_rate, steps):
        x, y = x.to(self.device), y.to(self.device)
        delta1 = nn.Parameter(0.001*torch.rand_like(x))
        delta2 = nn.Parameter(0.001*torch.rand_like(x))
        optimizer = optim.Adam([delta1, delta2], lr=learning_rate)

        for id_step in range(steps):
            optimizer.zero_grad()
            delta_m = (delta1 + delta2)/2.
            adv1 = torch.clamp(x + delta1, 0, 1)
            adv2 = torch.clamp(x + delta2, 0, 1)
            adv_m = torch.clamp(x + delta_m, 0, 1)

            label_m = self.model(adv_m)
            label1 = self.model(adv1)
            label2 = self.model(adv2)

            loss = -self.l2(label1, label_m) - self.l2(label2, label_m)\
                   + (c1 * delta1**2 + c1 * delta2**2 + c2 * (delta1-delta2)**2).sum()
            loss.backward()
            optimizer.step()

            _, predicted1 = label1.max(1)
            _, predicted2 = label2.max(1)
            _, predicted_m = label_m.max(1)
            y1, y0, y2 = predicted1.item(), predicted_m.item(), predicted2.item()
            print("step:{}/{}, {}, {}, {}, {}".format(id_step, steps,loss.item(), y1, y0, y2))
            if (y1 != y0) and (y2 != y0):
                break
        return delta1.detach(), delta1.detach(), delta_m.detach(), adv1.detach(), adv2.detach(), adv_m.detach()

    def pert3(self, x, y, c1, c2, learning_rate, steps):
        x, y = x.to(self.device), y.to(self.device)
        delta1 = nn.Parameter(0.001*torch.rand_like(x))
        delta2 = nn.Parameter(0.001*torch.rand_like(x))
        optimizer = optim.Adam([delta1, delta2], lr=learning_rate)

        for id_step in range(steps):
            optimizer.zero_grad()
            delta_m = (delta1 + delta2)/2.
            adv1 = torch.clamp(x + delta1, 0, 1)
            adv2 = torch.clamp(x + delta2, 0, 1)
            adv_m = torch.clamp(x + delta_m, 0, 1)

            label_pre = self.model(x)
            y = label_pre.argmax(-1)
            label_m = self.model(adv_m)
            label1 = self.model(adv1)
            label2 = self.model(adv2)

            loss = self.criterion(label1, y) + self.criterion(label2, y) - self.criterion(label_m, y)\
                   + (c1 * delta1**2 + c1 * delta2**2 + c2 * (delta1-delta2)**2).sum()
            loss.backward()
            optimizer.step()

            _, predicted1 = label1.max(1)
            _, predicted2 = label2.max(1)
            _, predicted_m = label_m.max(1)
            y1, y0, y2 = predicted1.item(), predicted_m.item(), predicted2.item()
            print("step:{}/{}, {}, {}, {}, {}".format(id_step, steps,loss.item(), y1, y0, y2))
            if (y1 == y) and (y2 == y) and (y0 != y):
                break
        return delta1.detach(), delta1.detach(), delta_m.detach(), adv1.detach(), adv2.detach(), adv_m.detach()