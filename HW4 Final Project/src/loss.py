import torch.nn as nn


class PunchesLoss(nn.Module):
    def __init__(self, weight_punches=None):
        super(PunchesLoss, self).__init__()
        self.loss_weights = nn.BCELoss()

    def forward(self, pred, target):
        pred_stances = pred[3]
        trg_stances = target[3]
        loss = self.loss_sides(pred_stances, trg_stances)
        return loss
