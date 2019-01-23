import torch
import torch.nn as nn
import warnings


class XCoder(nn.Module):
    def __init__(self, mod):
        super(XCoder, self).__init__()
        self.mod = mod

    def forward(self, X, Y):
        XY = torch.cat([X, Y], dim=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            XY = self.mod(XY)
        X, Y = XY[:, 0:1], XY[:, 1:2]
        return X, Y


class Rotor(nn.Module):
    def __init__(self, init_theta=0.):
        super(Rotor, self).__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta).float())

    def make_rotmat(self):
        mat = self.theta.new(2, 2)
        mat[0, 0] = self.theta.cos()
        mat[0, 1] = -self.theta.sin()
        mat[1, 0] = self.theta.sin()
        mat[1, 1] = self.theta.cos()
        return mat

    def forward(self, X, Y):
        XY = torch.cat([X, Y], dim=-1)
        rotmat = self.make_rotmat()
        XY = XY @ rotmat
        X, Y = XY[:, 0:1], XY[:, 1:2]
        return X, Y


class XSeq(nn.Sequential):
    # noinspection PyMethodOverriding
    def forward(self, X, Y):
        for module in self._modules.values():
            X, Y = module(X, Y)
        return X, Y
