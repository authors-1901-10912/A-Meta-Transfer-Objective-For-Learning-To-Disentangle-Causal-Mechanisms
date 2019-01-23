import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_meta.utils.torch_utils import logsumexp
from tqdm import tnrange, tqdm_notebook
from argparse import Namespace
from copy import deepcopy

def train_nll(opt, model, scm, train_distr_fn, polarity='X2Y', loss_fn=nn.MSELoss(),
              decoder=None, encoder=None):
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    if opt.CUDA:
        model = model.cuda()
        if encoder is not None: 
            encoder = encoder.cuda()
        if decoder is not None: 
            decoder = decoder.cuda()
    frames = []
    for iter_num in tnrange(opt.NUM_ITER, leave=False):
        # Generate samples from the training distry
        X = train_distr_fn()
        with torch.no_grad():
            Y = scm(X)
        if opt.CUDA:
            X, Y = X.cuda(), Y.cuda()
        with torch.no_grad():
            if decoder is not None:
                # X and Y are sampled from the underlying distribution.
                # We apply a secret random transformation to the true latent
                # variables to obtain the raw input
                X, Y = decoder(X, Y)
            if encoder is not None:
                # Apply the encoder, meant to "undo" the decoder up to swapping X and Y.
                X, Y = encoder(X, Y)
        # Now, train as usual
        if polarity == 'X2Y':
            inp, tar = X, Y
        elif polarity == 'Y2X':
            inp, tar = Y, X
        else:
            raise ValueError
        if opt.CUDA:
            inp, tar = inp.cuda(), tar.cuda()
        # Train
        out = model(inp)
        loss = loss_fn(out, tar)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Append info
        if iter_num % opt.REC_FREQ or iter_num == (opt.NUM_ITER - 1):
            info = Namespace(loss=loss.item(),
                             iter_num=iter_num)
            frames.append(info)
    return frames

def train_alpha(opt, model_x2y, model_y2x, model_g2y, model_g2x, alpha, gt_scm, 
                distr, sweep_distr, nll, transfer_metric, mixmode='logmix'):
    # Everyone to CUDA
    if opt.CUDA: 
        model_x2y.cuda()
        model_y2x.cuda()
    alpha_optim = torch.optim.Adam([alpha], lr=opt.ALPHA_LR)
    frames = []
    iterations = tnrange(opt.ALPHA_NUM_ITER, leave=False)
    for iter_num in iterations:
        # Sample parameter for the transfer distribution
        sweep_param = sweep_distr()
        # Sample X from transfer
        X_gt = distr(sweep_param)
        Y_gt = gt_scm(X_gt)
        with torch.no_grad():
            if opt.CUDA:
                X_gt, Y_gt = X_gt.cuda(), Y_gt.cuda()
        # Evaluate performance
        metric_x2y = transfer_metric(opt, model_x2y, model_g2x, X_gt, Y_gt, nll)
        metric_y2x = transfer_metric(opt, model_y2x, model_g2y, Y_gt, X_gt, nll)
        # Estimate gradient
        if mixmode == 'logmix':
            loss_alpha = torch.sigmoid(alpha) * metric_x2y + (1 - torch.sigmoid(alpha)) * metric_y2x
        else:
            log_alpha, log_1_m_alpha = F.logsigmoid(alpha), F.logsigmoid(-alpha)
            as_lse = logsumexp(log_alpha + metric_x2y, log_1_m_alpha + metric_y2x)
            if mixmode == 'logsigp': 
                loss_alpha = as_lse
            elif mixmode == 'sigp':
                loss_alpha = as_lse.exp()
        # Optimize
        alpha_optim.zero_grad()
        loss_alpha.backward()
        alpha_optim.step()
        # Append info
        with torch.no_grad():
            frames.append(Namespace(iter_num=iter_num,
                                    alpha=alpha.item(), 
                                    sig_alpha=torch.sigmoid(alpha).item(), 
                                    metric_x2y=metric_x2y, 
                                    metric_y2x=metric_y2x, 
                                    loss_alpha=loss_alpha.item()))
        iterations.set_postfix(alpha='{0:.4f}'.format(torch.sigmoid(alpha).item()))
    return frames

def make_alpha(opt):
    alpha = nn.Parameter(torch.tensor(0.).to('cuda' if opt.CUDA else 'cpu'))
    return alpha
