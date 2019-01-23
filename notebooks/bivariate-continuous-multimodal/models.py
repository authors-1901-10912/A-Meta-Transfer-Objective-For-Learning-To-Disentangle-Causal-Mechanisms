import torch

from causal_meta.modules.gmm import GaussianMixture
from causal_meta.modules.mdn import MDN
from copy import deepcopy

def mdn(opt): 
    return MDN(opt.CAPACITY, opt.NUM_COMPONENTS)

def gmm(opt): 
    return GaussianMixture(opt.GMM_NUM_COMPONENTS)

def marginal_nll(opt, inp, nll): 
    model_g = gmm(opt)
    if opt.CUDA: 
        model_g = model_g.cuda()
    model_g.fit(inp)
    with torch.no_grad():
        loss_marginal = nll(model_g(inp), inp)
    return loss_marginal

def transfer_tune(opt, model, model_g, inp, tar, nll): 
    model = deepcopy(model)
    optim_model = torch.optim.Adam(model.parameters(), 
                                   opt.FINETUNE_LR)
    loss_marginal = marginal_nll(opt, inp, nll).item()
    joint_losses = []
    for iter_num in range(opt.FINETUNE_NUM_ITER): 
        # Train conditional
        prd = model(inp)
        loss_conditional = nll(prd, tar)
        optim_model.zero_grad()
        loss_conditional.backward()
        optim_model.step()
        joint_losses.append(loss_conditional.item() + loss_marginal)
    # Return losses
    return joint_losses

def auc_transfer_metric(opt, model, model_g, inp, tar, nll):
    # Tune
    losses = transfer_tune(opt, model, model_g, inp, tar, nll)
    # Compute the integral of the loss curve
    return sum(losses)
