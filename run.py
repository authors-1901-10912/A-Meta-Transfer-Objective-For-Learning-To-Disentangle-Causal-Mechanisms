#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
import pdb, nauka, os, sys


class root(nauka.ap.Subcommand):
    class train(nauka.ap.Subcommand):
        @classmethod
        def addArgs(kls, argp):
            mtxp = argp.add_mutually_exclusive_group()
            mtxp.add_argument("-w", "--workDir",        default=None,         type=str,
                help="Full, precise path to an experiment's working directory.")
            mtxp.add_argument("-b", "--baseDir",        action=nauka.ap.BaseDir)
            argp.add_argument("-d", "--dataDir",        action=nauka.ap.DataDir)
            argp.add_argument("-t", "--tmpDir",         action=nauka.ap.TmpDir)
            argp.add_argument("-n", "--name",           default=[],
                action="append",
                help="Build a name for the experiment.")
            argp.add_argument("-s", "--seed",           default=0,            type=int,
                help="Seed for PRNGs. Default is 0.")
            argp.add_argument("--model",                default="cat",        type=str,
                choices=["cat", "cat2", "gauss"],
                help="Model Selection.")
            argp.add_argument("-e", "--num-epochs",     default=200,          type=int,
                help="Number of epochs")
            argp.add_argument("--batch-size", "--bs",   default=256,          type=int,
                help="Batch Size")
            argp.add_argument("-Q", "--dpe",            default=1000,         type=int,
                help="Number of training distributions per epoch")
            argp.add_argument("--pretrain",             default=0,            type=int,
                help="Number of pretraining batches per distribution")
            argp.add_argument("-J", "--ipd",            default=100,          type=int,
                help="Number of interventions per distribution")
            argp.add_argument("-M", "--num-vars",       default=5,            type=int,
                help="Number of variables in system")
            argp.add_argument("-N", "--num-cats",       default=10,           type=int,
                help="Number of categories per variable, for categorical models")
            argp.add_argument("-R", "--cpi",            default=20,           type=int,
                help="Configurations per intervention")
            argp.add_argument("-T", "--xfer-epi-size",  default=10,           type=int,
                help="Transfer episode size")
            argp.add_argument("-v", "--verbose",        default=0,            type=int,
                nargs="?",   const=10,
                help="Printing interval")
            argp.add_argument("--cuda",                 action=nauka.ap.CudaDevice)
            argp.add_argument("-p", "--preset",         action=nauka.ap.Preset,
                choices={"default":  [],},
                help="Named experiment presets for commonly-used settings.")
            optp = argp.add_argument_group("Optimizers", "Tunables for all optimizers.")
            optp.add_argument("--model-optimizer", "--mopt", action=nauka.ap.Optimizer,
                default="nag:0.001,0.9",
                help="Model Optimizer selection.")
            optp.add_argument("--gamma-optimizer", "--gopt", action=nauka.ap.Optimizer,
                default="nag:0.0001,0.9",
                help="Gamma Optimizer selection.")
            optp.add_argument("--lmaxent",              default=0.000,        type=float,
                help="Regularizer for maximum entropy")
            optp.add_argument("--lsparse",              default=0.000,        type=float,
                help="Regularizer for maximum entropy")
            dbgp = argp.add_argument_group("Debugging", "Flags for debugging purposes.")
            dbgp.add_argument("--summary",              action="store_true",
                help="Print a summary of the network.")
            dbgp.add_argument("--fastdebug",            action=nauka.ap.FastDebug)
            dbgp.add_argument("--pdb",                  action="store_true",
                help="""Breakpoint before run start.""")
        
        @classmethod
        def run(kls, a):
            from   causal.experiment import Experiment;
            if a.pdb: pdb.set_trace()
            return Experiment(a).rollback().run().exitcode


def main(argv=sys.argv):
    argp = root.addAllArgs()
    try:    import argcomplete; argcomplete.autocomplete(argp)
    except: pass
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
