import argparse

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, help='Root save dataset', default='demo_ds')
    argparser.add_argument('--exp', type=str, help='Root experiment dataset', default='exp')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--epoch_test', type=int, help='epoch test number', default= 100)
    argparser.add_argument('--batch_size', type=int, help='Input batch of task', default= 128)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--output_lr', type=float, help='task-level update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--cuda', action='store_true', help='enables cuda')
    argparser.add_argument('--seed', type=int, help='seet for reproduce', default=2103)
    args = argparser.parse_args()

    return args