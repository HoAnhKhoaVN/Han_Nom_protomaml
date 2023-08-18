from torch import Tensor
from typing import Tuple

def split_batch(
    imgs: Tensor,
    targets: Tensor,
    )-> Tuple[Tensor, Tensor, Tensor, Tensor]:
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets

if __name__ == '__main__':
    print('utils.py')