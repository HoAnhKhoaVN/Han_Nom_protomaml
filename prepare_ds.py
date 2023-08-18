import os
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Text, List, Tuple
import torchvision
from dataset import ImageDataset, FewShotBatchSampler, TaskBatchSampler

def load_dataset(
    args
)-> Tuple:
    DS_PATH = args.root
    N_WAY= args.n_way
    K_SHOT = args.k_spt
    BATCH_SIZE = args.batch_size

    # region Dataset and Dataloader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias= False),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    trainset = torchvision.datasets.ImageFolder(root=f'{DS_PATH}/train', transform=transform)
    valset = torchvision.datasets.ImageFolder(root=f'{DS_PATH}/val', transform=transform)
    testset = torchvision.datasets.ImageFolder(root=f'{DS_PATH}/test', transform=transform)
    classes = trainset.classes
    # endregion

    # region Convert to ImageDataset
    trainset_new = ImageDataset(
        imgs_lbs= trainset.imgs,
        targets = trainset.targets,
        img_transform= transform
        )
    
    valset_new = ImageDataset(
    imgs_lbs= valset.imgs,
    targets = valset.targets,
    img_transform= transform
    )

    testset_new = ImageDataset(
    imgs_lbs= testset.imgs,
    targets = testset.targets,
    img_transform= transform
    )
    # endregion

    # region Convert to TaskDataset
    train_protomaml_sampler = TaskBatchSampler(trainset_new.targets,
                                            include_query=True,
                                            N_way=N_WAY,
                                            K_shot=K_SHOT,
                                            batch_size=BATCH_SIZE)
    

    val_promaml_sampler = TaskBatchSampler(valset_new.targets,
                                         include_query=True,
                                         N_way=N_WAY,
                                         K_shot=K_SHOT,
                                         batch_size=32,  # We do not update the parameters, hence the batch size is irrelevant here
                                         shuffle=False)
    test_promaml_sampler = TaskBatchSampler(testset_new.targets,
                                         include_query=True,
                                         N_way=N_WAY,
                                         K_shot=K_SHOT,
                                         batch_size=32,  # We do not update the parameters, hence the batch size is irrelevant here
                                         shuffle=False)
    # endregion

    # region Loader
    train_promaml_loader = DataLoader(trainset_new,
                                         batch_sampler=train_protomaml_sampler,
                                         collate_fn=train_protomaml_sampler.get_collate_fn(),
                                         num_workers=2)
    val_promaml_loader = DataLoader(valset_new,
                                       batch_sampler=val_promaml_sampler,
                                       collate_fn=val_promaml_sampler.get_collate_fn(),
                                       num_workers=2)

    test_promaml_loader = DataLoader(testset_new,
                                       batch_sampler=test_promaml_sampler,
                                       collate_fn=val_promaml_sampler.get_collate_fn(),
                                       num_workers=2)
    # endregion
    return (
        train_promaml_loader,
        val_promaml_loader,
        test_promaml_loader,
        classes
    )




