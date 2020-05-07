# this code was derived from a medium article:
# https://github.com/tkshnkmr/frcnn_medium_sample

import torch
import time
from utils import (
    get_model_instance_segmentation,
    collate_fn,
    get_transform,
    myOwnDataset,
)

if __name__ == "__main__":

    train_data_dir="my_data/contests"
    train_coco="options_coco.json"
    eval_dir="my_data/eval"
    train_batch_size=2
    train_shuffle_dl=True
    num_workers_dl=4
    num_classes=2
    lr=.005
    momentum=.9
    weight_decay=.005
    num_epochs=25
    save="option_model"

    print("Torch version:", torch.__version__)

    # create own Dataset
    my_dataset = myOwnDataset(
        root=train_data_dir, annotation=train_coco, transforms=get_transform()
    )

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=train_batch_size,
        shuffle=train_shuffle_dl,
        num_workers=num_workers_dl,
        collate_fn=collate_fn,
    )

    print(data_loader)

    # select device (whether GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("device: ",device)

    # DataLoader is iterable over Dataset
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        temp = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # print(imgs)
        print(temp)

    # variable = input("press soemthign")


    contest_detector = get_model_instance_segmentation(num_classes)

    # move model to the right device
    contest_detector.to(device)

    # parameters
    params = [p for p in contest_detector.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        start = time.time()
        contest_detector.train()
        i = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = contest_detector(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}, Time: {time.time()-start}s")

    # torch.save(contest_detector,save)