# this code was derived from a medium article:
# https://github.com/tkshnkmr/frcnn_medium_sample
import argparse
import torch
import time
from PIL import ImageDraw, Image
import numpy as np
from utils import (
    get_model_instance_segmentation,
    collate_fn,
    get_transform,
    myOwnDataset,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_dir", help="directory of training images", required=True)
    parser.add_argument("--train_coco", help="path to coco dataset", required=True)
    parser.add_argument("--eval_dir", help="directory of evaluation images", default=None)
    parser.add_argument("--eval_dest", help="destination of evaluated images with boundings drawn", default=None)
    parser.add_argument("--train_batch_size", help="size of the batch during training", default=2)
    parser.add_argument("--train_shuffle_dl", help="boolean value to shuffle during training", default=True)
    parser.add_argument("--num_workers_dl", help="number of workers, default 4", default=4)
    parser.add_argument("--num_classes", help="number of categories in this coco dataset including the background", default=2)
    parser.add_argument("--lr", help="learning rate hyperparam", default=.005)
    parser.add_argument("--momentum", help="momentum hyperparam", default=.9)
    parser.add_argument("--weight_decay", help="weight decay hyperparam", default=.005)
    parser.add_argument("--num_epochs", help="number of training epochs", default=25)
    parser.add_argument("--save", help="path to save model for loading later", default=None)

    args = parser.parse_args()

    print("Torch version:", torch.__version__)

    # create own Dataset
    my_dataset = myOwnDataset(
        root=args.train_data_dir, annotation=args.train_coco, transforms=get_transform()
    )

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.train_shuffle_dl,
        num_workers=args.num_workers_dl,
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

        print(temp)


    contest_detector = get_model_instance_segmentation(args.num_classes)

    # move model to the right device
    contest_detector.to(device)

    # parameters
    params = [p for p in contest_detector.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    len_dataloader = len(data_loader)

    for epoch in range(int(args.num_epochs)):
        print(f"Epoch: {epoch+1}/{args.num_epochs}")
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


    # evaluation
    if args.eval_dir and args.eval_dest:

        for imgpath in os.listdir(args.eval_dir):

            if 'jpg' in imgpath:  
                eval_img = Image.open(args.eval_dir+"/"+imgpath)
                trans_img = [transforms(eval_img).to(device)]
                contest_detector.eval()
                eval_results = contest_detector(trans_img)
                boxes=[]
                scores=[]
                for i in range(len(eval_results[0]['scores'])):
                    if eval_results[0]['scores'][i] >= .7:
                        boxes.append(eval_results[0]['boxes'][i].cpu().detach().tolist())

                for box in boxes:
                    shape = [(box[0], box[1]), (box[2], box[3])] 
                    img1 = ImageDraw.Draw(eval_img)   
                    img1.rectangle(shape, outline ="red")
                    eval_img.save(f"{args.eval_dir}/{imgpath}")


    if args.save:
        torch.save(contest_detector,args.save)