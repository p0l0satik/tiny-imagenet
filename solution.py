# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys
import os
from sklearn import datasets

# List any extra packages you need here
PACKAGES_TO_INSTALL = ["gdown==4.4.0", "torchvision", "einops"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
from torch import optim, nn
from torchvision import transforms as T
from torchvision import datasets as datasetss
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from einops import rearrange

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 1024
num_epochs = 30

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    data = os.path.join(path, kind)
    dataset = datasetss.ImageFolder(data, transform=T.ToTensor())
    
    
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(kind=="train"), **kwargs)
    
    return dataloader

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    resnet18 = models.resnet18(num_classes=200)
    model = nn.DataParallel(resnet18, device_ids=[0, 1])
    model = model.to(device)
    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    
    loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
    writer = SummaryWriter(f'runs/tim')
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train(True) # enable dropout / batch_norm training behavior

        train_accuracy_batch = []

        for batch_no, (X_batch, y_batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.zero_grad()
            # transferring batch to GPU
            X_batch_gpu = X_batch.to(device)
            y_batch = y_batch.to(device)
            # forward propagation through the model
            prediction = model(X_batch_gpu)
            # calculating loss
            loss_val = loss(prediction, y_batch)
            # backward propagation through the model
            loss_val.backward()
            # optimizer step
            optimizer.step()

            train_loss.append(loss_val.item())

            accuracy = compute_accuracy(prediction, y_batch, device=device)
            train_accuracy_batch.append(accuracy.item())

            if batch_no % 15 == 0:
                # plot_loss_and_accuracy(train_loss, train_accuracy, val_accuracy, clear_output=True)
                # print(f'epoch {epoch} training stage...')
                

                writer.add_scalar('training loss', loss_val.item(), len(train_loss))

        train_accuracy_overall = np.mean(train_accuracy_batch) * 100
        train_accuracy.append(train_accuracy_overall.item())

        print(f'epoch {epoch} testing stage...')
        model.train(False) # disable dropout / use averages for batch_norm
        val_accuracy_batch = []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_dataloader):
                # transferring batch to GPU
                X_batch_gpu = X_batch.to(device)
                # y_batch = to
                # forward propagation through the model
                prediction = model(X_batch_gpu)

                # let's calculate the accuracy:
                accuracy = compute_accuracy(prediction, y_batch, device=device)
                val_accuracy_batch.append(accuracy.item())
            
                # sending pictures to TensorBoard (don't think about this for now)
                # writer.add_figure('predictions vs. actuals', plot_classes_preds(model, X_batch_gpu, y_batch), global_step=epoch)

            val_accuracy_overall = np.mean(val_accuracy_batch) * 100
            val_accuracy.append(val_accuracy_overall.item())
            writer.add_scalar('accuracy', val_accuracy_overall.item(), len(val_accuracy))

def compute_loss(prediction, y_true, device='cuda:0'):
    y_true_on_device = y_true.to(device)
    return nn.functional.nll_loss(prediction, y_true_on_device)

def compute_accuracy(prediction, y_true, device='cuda:0'):
    y_pred = torch.argmax(prediction, dim=1)
    y_true_on_device = y_true.to(device)
    accuracy = (y_pred == y_true_on_device).float().mean()
    return accuracy

# functions for tensorboard logging:
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images).cpu()
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [torch.exp(el)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 12))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        plt.imshow(rearrange(images[idx].cpu(), 'c h w -> h w c'))
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here




def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "747822ca4436819145de8f9e410ca9ca"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"

    return md5_checksum, google_drive_link
