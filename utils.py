import logging
import os
import random

import torch
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)

    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.maximum(distances, torch.Tensor([0]).to(device))

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]

    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0])
    indices_not_equal = torch.logical_not(indices_equal).to(device)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

    # Combine the two masks
    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

    mask = torch.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.shape[0])#[batch, batch]
    indices_not_equal = torch.logical_not(indices_equal) #[batch, batch]
    i_not_equal_j = indices_not_equal.unsqueeze(2) #[batch, batch, 1] [batch, batch, ||]
    i_not_equal_k = indices_not_equal.unsqueeze(1) #[batch, 1, batch] [batch, batch, ==]
    j_not_equal_k = indices_not_equal.unsqueeze(0) #[1, batch, batch] [[], batch, batch]

    # loáşĄi báť 3 th A=P, A=N, P=N
    distinct_indices =torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # Combine the two masks
    mask = torch.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin  #(batch, batch, batch) T(0,1,1)

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = torch.mul(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets) 
    triplet_loss = torch.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.greater(triplet_loss, 1e-16)
    num_positive_triplets = torch.sum(valid_triplets) # Tong so triplets > 0
    num_valid_triplets = torch.sum(mask) # Tong so triplets hop le
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared) #(batch_size, batch_size)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels) #(batch_size, batch_size)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + margin, torch.Tensor([0]))

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def dist_two_emb(emb1, emb2, epsilon = 10e-10, squared = False):
    """ Tinh khoang cach giua 2 vector emb1 va emb2

    Args:
        emb1: embeding vector, of size (batch_size, embeding_dim)
        emb2: embeding vector, of size (batch_size, embeding_dim)
        squared (bool, optional): Defaults to False.
    """
    minus = emb1 - emb2
    minus_square = torch.pow(minus, 2) #(batch_size, embeding_dim)
    dis = torch.sum(minus_square, dim=1, keepdim=True) #(batch_size, 1)
    if squared:
        return dis
    else:
        dis = torch.sqrt(dis + epsilon)
        return dis

def batch_weight_triplet_loss(labels, embeddings, margin=1, epsilon = 1e-10 ,squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # print('embeddings: ', embeddings)
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared) #(batch_size, batch_size)
    # print('pairwise_dis: ',pairwise_dist)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels) #(binary) (batch_size, batch_size)
    embed = embeddings.unsqueeze(0)
    # print("mask_anchor_positive: ",mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist) #(batch_size, batch_size)
    anchor_positive_weight = embed*(anchor_positive_dist.unsqueeze(2)) #(batch_size, batch_size, embed)
    anchor_positive_weight = torch.sum(anchor_positive_weight, dim=1) #(batch_size, embed)
    sum_weight = torch.sum(anchor_positive_dist, dim=1, keepdim=True) #(batch_size, 1)
    anchor_positive_weight = anchor_positive_weight/(sum_weight+epsilon) #(batch_size, embed)
    tem_sum_positive = torch.sum(anchor_positive_weight, dim=1, keepdim=True)
    tem_sum_positive = torch.where(tem_sum_positive == 0, torch.tensor(0.).to(device), torch.tensor(1.).to(device))

    # print("sum: ", sum_weight)
    # print("anchor_positive_weight: ", anchor_positive_weight)

    # shape (batch_size, 1)
    hardest_positive_dist = dist_two_emb(anchor_positive_weight, embeddings)
    # print('hardest_positive_dist: ', hardest_positive_dist)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float() #(batch_size, batch_size)

    invert_dist = 1/(pairwise_dist + epsilon)
    anchor_negative_invert_dist = torch.mul(invert_dist, mask_anchor_negative) #(batch_size, batch_size)
    anchor_negative_weight = embed*(anchor_negative_invert_dist.unsqueeze(2)) #(batch_size, batch_size, embed)
    anchor_negative_weight = torch.sum(anchor_negative_weight, dim=1) #(batch_size, embed)
    sum_weight = torch.sum(anchor_negative_invert_dist, dim=1, keepdim=True) #(batch_size, 1)
    anchor_negative_weight = anchor_negative_weight/(sum_weight+epsilon) #(batch_size, embed)
    tem_sum_negative = torch.sum(anchor_negative_weight, dim=1, keepdim=True)
    tem_sum_negative = torch.where(tem_sum_negative == 0, torch.tensor(0.).to(device), torch.tensor(1.).to(device))

    tem_and = torch.logical_and(tem_sum_negative, tem_sum_positive)
    # shape (batch_size, 1)
    hardest_negative_dist = dist_two_emb(anchor_negative_weight, embeddings)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + margin, torch.Tensor([0]).to(device))
    triplet_loss = torch.mul(tem_and, triplet_loss)
    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss


def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(
        true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)
    return true_dist.to(device)


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_model(arch='ResNet18'):
    if arch == 'ResNet18':
        model = ResNet18()
    elif arch == 'SENet18':
        model = SENet18()
    elif arch == 'DenseNet':
        model = densenet_cifar()
    elif arch == 'VGG19':
        model = VGG('VGG19')
    elif arch == 'PreActResNet18':
        model = PreActResNet18()
    elif arch == 'PreActResNet34':
        model = PreActResNet34()
    elif arch == 'DLA':
        model = DLA()
    elif arch == 'DPN':
        model = DPN26()
    return model


def random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, logfile='output.log'):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(state, epoch, is_best, save_path, save_freq=10):
    filename = os.path.join(save_path, 'checkpoint_' + str(epoch) + '.tar')
    if epoch % save_freq == 0:
        torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, 'best_checkpoint.tar')
        torch.save(state, best_filename)
