#inspired by clip_robustbench.py

import json
import os
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
import wandb
import argparse
from robustbench import benchmark
from robustbench.data import load_clean_dataset
from autoattack import AutoAttack
from robustbench.model_zoo.enums import BenchmarkDataset
from CLIP_eval.eval_utils import compute_accuracy_no_dataloader, load_clip_model
from train.utils import str2bool

parser = argparse.ArgumentParser(description="Script arguments")

parser.add_argument('--clip_model_name', type=str, default='none', help='ViT-L-14, ViT-B-32, don\'t use if wandb_id is set')
parser.add_argument('--pretrained', type=str, default='none', help='Pretrained model ckpt path, don\'t use if wandb_id is set')
parser.add_argument('--wandb_id', type=str, default='none', help='Wandb id of training run, don\'t use if clip_model_name and pretrained are set')
parser.add_argument('--logit_scale', type=str2bool, default=True, help='Whether to scale logits')
parser.add_argument('--full_benchmark', type=str2bool, default=False, help='Whether to run full RB benchmark')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
parser.add_argument('--cifar10_root', type=str, default='/mnt/datasets/CIFAR10', help='CIFAR10 dataset root directory')
parser.add_argument('--cifar100_root', type=str, default='/mnt/datasets/CIFAR100', help='CIFAR100 dataset root directory')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_samples_imagenet', type=int, default=5000, help='Number of samples from ImageNet for benchmark')
parser.add_argument('--n_samples_cifar', type=int, default=1000, help='Number of samples from CIFAR for benchmark')
parser.add_argument('--template', type=str, default='std', help='Text template type; std, ensemble')
parser.add_argument('--norm', type=str, default='linf', help='Norm for attacks; linf, l2')
parser.add_argument('--eps', type=float, default=4., help='Epsilon for attack')
parser.add_argument('--beta', type=float, default=0., help='Model interpolation parameter')
parser.add_argument('--alpha', type=float, default=2., help='APGD alpha parameter')
parser.add_argument('--experiment_name', type=str, default='', help='Experiment name for logging')
parser.add_argument('--blackbox_only', type=str2bool, default=False, help='Run blackbox attacks only')
parser.add_argument('--save_images', type=str2bool, default=False, help='Save images during benchmarking')
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')


CIFAR100_LABELS = ("beaver", "dolphin", "otter", "seal", "whale",
    "aquarium fish", "flatfish", "perch", "shark", "trout",
    "orchids", "poppies", "roses", "sunflowers", "tulips",
    "bottles", "bowls", "cans", "cups", "plates",
    "apple", "mushrooms", "oranges", "pears", "strawberries",
    "clock", "computer", "keyboard", "microwave", "toaster",
    "bed", "chair", "couch", "dining table", "toilet",
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    "bear", "leopard", "lion", "tiger", "wolf",
    "bridge", "castle", "house", "road", "skyscraper",
    "cloud", "forest", "mountain", "plain", "sea",
    "possum", "raccoon", "skunk", "weasel", "otter",
    "crab", "snail", "spider", "worm", "jellyfish",
    "baby", "boy", "girl", "man", "woman",
    "lizard", "snake", "turtle", "crocodile", "gecko",
    "hamster", "rabbit", "rat", "squirrel", "guinea pig",
    "maple", "oak", "palm", "pine", "willow",
    "bicycle", "bus", "motorcycle", "pickup truck", "train",
    "car", "limousine", "rocket", "taxi", "van",
    "bag", "belt", "bow", "handbag", "suitcase")

class ClassificationModel(torch.nn.Module):
    def __init__(self, model, text_embedding, args, input_normalize, resizer=None, logit_scale=True):
        super().__init__()
        self.model = model
        self.args = args
        self.input_normalize = input_normalize
        self.resizer = resizer if resizer is not None else lambda x: x
        self.text_embedding = text_embedding
        self.logit_scale = logit_scale

    def forward(self, vision, output_normalize=True):
        assert output_normalize
        embedding_norm_ = self.model.encode_image(
            self.input_normalize(self.resizer(vision)),
            normalize=True
        )
        logits = embedding_norm_ @ self.text_embedding
        if self.logit_scale:
            logits *= self.model.logit_scale.exp()
        return logits

def interpolate_state_dict(m1, beta=0.2):
    m = {}

    m2 = torch.load("/path/to/ckpt.pt", map_location='cpu')
    for k in m1.keys():
        # print(m1[k].shape, m2[k].shape)
        m[k] = beta * m1[k] + (1 - beta) * m2[k]
    return m


if __name__ == '__main__':
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse command-line arguments
    args = parser.parse_args()
    # print args
    print(f"Arguments:\n{'-' * 20}", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"{'-' * 20}")

    args.eps /= 255
    # make sure there is no string in args that should be a bool
    assert not any(
        [isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values(
        )])
    
    num_classes = 100
    data_dir = args.cifar100_root
    n_samples = args.n_samples_cifar
    resizer = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=False)

    eps = args.eps

    # init wandb
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    wandb_user, wandb_project = None, None
    while True:
        try:
            run_eval = wandb.init(
                project=wandb_project,
                job_type='eval',
                name=f'{"rb" if args.full_benchmark else "aa"}-clip-{args.dataset}-{args.norm}-{eps:.2f}'
                     f'-{args.wandb_id if args.wandb_id is not None else args.pretrained}-{args.blackbox_only}-{args.beta}',
                save_code=True,
                config=vars(args),
                mode='online' if args.wandb else 'disabled'
            )
            break
        except wandb.errors.CommError as e:
            print('wandb connection error', file=sys.stderr)
            print(f'error: {e}', file=sys.stderr)
            time.sleep(1)
            print('retrying..', file=sys.stderr)

    if args.devices != '':
        # set cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    main_device = 0
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("No multiple GPUs available.")

    if not args.blackbox_only:
        attacks_to_run = ['apgd-ce', 'apgd-t']
    else:
        attacks_to_run = ['square']
    print(f'[attacks_to_run] {attacks_to_run}')

    clip_model_name = args.clip_model_name
    pretrained = args.pretrained
    run_train = None
    del args.clip_model_name, args.pretrained

    print(f'[loading pretrained clip] {clip_model_name} {pretrained}')

    model, preprocessor_without_normalize, normalize = load_clip_model(clip_model_name, pretrained, args.beta)

    print(f'[resizer] {resizer}')
    print(f'[preprocessor] {preprocessor_without_normalize}')

    model.eval()
    model.to(main_device)
    with torch.no_grad():
        if args.template == 'std':
            template = 'This is a photo of a {}'
        else:
            raise ValueError(f'Unknown template: {args.template}')
        print(f'template: {template}')
        texts = [template.format(c) for c in CIFAR100_LABELS]
        text_tokens = open_clip.tokenize(texts)
        embedding_text_labels_norm = []
        text_batches = [text_tokens]
        for el in text_batches:
            # we need to split the text tokens into two batches because otherwise we run out of memory
            # note that we are accessing the model directly here, not the CustomModel wrapper
            # thus its always normalizing the text embeddings
            embedding_text_labels_norm.append(
                model.encode_text(el.to(main_device), normalize=True).detach().cpu()
            )
        model.cpu()
        embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(main_device)

        assert torch.allclose(
            F.normalize(embedding_text_labels_norm, dim=0),
            embedding_text_labels_norm
        )
        if clip_model_name == 'ViT-B-32':
            assert embedding_text_labels_norm.shape == (512, num_classes), embedding_text_labels_norm.shape
        elif clip_model_name == 'ViT-L-14':
            assert embedding_text_labels_norm.shape == (768, num_classes), embedding_text_labels_norm.shape
        else:
            raise ValueError(f'Unknown model: {clip_model_name}')

    # get model
    model = ClassificationModel(
        model=model,
        text_embedding=embedding_text_labels_norm,
        args=args,
        resizer=resizer,
        input_normalize=normalize,
        logit_scale=args.logit_scale,
    )

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    model_name = None
    # device = [torch.device(el) for el in range(num_gpus)]  # currently only single gpu supported
    device = torch.device(main_device)
    torch.cuda.empty_cache()

    dataset_short = 'c100'

    start = time.time()
    if args.full_benchmark:
        clean_acc, robust_acc = benchmark(
            model, model_name=model_name, n_examples=n_samples,
            batch_size=args.batch_size,
            dataset=args.dataset, data_dir=data_dir,
            threat_model=args.norm.replace('l', 'L'), eps=eps,
            preprocessing=preprocessor_without_normalize,
            device=device, to_disk=False
            )
        clean_acc *= 100
        robust_acc *= 100
        duration = time.time() - start
        print(f"[Model] {pretrained}")
        print(
            f"[Clean Acc] {clean_acc:.2f}% [Robust Acc] {robust_acc:.2f}% [Duration] {duration / 60:.2f}m"
            )
        if run_train is not None:
            # reload the run to make sure we have the latest summary
            del api, run_train
            api = wandb.Api()
            run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
            eps_descr = str(int(eps * 255)) if args.norm == 'linf' else str(eps)
            run_train.summary.update({f'rb/acc-{dataset_short}': clean_acc})
            run_train.summary.update({f'rb/racc-{dataset_short}-{args.norm}-{eps_descr}': robust_acc})
            run_train.update()
    else:
        adversary = AutoAttack(
            model, norm=args.norm.replace('l', 'L'), eps=eps, version='custom', attacks_to_run=attacks_to_run, verbose=True
        )

        x_test, y_test = load_clean_dataset(
            BenchmarkDataset(args.dataset), n_examples=n_samples, data_dir=data_dir,
            prepr=preprocessor_without_normalize,)

        acc = compute_accuracy_no_dataloader(model, data=x_test, targets=y_test, device=device, batch_size=args.batch_size)
        acc = acc*100
        print(f'[acc] {acc:.2f}%', flush=True)
        x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)  # y_adv are preds on x_adv
        racc = compute_accuracy_no_dataloader(model, data=x_adv, targets=y_test, device=device, batch_size=args.batch_size) * 100
        print(f'[acc] {acc:.2f}% [racc] {racc:.2f}%')

        # possibility to save adv images (see in clip_robustbench)

        # write to wandb
        if run_train is not None:
            # reload the run to make sure we have the latest summary
            del api, run_train
            api = wandb.Api()
            run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
            if args.dataset == 'imagenet':
                assert args.norm == 'linf'
                eps_descr = str(int(eps * 255))
                if eps_descr == '4':
                    descr = dataset_short
                else:
                    descr = f'{dataset_short}-eps{eps_descr}'
                if n_samples != 5000:
                    acc = f'{acc:.2f}*'
                    racc = f'{racc:.2f}*'
            elif args.dataset == 'cifar10':
                if args.norm == 'linf':
                    descr = dataset_short
                else:
                    descr = f'{dataset_short}-{args.norm}'
                if n_samples != 10000:
                    acc = f'{acc:.2f}*'
                    racc = f'{racc:.2f}*'
            else:
                raise ValueError(f'Unknown dataset: {args.dataset}')
            run_train.summary.update({f'aa/acc-{dataset_short}': acc})
            run_train.summary.update({f'aa/racc-{descr}': racc})
            run_train.summary.update()
            run_train.update()
    run_eval.finish()