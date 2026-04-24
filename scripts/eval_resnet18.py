#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class JSONDataset(Dataset):
    def __init__(self, entries: List[dict], base_dir: str = None, label_map: dict = None, transform=None):
        self.entries = entries
        self.base_dir = Path(base_dir) if base_dir else None
        self.transform = transform
        self.label_map = label_map
        self._detect_fields()
        self._build_label_map_if_needed()

    def _detect_fields(self):
        example = self.entries[0]
        possible_image_keys = [k for k in example.keys() if 'image' in k or 'path' in k or 'file' in k]
        self.image_key = possible_image_keys[0] if possible_image_keys else 'image'
        possible_label_keys = [k for k in example.keys() if 'label' in k or 'class' in k or 'category' in k]
        self.label_key = possible_label_keys[0] if possible_label_keys else 'label'

    def _build_label_map_if_needed(self):
        labels = [e.get(self.label_key) for e in self.entries]
        if all(isinstance(l, int) for l in labels):
            self.label_map = None
            self.num_classes = max(labels) + 1 if labels else 0
        else:
            if self.label_map is None:
                unique = sorted([str(l) for l in labels])
                self.label_map = {v: i for i, v in enumerate(unique)}
            self.num_classes = max(self.label_map.values()) + 1

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img_path = e.get(self.image_key)
        if self.base_dir and not os.path.isabs(img_path):
            img_path = str(self.base_dir / img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        raw_label = e.get(self.label_key)
        if isinstance(raw_label, int):
            label = raw_label
        else:
            label = self.label_map[str(raw_label)]
        return img, label, img_path


def load_json_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def evaluate(args):
    data = load_json_data(args.data_file)
    train_entries = [e for e in data if e.get('split') == 'train']
    if not train_entries:
        print('No entries with split=="train" found in the provided JSON.')
        return

    base_dir = Path(args.data_file).parent
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    label_map = None
    if args.label_map:
        with open(args.label_map, 'r') as f:
            label_map = json.load(f)

    dataset = JSONDataset(train_entries, base_dir=base_dir, label_map=label_map, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(4, os.cpu_count() or 1))

    num_classes = dataset.num_classes
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu')

    if args.pretrained and args.checkpoint is None:
        model = models.resnet18(pretrained=True)
        if num_classes != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
            try:
                model.load_state_dict(state)
            except Exception:
                model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})

    model = model.to(device)
    model.eval()

    misclassified = []
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels, paths in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            matches = (preds == labels)
            total += labels.size(0)
            correct += matches.sum().item()
            for i in range(labels.size(0)):
                if not matches[i].item():
                    item = {
                        'image': paths[i],
                        'true_label': int(labels[i].item()),
                        'pred_label': int(preds[i].item()),
                    }
                    misclassified.append(item)

    acc = correct / total if total else 0.0
    print(f'Evaluated {total} samples — Accuracy: {acc:.4f} ({correct}/{total})')
    print('Misclassifications:')
    for m in misclassified:
        print(f"{m['image']}  -> true: {m['true_label']}  pred: {m['pred_label']}")


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ResNet-18 on dataset described by a JSON file')
    p.add_argument('--data-file', required=True, help='Path to data_file.json')
    p.add_argument('--checkpoint', required=False, help='Optional model checkpoint (state_dict or saved dict)')
    p.add_argument('--label-map', required=False, help='Optional JSON mapping from string labels to ints')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--pretrained', action='store_true', help='Use torchvision pretrained weights')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
