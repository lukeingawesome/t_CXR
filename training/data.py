import io
import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
import tarfile
import zipfile
import glob
import yaml
from pathlib import Path
import wids

# import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image, TarIO, ImageFile
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from torch.utils.data import default_collate, ConcatDataset
from health_multimodal.image.data.io import load_image
from .distributed import is_master

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import warnings
warnings.filterwarnings("ignore")
import re
# from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Truncated File Read
Image.MAX_IMAGE_PIXELS = None # DecompressionBombWarning
ImageFile.MAX_IMAGE_PIXELS = None



import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import random
def apply_dropout(text, age_dropout=0.5, view_dropout=0.3, gender_dropout=0.5, bmi_dropout=0.3):
    # 30% chance to drop each attribute
    if random.random() < view_dropout:
        # Replace view position
        text = re.sub(r"This is a (\w+) view, CXR image of the whole chest.", "", text)
    
    if random.random() < age_dropout:
        # Replace age
        text = re.sub(r"The patient's age is (\d+).", "", text)

    if random.random() < bmi_dropout:
        # Replace age
        text = re.sub(r"The patient's bmi is (\d+).", "", text)
    
    if random.random() < gender_dropout:
        # Replace gender
        text = re.sub(r"The patient's gender is (\w+).", "", text)
    
    return text.replace('  ', ' ').strip()

def shuffle_sentences(text, probability=0.5):
    # If no markers or probability check fails, return original text
    if not ('[FINDINGS]' in text or '[IMPRESSION]' in text) or random.random() >= probability:
        return text
        
    # Find the marker position
    findings_pos = text.find('[FINDINGS]')
    impression_pos = text.find('[IMPRESSION]')
    
    # Determine which marker appears first (if both present)
    if findings_pos >= 0 and impression_pos >= 0:
        marker_pos = min(findings_pos, impression_pos)
        marker = '[FINDINGS]' if findings_pos == marker_pos else '[IMPRESSION]'
    else:
        # Use whichever marker is present
        marker_pos = findings_pos if findings_pos >= 0 else impression_pos
        marker = '[FINDINGS]' if findings_pos >= 0 else '[IMPRESSION]'
    
    # Split the text into parts: before marker and after marker
    pre_marker = text[:marker_pos + len(marker)]
    post_marker = text[marker_pos + len(marker):]
    
    # Only shuffle the sentences after the marker
    if post_marker:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', post_marker.strip())
        # Shuffle the sentences
        random.shuffle(sentences)
        # Rejoin the text
        shuffled_text = pre_marker + ' ' + ' '.join(sentences)
        return shuffled_text.replace('  ', ' ').strip()
    
    return text

def shuffle_sentences2(text, probability=0.5):
    # Split the text into sentences using a regex to account for periods that end sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Shuffle the sentences
    if random.random() < probability:
        random.shuffle(sentences)
    
    # Join the shuffled sentences back into a single string
    shuffled_text = ' '.join(sentences)
    return shuffled_text


class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None, img_key='image_path', caption_key='caption', tokenizer=None, is_train=True, separator='!@#$%^&*()'):
        """
        Args:
            csv_file (string): Path to the csv file
            transform (callable, optional): Optional transform to be applied on images
            img_key (string): Column name for image paths
            caption_key (string): Column name for captions
            tokenizer (callable, optional): Optional tokenizer for processing captions
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.img_key = img_key
        self.caption_key = caption_key
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.separator = separator
        self.max_length = 512
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """Returns one sample of data"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and caption
        oid = self.data_frame.iloc[idx]['object_id']
        prev_path = Path(self.data_frame.iloc[idx]['img_path'])
        cur_path = Path(self.data_frame.iloc[idx]['previous_img_path'])
        caption = str(self.data_frame.iloc[idx]['caption'])
        label = self.data_frame.iloc[idx]['label']
        try:
            instruction = self.data_frame.iloc[idx]['instruction']
        except:
            instruction = 'Retrieve the CXR image that is similar to the given report.; '
        # metadata = ''
        if self.is_train:
            # metadata = apply_dropout(metadata)
            caption = shuffle_sentences(caption, probability=0.2)
        caption = instruction + self.separator + caption
        # Load and process image
        prev_image = load_image(prev_path)
        cur_image = load_image(cur_path)
        if self.transform:
            prev_image = self.transform(prev_image)
            cur_image = self.transform(cur_image)
        
        # # Process caption if tokenizer is provided
        # if self.tokenizer:
        #     caption = self.tokenizer([caption])[0]
        # caption = metadata + ' ' + caption
        return prev_image, cur_image, caption, oid, label
    def collate_fn(self, batch):
        prev_images, cur_images, captions, oids, labels = zip(*batch)
        prev_images = torch.stack(prev_images)
        cur_images = torch.stack(cur_images)
        texts_2 = []
        original_texts = []
        for text in captions:
            t = text.split(self.separator)
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return prev_images, cur_images, original, oids, labels
        

class CustomZeroshotDataset(Dataset):
    def __init__(self, csv_file, transform=None, img_key='img_path', tokenizer=None, separator='!@#$%^&*()'):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.img_key = img_key
        self.tokenizer = tokenizer
        self.separator = separator
        self.instruction = 'Determine the change or the status of the '
        self.max_length = 512
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        """Returns one sample of data"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and caption
        cur_path = Path(self.data_frame.iloc[idx]['img_path'])
        prev_path = Path(self.data_frame.iloc[idx]['previous_img_path'])
        label = str(self.data_frame.iloc[idx]['label'])
        quality = str(self.data_frame.iloc[idx]['label_quality'])
        findings = str(self.data_frame.iloc[idx]['findings'])
        improving = str(self.data_frame.iloc[idx]['improving'])
        stable = str(self.data_frame.iloc[idx]['stable'])
        worsening = str(self.data_frame.iloc[idx]['worsening'])
        findings = findings.replace('_', ' ')
        try:
            instruction = self.data_frame.iloc[idx]['instruction']
        except:
            instruction = self.instruction + findings + '.; '
        # Load and process image
        # metadata = ''
        # Load and process image
        prev_image = load_image(prev_path)
        cur_image = load_image(cur_path)
        if self.transform:
            prev_image = self.transform(prev_image)
            cur_image = self.transform(cur_image)
        
        status = ['improving', 'stable', 'worsening']
        improving = instruction + self.separator + ' ' + improving
        stable = instruction + self.separator + ' ' + stable
        worsening = instruction + self.separator + ' ' + worsening
        labels = int(status.index(label))
        return cur_image, prev_image, quality, improving, stable, worsening, labels, findings
    def tokenize(self, texts):
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split(self.separator)
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original
    def collate_fn(self, batch):
        cur_images, prev_images, qualities, improving, stable, worsening, labels, findings = zip(*batch)
        cur_images = torch.stack(cur_images)
        prev_images = torch.stack(prev_images)
        label = torch.tensor(labels)
        improving = self.tokenize(improving)
        stable = self.tokenize(stable)
        worsening = self.tokenize(worsening)

        return cur_images, prev_images, qualities, improving, stable, worsening, label, findings
            

# Example usage:



## Old
class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.zfile = self.read_zipfile()
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts
    
class RetrievalDataset(Dataset):
    def __init__(self, input_filename, transforms, img_root=None, img_feature_path=None,
                text_feature_path=None, tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        
        self.meta = json.load(open(input_filename, 'r'))
        self.img_features, self.text_features = None, None
        if img_feature_path:
            if Path(img_feature_path).suffix == '.npy':
                self.img_features = np.memmap(img_feature_path, dtype='float32', mode='r', 
                                        shape=(len(self.meta), 5, 4096))
                logging.warning('np.memmap has some bugs.')
            elif Path(img_feature_path).suffix == '.dpt':
                self.img_features = torch.load(img_feature_path) 
                       
        if text_feature_path:
            if Path(text_feature_path).suffix == '.npy':
                self.text_features = np.memmap(text_feature_path, dtype='float32', mode='r', 
                                        shape=(len(self.meta), 5, 4096))
                logging.warning('np.memmap has some bugs.')
            elif Path(text_feature_path).suffix == '.dpt':
                self.text_features = torch.load(text_feature_path)
    
        self.img_root = img_root
        self.transforms = transforms
        logging.debug('Done loading data.')
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        images, texts = None, None
        if self.img_features is not None:
            images = self.img_features[str(idx)]

        if self.text_features is not None:
            if isinstance(self.text_features, dict):
                texts = np.array(self.text_features[str(idx)])
            else:
                texts = np.array(self.text_features[idx])

        if images is None:
            image_path = os.path.join(self.img_root, self.meta[idx]['image'])
            images = self.transforms(Image.open(image_path))
        if texts is None:
            texts = self.meta[idx]['caption']
            if self.tokenize:
                texts = self.tokenize([texts])[0]     
         
        return images, texts
    
    
class JsonDataset(Dataset):
    def __init__(self, input_filename, transforms, img_root=None, img_feature_path=None,
                text_feature_path=None, tokenizer=None, is_wds=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        
        self.meta = json.load(open(input_filename, 'r'))
        self.img_features, self.text_features = None, None
        if img_feature_path:
            if Path(img_feature_path).suffix == '.npy':
                self.img_features = np.memmap(img_feature_path, dtype='float32', mode='r', 
                                        shape=(len(self.meta), 1024))
                logging.warning('np.memmap has some bugs.')
            elif Path(img_feature_path).suffix == '.dpt':
                self.img_features = torch.load(img_feature_path, mmap=True) 
              
        if text_feature_path:
            self.random_text = False
            text_features_list = []
            if type(text_feature_path) is list:
                self.random_text = True
                self.text_features = [torch.load(path, mmap=True) for path in text_feature_path ]
                # self.text_features = [(np.memmap(path, dtype='float32', mode='r', 
                #                         shape=(len(self.meta), 4096))) for path in text_feature_path]
            else:
                self.text_features = torch.load(text_feature_path, mmap=True)
                # self.text_features = np.memmap(text_feature_path, dtype='float32', mode='r', 
                #                         shape=(len(self.meta), 4096))
        self.img_root = img_root
        if is_wds:
            self.wds =  wids.ShardListDataset(img_root)
        else:
            self.wds = None
        self.transforms = transforms
        logging.debug('Done loading data.')
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        images, texts = None, None
        if self.img_features is not None:
            images = self.img_features[str(idx)]
        if self.text_features is not None:
            if self.random_text:
                texts = random.choice([fs[str(idx)] for fs in self.text_features])
            else:
                texts = self.text_features[str(idx)]

        if images is None:
            if self.wds is not None:
                sample = self.wds[idx]
                caption = sample['.json']['caption']
                images = sample['.jpg'].convert("RGB")
                images = self.transforms(images)
            else:
                image_path = os.path.join(self.img_root, self.meta[idx]['image'])
                images = self.transforms(Image.open(image_path))
        if texts is None:
            caption = self.meta[idx]['caption']
            if self.tokenize:
                texts = self.tokenize([caption])[0]      
        return images, texts

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_path, txt_path, img_feature_path=None, transform=None):
        self.zip_path = zip_path
        self.transforms = transform
        self.zip_handle = None
        self.txtlines = []
        with open(txt_path,'r') as f:
            self.txtlines = f.readlines()
        self.img_features = None
        if img_feature_path:
            self.img_features = np.load(img_feature_path)

    def __getitem__(self, idx):
        line = self.txtlines[idx]
        name,label = line.split()
        label = int(label)
        
        if self.img_features is not None:
            image = np.array(self.img_features[idx])
            return image, label

        if self.zip_handle is None:
            self.zip_handle = zipfile.ZipFile(self.zip_path,'r')
        image = Image.open(io.BytesIO(self.zip_handle.read(name)))
        image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
            
        return image,label

    def __len__(self):
        return len(self.txtlines)

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    multi_shards = shards.split(";")
    shards_list = []
    for sd in multi_shards:
        shards_list.extend(wds.shardlists.expand_urls(sd))
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084 (we have: 407259192)
        # LAION-2B (english): 2170337258 (we have: 1.7b)
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2", "val_zip"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    elif split == 'val_zip':
        data_path = args.imagenet_val
        text_path = args.imagenet_val_text
        preprocess_fn = preprocess_val
        img_feature_path = None
        dataset = ZipDataset(data_path, text_path, img_feature_path, transform=preprocess_fn)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path
        
        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample or '0.txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample or '0.png' in sample)
    return has_caption and has_image 


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if len(filesample) ==0:
            continue
    
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
        search=False
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        multi_urls = urls.split(";")
        urls = []
        for sd in multi_urls:
            urls.extend(wds.shardlists.expand_urls(sd))
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    # def search_urls(self, urls):
    #     seach_all = []
    #     for root in urls:
    #         parts = os.listdir(root)
    #         part_paths = [os.path.join(root, part) for part in parts]

    #         all_paths = []
    #         for part_path in part_paths:
    #             temp_paths = glob.glob(os.path.join(part_path, '*.tar'))
    #             all_paths.extend(temp_paths)
    #         seach_all.extend(all_paths)
    #     return seach_all

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))

def search_urls(urls):
    multi_urls = urls.split(";")
    urls = []
    for sd in multi_urls:
        urls.extend(wds.shardlists.expand_urls(sd))
    seach_all = []
    for root in urls:
        parts = os.listdir(root)
        part_paths = [os.path.join(root, part) for part in parts]

        all_paths = []
        for part_path in part_paths:
            temp_paths = glob.glob(os.path.join(part_path, '*.tar'))
            all_paths.extend(temp_paths)
        seach_all.extend(all_paths)
    return seach_all

def get_real_wds_size(shards):
    all_wds_tars = wds.shardlists.expand_urls(shards)
    count = 0
    for tar_path in all_wds_tars:
        with tarfile.open(tar_path, 'r') as tf:
            all_names = tf.getnames()
            count += len(all_names)//2
    return count



def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if is_master(args, local=args.log_local):
        logging.info(f"Num of shards in {input_shards}: {str(num_shards)}")
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        multi_datasets = input_shards.split(";")
        all_shards = []
        for ds in multi_datasets:
            all_shards.extend(wds.shardlists.expand_urls(ds))
        pipeline = [wds.SimpleShardList(all_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        if args.extract_features:
            pipeline.append(wds.split_by_node)
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
        
    def get_text(item):
        sample = {}
        sample['image'] = item['jpg']            
        f1 = np.frombuffer(item['short_feature'], dtype=np.float32)
        f2 = np.frombuffer(item['long_feature'], dtype=np.float32)
        sample['text'] =  random.choice([f1, f2])
        return sample

    pipeline.extend([
        # wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        # wds.rename(image="jpg;png;jpeg;webp",),
        wds.map(get_text),
        wds.map_dict(image=preprocess_img),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])
    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)
    
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=False,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

class MixingDataLoader:
    """Mixing different datasets with round-robin or weighted round-robin
    Adpated from https://github.com/mlfoundations/open_clip/pull/107/files
    """
    def __init__(self, args, preprocess_train, epoch, tokenizer, sample_weights=False):
        train_data_list = args.train_data_list.split(';')
        dataset_type_list = args.dataset_type_list.split(';') if args.dataset_type_list else ['webdataset' for _ in range(len(train_data_list))]
        if not args.train_num_samples_list or len(args.train_num_samples_list) != len(train_data_list):
            train_num_samples_list = [args.train_num_samples//len(train_data_list) for _ in range(len(train_data_list))]
        else:
            train_num_samples_list = args.train_num_samples_list

        data_train = []
        for train_data, dataset_type, train_num_samples in zip(train_data_list, dataset_type_list, train_num_samples_list):
            args.train_data = train_data
            args.train_num_samples = train_num_samples
            data_train.append(
                get_dataset_fn(train_data, dataset_type)(
                    args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer))

        self.args = args
        self.num_datasets = len(data_train)
        self.dataloaders = [dataset.dataloader for dataset in data_train]
        self.dataiters = [iter(dataloader) for dataloader in self.dataloaders]
        self.datasets = train_data_list
        self.num_batches = sum([dataloader.num_batches for dataloader in self.dataloaders])
        self.num_samples = sum([dataloader.num_samples for dataloader in self.dataloaders])

        # calculate sample weights according to num_samples of multiple datasets
        self.sample_weights = np.array([float(dataloader.num_samples) / self.num_samples for dataloader in self.dataloaders]) if sample_weights else None

        if is_master(args, local=args.log_local):
            # print("List of training datasets in MixingDataLoader: ", train_data_list)
            logging.info("Training datasets with virtual epcoh samples in MixingDataLoader:")
            for dataset, num_samples in zip(train_data_list, train_num_samples_list):
                logging.info(f"\t{num_samples} samples per virtual epoch -> {dataset}")
            logging.info(f"Num of datasets in MixingDataLoader: {self.num_datasets}")
            logging.info(f"Num of samples in MixingDataLoader: {self.num_samples}")
            logging.info(f"Num of batches in MixingDataLoader: {self.num_batches}")
            if self.sample_weights is None:
                logging.info("Disable sample_weights...")
            else:
                logging.info(f"Enable sample_weights: {self.sample_weights}")

        self.count = 0
        self.current_epoch = epoch #0
        self.data_train = data_train
        if self.args.distributed and data_train is not None:
            for data_info in data_train:
                data_info.set_epoch(epoch)
    def __len__(self):
        return self.num_samples

    def __iter__(self):
        while True:
            if self.count == self.num_batches:
                self.current_epoch += 1
                self.count = 0
                if self.args.distributed and self.data_train is not None:
                    for data_info in self.data_train:
                        data_info.set_epoch(self.current_epoch)
                return # end each epoch

            # set random seed for sampling from the same dataset.
            # sample a dataset according to sample_weights
            if self.sample_weights is not None:
                stable_random_seed = int(self.count + self.num_batches * self.current_epoch)
                np.random.seed(stable_random_seed)
                iter_index = np.random.choice(range(self.num_datasets), p=self.sample_weights)
            else:
                iter_index = self.count % self.num_datasets
            # generate training image-text pairs from the sampled dataset.
            try:
                data_iter = self.dataiters[iter_index]
                batch = next(data_iter)
            except StopIteration:
                # refresh dataiter if dataloader is used up.
                self.dataiters[iter_index] = iter(self.dataloaders[iter_index])
                data_iter = self.dataiters[iter_index]
                batch = next(data_iter)

            self.count += 1

            yield batch

def get_cxr_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomCSVDataset(
        csv_file=input_filename,
        transform=preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        is_train=is_train)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_zeroshot_dataset(args, preprocess_fn, is_train=False, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomZeroshotDataset(
        csv_file=input_filename,
        transform=preprocess_fn,
        img_key=args.csv_img_key,
        tokenizer=tokenizer
    )
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
		tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_json_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset(
        input_filename,
        preprocess_fn,
        img_root=args.img_root,
        img_feature_path=args.img_feature_path,
        text_feature_path=args.text_feature_path,
		tokenizer=tokenizer,
        is_wds=False)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts

def get_retrieval_dataset(args, preprocess_fn, is_train=False, tokenizer=None,
                          input_filename=None):
    # input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomCSVDataset(
        csv_file=input_filename,
        transform=preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        is_train=is_train)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == 'json':
        return get_json_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "cxr":
        return get_cxr_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
def get_mixing_dataset_fn(args, preprocess_train, epoch, tokenizer):
    sample_weights = True if args.train_num_samples_list else False
    dataloader = MixingDataLoader(args, preprocess_train, epoch, tokenizer, sample_weights)
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    return DataInfo(dataloader, shared_epoch)

def get_concat_dataset_fn(args, preprocess_train, epoch, tokenizer, is_train=True):
    path = args.train_data_file
    datasets_info = yaml.safe_load(open(path,'r'))
    datasets = []
    for info in datasets_info:
        name = info['name']
        logging.info(f'Loading {name} dataset.')
        json_file = info['json_file']
        img_root = info['img_root']
        text_feature_path = info['text_feature_path']
        img_feature_path = info.get('img_feature_path', None)
        ds_type = info.get('type', None)
        is_wds = False
        if ds_type is not None:
            is_wds = ds_type=='wds'
        
        dataset = JsonDataset(
            input_filename=json_file,
            transforms=preprocess_train,
            img_root=img_root,
            img_feature_path=img_feature_path,
            text_feature_path=text_feature_path,
            tokenizer=tokenizer,
            is_wds=is_wds
            )
        datasets.append(dataset)
    
    dataset = ConcatDataset(datasets)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns ##TODO: Modify this
    data = {}

    if args.train_data_list:
        data["train"] = get_mixing_dataset_fn(args, preprocess_train, epoch, tokenizer)
    elif args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
    elif args.train_data_file:
        data["train"] = get_concat_dataset_fn(args, preprocess_train, epoch, tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val_zip")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")
    
    if args.chexpertplus:
        data['chexpertplus'] = get_retrieval_dataset(args, preprocess_val, 
                                                   input_filename=args.chexpertplus, tokenizer=tokenizer)
    if args.zeroshot_data:
        args.val_data = args.zeroshot_data  # Temporarily set val_data to zeroshot_data
        data['zeroshot'] = get_zeroshot_dataset(args, preprocess_val, is_train=False, tokenizer=tokenizer)
        args.val_data = args.val_data  # Restore original val_data if needed
    return data
