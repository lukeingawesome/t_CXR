#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import types
from torch.hub import load_state_dict_from_url
from torchvision.datasets.utils import download_url
from torchvision.models.resnet import ResNet50_Weights

from .model import ImageModel, MultiImageModel
from .types import ImageEncoderType, ImageEncoderWeightTypes


JOINT_FEATURE_SIZE = 768

BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
BIOMED_VLP_BIOVIL_T = "microsoft/BiomedVLP-BioViL-T"
HF_URL = "https://huggingface.co"

CXR_BERT_COMMIT_TAG = "v1.1"
BIOVIL_T_COMMIT_TAG = "v1.0"

BIOVIL_IMAGE_WEIGHTS_NAME = "biovil_image_resnet50_proj_size_128.pt"
BIOVIL_IMAGE_WEIGHTS_URL = f"{HF_URL}/{BIOMED_VLP_CXR_BERT_SPECIALIZED}/resolve/{CXR_BERT_COMMIT_TAG}/{BIOVIL_IMAGE_WEIGHTS_NAME}"  # noqa: E501
BIOVIL_IMAGE_WEIGHTS_MD5 = "02ce6ee460f72efd599295f440dbb453"

BIOVIL_T_IMAGE_WEIGHTS_NAME = "biovil_t_image_model_proj_size_128.pt"
BIOVIL_T_IMAGE_WEIGHTS_URL = (
    f"{HF_URL}/{BIOMED_VLP_BIOVIL_T}/resolve/{BIOVIL_T_COMMIT_TAG}/{BIOVIL_T_IMAGE_WEIGHTS_NAME}"  # noqa: E501
)
BIOVIL_T_IMAGE_WEIGHTS_MD5 = "a83080e2f23aa584a4f2b24c39b1bb64"


def _download_biovil_image_model_weights() -> Path:
    """Download image model weights from Hugging Face.

    More information available at https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized.
    """
    root_dir = tempfile.gettempdir()
    download_url(
        BIOVIL_IMAGE_WEIGHTS_URL,
        root=root_dir,
        filename=BIOVIL_IMAGE_WEIGHTS_NAME,
        md5=BIOVIL_IMAGE_WEIGHTS_MD5,
    )
    return Path(root_dir, BIOVIL_IMAGE_WEIGHTS_NAME)


def _download_biovil_t_image_model_weights() -> Path:
    """Download image model weights from Hugging Face.

    More information available at https://huggingface.co/microsoft/microsoft/BiomedVLP-BioViL-T.
    """
    root_dir = tempfile.gettempdir()
    download_url(
        BIOVIL_T_IMAGE_WEIGHTS_URL, root=root_dir, filename=BIOVIL_T_IMAGE_WEIGHTS_NAME, md5=BIOVIL_T_IMAGE_WEIGHTS_MD5
    )
    return Path(root_dir, BIOVIL_T_IMAGE_WEIGHTS_NAME)


def get_biovil_image_encoder(pretrained: bool = True) -> ImageModel:
    """Download weights from Hugging Face and instantiate the image model."""
    resnet_checkpoint_path = _download_biovil_image_model_weights() if pretrained else None

    image_model = MultiImageModel(
        img_encoder_type=ImageEncoderType.RESNET50,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=resnet_checkpoint_path,
    )
    return image_model


def get_biovil_t_image_encoder(**kwargs: Any) -> ImageModel:
    """Download weights from Hugging Face and instantiate the image model."""

    # biovilt_checkpoint_path = _download_biovil_t_image_model_weights()
    model_type = ImageEncoderType.RESNET50_MULTI_IMAGE
    image_model = MultiImageModel(
        img_encoder_type=model_type,
        joint_feature_size=JOINT_FEATURE_SIZE,
        # pretrained_model_path=biovilt_checkpoint_path,
        **kwargs,
    )
    
    # Add gradient checkpointing method
    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing for the image encoder."""
        # Find the underlying model (usually ResNet)
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'encoder'):
            # For ResNet models
            if enable and hasattr(self.encoder.encoder, 'layer1'):
                self.encoder.encoder.layer1.apply(lambda m: setattr(m, 'gradient_checkpointing', enable))
                self.encoder.encoder.layer2.apply(lambda m: setattr(m, 'gradient_checkpointing', enable))
                self.encoder.encoder.layer3.apply(lambda m: setattr(m, 'gradient_checkpointing', enable))
                self.encoder.encoder.layer4.apply(lambda m: setattr(m, 'gradient_checkpointing', enable))
                print("Gradient checkpointing enabled for ResNet layers")
            else:
                print("Could not enable gradient checkpointing - ResNet layers not found")
        else:
            print("Could not enable gradient checkpointing - encoder structure not recognized")
    
    # Add the method to the model instance
    image_model.set_grad_checkpointing = types.MethodType(set_grad_checkpointing, image_model)
    
    return image_model


def get_imagenet_init_encoder() -> ImageModel:
    """Download ImageNet pre-trained weights and instantiate the image model."""
    url = ResNet50_Weights.IMAGENET1K_V1.url
    state_dict = load_state_dict_from_url(url)
    image_model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=None,
    )
    image_model.encoder.encoder.load_state_dict(state_dict)
    return image_model


def get_image_encoder(weights: str) -> ImageModel:
    """Instantiate image model with random or pre-trained weights.
    :param weights: Select one of `random`, `imagenet`, `biovil`, `biovil_t`
    """

    if weights == ImageEncoderWeightTypes.RANDOM:
        image_model = ImageModel(
            img_encoder_type=ImageEncoderType.RESNET50,
            joint_feature_size=JOINT_FEATURE_SIZE,
            pretrained_model_path=None,
        )
    elif weights == ImageEncoderWeightTypes.IMAGENET:
        image_model = get_imagenet_init_encoder()
    elif weights == ImageEncoderWeightTypes.BIOVIL:
        image_model = get_biovil_image_encoder()
    elif weights == ImageEncoderWeightTypes.BIOVIL_T:
        image_model = get_biovil_t_image_encoder()
    else:
        raise ValueError(f"Weights option not found: {weights}")

    return image_model
