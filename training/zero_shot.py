import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from eva_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel

def zero_shot_classifier(model, classnames, templates, args):
    im_features = torch.load(args.imagenet_classname_feautres)
    cast_dtype = get_cast_dtype(args.precision)
    autocast = get_autocast(args.precision)
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for i, classname in tqdm(enumerate(classnames)):
            texts = im_features[i].to(args.device, dtype=cast_dtype)
            # texts = tokenizer(texts).to(args.device)  # tokenize
            if args.distributed:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device, dtype=cast_dtype)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.eval_batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")

def zero_shot_classifier_medical(model, text_categories, args):
    """Create zero-shot classifier for medical conditions."""
    cast_dtype = get_cast_dtype(args.precision)
    autocast = get_autocast(args.precision)
    
    # Use BiomedVLP-BioViL-T model and tokenizer
    url = "microsoft/BiomedVLP-BioViL-T"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(url, trust_remote_code=True).to(args.device)
    
    with torch.no_grad(), autocast():
        zeroshot_weights = {}
        for category, texts in text_categories.items():
            # Tokenize and create embeddings
            tokenized = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(args.device)
            
            # Get text features using BioViL-T
            text_features = text_model(**tokenized).last_hidden_state[:, 0, :]
            if cast_dtype is not None:
                text_features = text_features.to(dtype=cast_dtype)
            
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
            zeroshot_weights[category] = text_features
            
    return zeroshot_weights

def get_medical_zeroshot(model, improving, stable, worsening, args):
    """Create zero-shot classifier for medical conditions."""
    cast_dtype = get_cast_dtype(args.precision)
    autocast = get_autocast(args.precision)
    
    with torch.no_grad(), autocast():
        # Use BiomedVLP-BioViL-T model and tokenizer
        url = "microsoft/BiomedVLP-BioViL-T"
        tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        text_model = AutoModel.from_pretrained(url, trust_remote_code=True).to(args.device)
        
        # Process each item in the batch
        improving_features_list = []
        stable_features_list = []
        worsening_features_list = []
        
        # Process each example in the batch
        batch_size = len(improving) if isinstance(improving, list) else 1
        improves = [improving] if not isinstance(improving, list) else improving
        stables = [stable] if not isinstance(stable, list) else stable
        worsens = [worsening] if not isinstance(worsening, list) else worsening
        
        for imp, stb, wrs in zip(improves, stables, worsens):
            # Tokenize text if needed
            if isinstance(imp, str):
                try:
                    tokenized_improving = tokenizer(
                        imp,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    )
                    # Move everything to the device
                    tokenized_improving = {k: v.to(args.device) for k, v in tokenized_improving.items()}
                except Exception as e:
                    logging.warning(f"Tokenization error: {e}. Using raw text.")
                    tokenized_improving = imp
            else:
                # Ensure tensor is on the correct device
                tokenized_improving = {k: v.to(args.device) for k, v in imp.items()} if isinstance(imp, dict) else imp.to(args.device)
                
            if isinstance(stb, str):
                try:
                    tokenized_stable = tokenizer(
                        stb,
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=256,
                    )
                    # Move everything to the device
                    tokenized_stable = {k: v.to(args.device) for k, v in tokenized_stable.items()}
                except Exception as e:
                    logging.warning(f"Tokenization error: {e}. Using raw text.")
                    tokenized_stable = stb
            else:
                # Ensure tensor is on the correct device
                tokenized_stable = {k: v.to(args.device) for k, v in stb.items()} if isinstance(stb, dict) else stb.to(args.device)
                
            if isinstance(wrs, str):
                try:
                    tokenized_worsening = tokenizer(
                        wrs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    )
                    # Move everything to the device
                    tokenized_worsening = {k: v.to(args.device) for k, v in tokenized_worsening.items()}
                except Exception as e:
                    logging.warning(f"Tokenization error: {e}. Using raw text.")
                    tokenized_worsening = wrs
            else:
                # Ensure tensor is on the correct device
                tokenized_worsening = {k: v.to(args.device) for k, v in wrs.items()} if isinstance(wrs, dict) else wrs.to(args.device)
            
            # Process text through the BioViL-T model (which is now on the correct device)
            text_features_improving = text_model(**tokenized_improving).last_hidden_state[:, 0, :]
            text_features_stable = text_model(**tokenized_stable).last_hidden_state[:, 0, :]
            text_features_worsening = text_model(**tokenized_worsening).last_hidden_state[:, 0, :]
            
            # Convert output to the correct dtype if needed
            if cast_dtype is not None:
                text_features_improving = text_features_improving.to(dtype=cast_dtype)
                text_features_stable = text_features_stable.to(dtype=cast_dtype)
                text_features_worsening = text_features_worsening.to(dtype=cast_dtype)
            
            # Normalize features
            text_features_improving = F.normalize(text_features_improving, dim=-1)
            text_features_stable = F.normalize(text_features_stable, dim=-1)
            text_features_worsening = F.normalize(text_features_worsening, dim=-1)
            
            improving_features_list.append(text_features_improving)
            stable_features_list.append(text_features_stable)
            worsening_features_list.append(text_features_worsening)
        
        # If batch size is 1, return single items instead of lists
        if batch_size == 1:
            return improving_features_list[0], stable_features_list[0], worsening_features_list[0]
        else:
            return improving_features_list, stable_features_list, worsening_features_list

def run_metric_zeroshot(improving, stable, worsening, label, image_feature):
    # Make sure we're working with the right dimensions
    # Extract features if they have an extra dimension
    if improving.dim() > 2:
        improving = improving.squeeze(0)
    if stable.dim() > 2:
        stable = stable.squeeze(0)
    if worsening.dim() > 2:
        worsening = worsening.squeeze(0)
    
    # Ensure image_feature is 2D
    if image_feature.dim() == 1:
        image_feature = image_feature.unsqueeze(0)
    
    # Normalize features again to be safe
    improving = F.normalize(improving, dim=-1)
    stable = F.normalize(stable, dim=-1)
    worsening = F.normalize(worsening, dim=-1)
    
    # For each category, calculate similarity directly
    similarity_improving = image_feature @ improving.T
    similarity_stable = image_feature @ stable.T
    similarity_worsening = image_feature @ worsening.T
    
    # Combine similarities
    logits = torch.cat([
        similarity_improving.mean(dim=1, keepdim=True),
        similarity_stable.mean(dim=1, keepdim=True),
        similarity_worsening.mean(dim=1, keepdim=True)
    ], dim=1) * 10.0  # Scale factor
    
    prediction = logits.argmax(dim=-1).cpu().numpy()  # Move to CPU before numpy conversion
    gt = label.cpu().numpy()
    
    # Convert numpy arrays to plain integers
    prediction_int = int(prediction)
    gt_int = int(gt)
    correct = 1 if prediction_int == gt_int else 0
    
    # Return integers instead of arrays
    return correct, gt_int, prediction_int

def run_zeroshot2(model, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    results = {}  # Initialize as empty dictionary, we'll populate it with findings as we see them
    
    with torch.no_grad():
        for images, prev_images, qualities, improving, stable, worsening, labels, findings in tqdm(dataloader, unit_scale=args.eval_batch_size):
            images = images.to(args.device) 
            prev_images = prev_images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
                prev_images = prev_images.to(dtype=cast_dtype)
            
            with autocast():
                # Get image features
                if args.distributed:
                    image_features = model.module.visual(images, prev_images).projected_global_embedding
                else:
                    image_features = model.visual(images, prev_images).projected_global_embedding
                image_features = F.normalize(image_features, dim=-1)
                improving_features, stable_features, worsening_features = get_medical_zeroshot(model, improving, stable, worsening, args)
                
                for i, image_feature in enumerate(image_features):
                    # Get the finding for this specific example (convert from list/tensor if needed)
                    finding = findings[i]
                    if isinstance(finding, list):
                        finding = finding[0]  # Take the first item if it's a list
                    elif hasattr(finding, 'item'):
                        finding = finding.item()  # Convert tensor to Python scalar
                    elif not isinstance(finding, (str, int)):
                        finding = str(finding)  # Convert to string as fallback
                    
                    # Initialize result entry for this finding if not already present
                    if finding not in results:
                        results[finding] = {'correct': 0, 'total': 0, 'correct_improving': 0, 
                                          'correct_stable': 0, 'correct_worsened': 0, 
                                          'total_improving': 0, 'total_stable': 0, 'total_worsened': 0}
                    
                    # Get the appropriate feature vectors
                    imp_feature = improving_features[i] if isinstance(improving_features, list) else improving_features
                    stb_feature = stable_features[i] if isinstance(stable_features, list) else stable_features
                    wrs_feature = worsening_features[i] if isinstance(worsening_features, list) else worsening_features
                    
                    label = labels[i]
                    label_int = int(label.cpu().numpy())
                    correct, gt, prediction = run_metric_zeroshot(imp_feature, stb_feature, wrs_feature, label, image_feature)
                    
                    results[finding]['correct'] += correct
                    results[finding]['total'] += 1
                    
                    if label_int == 0:
                        results[finding]['correct_improving'] += correct
                        results[finding]['total_improving'] += 1
                    elif label_int == 1:
                        results[finding]['correct_stable'] += correct
                        results[finding]['total_stable'] += 1
                    elif label_int == 2:
                        results[finding]['correct_worsened'] += correct
                        results[finding]['total_worsened'] += 1

    print(f"final results: {results}")
    accuracies = {k: v['correct'] / v['total'] if v['total'] > 0 else 0 for k, v in results.items()}
    improving_accuracy = {k: v['correct_improving'] / v['total_improving'] if v['total_improving'] > 0 else 0 for k, v in results.items()}
    stable_accuracy = {k: v['correct_stable'] / v['total_stable'] if v['total_stable'] > 0 else 0 for k, v in results.items()}
    worsened_accuracy = {k: v['correct_worsened'] / v['total_worsened'] if v['total_worsened'] > 0 else 0 for k, v in results.items()}
    return results, accuracies, improving_accuracy, stable_accuracy, worsened_accuracy

def run_zeroshot(model, classifiers, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    results = {k: {'correct': 0, 'total': 0, 'correct_improving': 0, 'correct_stable': 0, 'correct_worsened': 0, 'total_improving': 0, 'total_stable': 0, 'total_worsened': 0} for k in classifiers.keys()}
    
    with torch.no_grad():
        for images, prev_images, qualities, improving, stable, worsening, labels, finding in tqdm(dataloader, unit_scale=args.eval_batch_size):
            images = images.to(args.device) 
            prev_images = prev_images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
                prev_images = prev_images.to(dtype=cast_dtype)
            
            with autocast():
                # Get image features
                if args.distributed:
                    image_features = model.module.visual(images, prev_images).projected_global_embedding
                else:
                    image_features = model.visual(images, prev_images).projected_global_embedding
                image_features = F.normalize(image_features, dim=-1)
                for i, image_feature in enumerate(image_features):
                    improving = improving[i]
                    stable = stable[i]
                    worsening = worsening[i]
                    label = labels[i]
                    label_int = int(label.cpu().numpy())
                    correct, gt, prediction = run_metric_zeroshot(improving, stable, worsening, label, image_feature, classifiers)
                    results[finding]['correct'] += correct
                    results[finding]['total'] += 1
                    
                    if label_int == 0:
                        results[finding]['correct_improving'] += correct
                        results[finding]['total_improving'] += 1
                    elif label_int == 1:
                        results[finding]['correct_stable'] += correct
                        results[finding]['total_stable'] += 1
                    elif label_int == 2:
                        results[finding]['correct_worsened'] += correct
                        results[finding]['total_worsened'] += 1

    print(f"final results: {results}")
    accuracies = {k: v['correct'] / v['total'] for k, v in results.items()}
    improving_accuracy = {k: v['correct_improving'] / v['total_improving'] for k, v in results.items()}
    stable_accuracy = {k: v['correct_stable'] / v['total_stable'] for k, v in results.items()}
    worsened_accuracy = {k: v['correct_worsened'] / v['total_worsened'] for k, v in results.items()}
    return results, accuracies, improving_accuracy, stable_accuracy, worsened_accuracy

def run_medical(model, classifiers, dataloader, args):
    """Run zero-shot evaluation for medical conditions."""
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    results = {k: {'correct': 0, 'total': 0, 'score': [], 'gt': []} for k in classifiers.keys()}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, unit_scale=args.eval_batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            
            with autocast():
                # Get image features
                if args.distributed:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute similarities for each condition
                for condition, classifier in classifiers.items():
                    logits = 100. * image_features @ classifier.T
                    predictions = logits.argmax(dim=-1).cpu().numpy()  # Move to CPU before numpy conversion
                    gt = np.array([int(float(x)) for x in labels])
                    correct = (predictions == gt).astype(int)
                    
                    results[condition]['correct'] += correct.sum()
                    results[condition]['total'] += images.size(0)
                    results[condition]['score'].extend(logits[:, 0].cpu().numpy())  # Use extend instead of +=
                    results[condition]['gt'].extend(gt)

    # Calculate metrics
    for condition in results:
        results[condition]['auc'] = roc_auc_score(results[condition]['gt'], results[condition]['score'])
    
    # Calculate accuracies
    aucs = {k: v['auc'] for k, v in results.items()}
    accuracies = {k: v['correct'] / v['total'] for k, v in results.items()}
    return aucs, accuracies

def zero_shot_eval(model, data, epoch, args):

    logging.info('Starting zero-shot rsna or siim.')

    results = {}
    if 'zeroshot' in data:
        # text_categories = {
        #     'pleural_effusion': ["pleural effusion is improving", "pleural effusion is stable", "pleural effusion is worsening"],
        #     'pneumonia': ["pneumonia is improving", "pneumonia is stable", "pneumonia is worsening"],
        #     'pneumothorax': ["pneumothorax is improving", "pneumothorax is stable", "pneumothorax is worsening"],
        #     'edema': ["edema is improving", "edema is stable", "edema is worsening"],
        #     'consolidation': ["consolidation is improving", "consolidation is stable", "consolidation is worsening"],
        # }
        # logging.info('Building medical zero-shot classifier')
        # medical_classifier = zero_shot_classifier_medical(model, text_categories, l2v, args)
        logging.info('Building medical zero-shot classifier')
        # medical_classifier = zero_shot_classifier_medical_zeroshot(model, data['zeroshot'].improving, data['zeroshot'].stable, data['zeroshot'].worsening, l2v, args)
        
        
        logging.info('Evaluating medical conditions')
        # medical_results, medical_accuracies, medical_improving_accuracy, medical_stable_accuracy, medical_worsened_accuracy = run_zeroshot(model, medical_classifier, data['zeroshot'].dataloader, args)
        medical_results, medical_accuracies, medical_improving_accuracy, medical_stable_accuracy, medical_worsened_accuracy = run_zeroshot2(model, data['zeroshot'].dataloader, args)
    results['accuracies'] = medical_accuracies
    results['improving_accuracy'] = medical_improving_accuracy
    results['stable_accuracy'] = medical_stable_accuracy
    results['worsened_accuracy'] = medical_worsened_accuracy
    logging.info('Finished zero-shot imagenet.')
    return results
