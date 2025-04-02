MODEL=EVA02-CLIP-L-14-336
PRETRAINED=eva_clip
python -m torch.distributed.launch --nproc_per_node=4 \
	--use_env training/main.py \
        --enable-deepspeed \
        --grad-checkpointing \
        --name="BIOVIL-base" \
        --local-loss \
        --save-frequency 10  \
        --zeroshot-frequency 1 \
        --report-to="tensorboard, wandb" \
        --wandb-project-name="TMC_final" \
        --wandb-notes="BIOVIL-T2-switch" \
        --train-data "/data/csv/llm2clip/nips/mimic_train3.csv" \
        --val-data "/data/csv/llm2clip/nips/mimic_test2.csv" \
        --pretrained=${PRETRAINED} \
        --precision "bf16" \
        --warmup 100 \
        --batch-size=48 \
        --eval-batch-size=48 \
        --bce-loss 0 \
        --log-every-n-steps 1000 \
        --epochs=40 \
        --lr=1e-4 \
        --visual-lr=1e-4 \
        --text-lr=5e-6 \
        --projection-lr=5e-4 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --text-ld=1.01 \
        --visual-ld=0.85 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=4 \
        --model=${MODEL} \
        --seed 4096 \
        --gather-with-grad \
        --text-base="microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned" \
        --force-custom-clip \
        --optimizer="ap_adamw" \
        --zero-stage=1 \
        --dataset-type "cxr" \
        --csv-img-key "img_path" \
        --csv-caption-key "caption_if_none" \
        --chexpertplus "/data/csv/llm2clip/nips/test_chexpert2.csv" \
        --model-pth "/model/llm2clip/llm2vec/1b_full/supervised4/checkpoint-4896/pytorch_model.bin" \
        --zeroshot-data "/data/csv/llm2clip/nips/zeroshot/all2.csv"

