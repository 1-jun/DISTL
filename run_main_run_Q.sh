CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 main_run_Q_multilabel.py \
    --name MIMIC_MULTILABEL_FOLD2 \
    --data_path data_preparation \
    --batch_size_per_gpu 10 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir outputs/mimic_multilabel_fold1/checkpoint.pth \
    --output_dir outputs/mimic_multilabel_fold2 \
    --total_folds 3 