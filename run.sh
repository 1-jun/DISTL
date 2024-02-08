CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 pretrain_multilabel.py \
    --name MIMIC_5LABELS \
    --data_path data_preparation/mimic_5labels \
    --lesions 'Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Pleural Effusion' \
    --batch_size_per_gpu 24 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir pretrained_weights/pretrain.ckpt \
    --output_dir outputs/mimic_5labels/pretrain


CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main_run_Q_multilabel.py \
    --name MIMIC_5LABELS_FOLD0 \
    --data_path data_preparation/mimic_5labels \
    --lesions 'Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Pleural Effusion' \
    --batch_size_per_gpu 10 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir outputs/mimic_5labels/pretrain/checkpoint.pth \
    --output_dir outputs/mimic_5labels/fold0 \
    --total_folds 1

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main_run_Q_multilabel.py \
    --name MIMIC_5LABELS_FOLD1 \
    --data_path data_preparation/mimic_5labels \
    --lesions 'Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Pleural Effusion' \
    --batch_size_per_gpu 10 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir outputs/mimic_5labels/fold0/checkpoint.pth \
    --output_dir outputs/mimic_5labels/fold1 \
    --total_folds 2

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main_run_Q_multilabel.py \
    --name MIMIC_5LABELS_FOLD2 \
    --data_path data_preparation/mimic_5labels \
    --lesions 'Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Pleural Effusion' \
    --batch_size_per_gpu 10 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir outputs/mimic_5labels/fold1/checkpoint.pth \
    --output_dir outputs/mimic_5labels/fold2 \
    --total_folds 3