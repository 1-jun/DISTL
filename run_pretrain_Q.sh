CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1234 pretrain_multilabel_fewer_labels.py \
    --name MIMIC_MULTILABEL_5LABELS \
    --data_path data_preparation/mimic_multilabel_all.csv \
    --batch_size_per_gpu 24 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir pretrained_weights/pretrain.ckpt \
    --output_dir outputs/mimic_multilabel_5labels_pretrain \