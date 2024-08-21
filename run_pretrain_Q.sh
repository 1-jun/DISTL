CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port 1234 pretrain_multilabel.py \
    --name MIMIC_MULTILABEL_TEST \
    --data_path data_preparation/mimic_5labels \
    --batch_size_per_gpu 24 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir pretrained_weights/pretrain.ckpt \
    --output_dir outputs/test \