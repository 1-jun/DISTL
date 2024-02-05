CUDA_VISIBLE_DEVICES=1,3 \
python -m torch.distributed.launch --nproc_per_node=2 main_run_Q_varyspvzlabels.py \
    --name MIMIC_VARYSPVZ_FOLD1 \
    --data_path data_preparation \
    --batch_size_per_gpu 10 \
    --mimic_path /home/wonjun/data/mimic-cxr-jpg-resized512 \
    --pretrained_dir outputs/mimic-vary-spvz-labels/fold1/checkpoint.pth \
    --output_dir outputs/mimic-vary-spvz-labels/fold2 \
    --fold 2