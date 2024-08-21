CUDA_VISIBLE_DEVICES=1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=3 --master_port 1234 pretrain_multicxr.py \
    --name MULTICXR_PRETRAIN \
    --batch_size_per_gpu 24 \
    --pretrained_dir pretrained_weights/pretrain.ckpt \
    --output_dir outputs/multicxr_pretrain \