export CUDA_VISIBLE_DEVICES=0
python train_deep_controller.py \
--n_class 12 \
--lr 0.1 \
--weight_decay 1e-4 \
--momentum 0.0 \
--epoch 150 \
--data_path "/media/ldy/" \
--meta_path "/media/ldy/" \
--model_path "/home/ldy/AgingProductModeling-DL/results/saved_models/" \
--log_path "/home/ldy/AgingProductModeling-DL/results/logs/" \
--task_name "resnet_controller" \
--batch_size 8 \
