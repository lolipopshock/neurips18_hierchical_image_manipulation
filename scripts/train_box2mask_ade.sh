python train_box2mask.py --dataroot=datasets/ade20k/ --dataloader ade20k --name neurips_box2mask_ade  --use_gan --prob_bg 0.05 --label_nc 49 --output_nc 49 --model AE_maskgen_twostream --which_stream obj_context --tf_log --batchSize 8 --first_conv_stride 1 --first_conv_size 5 --conv_size 4 --num_layers 3 --use_resnetblock 1 --num_resnetblocks 1 --nThreads 2 --niter 400 --beta1 0.5 --objReconLoss bce --norm_layer instance --cond_in ctx_obj --gan_weight 0.1 --which_gan patch_multiscale --num_layers_D 3 --n_blocks 6 --fineSize 256 --use_output_gate --no_comb --contextMargin 3 --use_ganFeat_loss --min_box_size 32 --max_box_size 256 --add_dilated_layers --lr_control --gpu_ids 2
