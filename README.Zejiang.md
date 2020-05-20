
Here I attach some quick notes about how to run the code for our stamp dataset. Please let me know if you have any questions!


## How the dataset folders are organized in `datasets`? 

```
datasets/
└── stamp/
    ├── train/
    └── val/
        ├── bus/
        |   ├── masks_1.png 
        |   └── .... 
        ├── buglarge/
        |   ├── masks.png 
        |   └── .... 
        ├── zebra/
        |   ├── masks.png 
        |   └── .... 
```

Ane the original COCO dataset could be put into other directory, and you can specify it in the line 108 in `process_stamp.py` code. 


## How to run the pre-processing, training, and inference code? 


1. Process the subset:
    ```
    python preprocess_stamp.py -s giraffe
    ```
    It will generate the dataset in `datasets/stamp_<category_name>` (OK this is a bit weird structure.). 
    ```
    datasets/stamp_<category_name>
    ├── train_bbox/
    ├── train_img/
    ├── train_inst/
    ├── train_label/
    └── val_<similar>/
    ```
2. Run the script for training:
    ```
    bash scripts/train_box2mask_stamp.sh
    bash scripts/train_mask2image_stamp.sh
    ```
    They will use the data in `datasets/stamp_<category_name>` and train the models. Please note, for different subsets, you will also need to change the path folder inside the file. (I should have included this as an input argument to the scripts..)


3. Run the scripts for inference: 
    ```
    bash scripts/test_pretrained_mask2image_stamp.sh
    bash scripts/test_pretrained_box2mask_stamp.sh
    ```
    By default, they will use the data in `datasets/stamp_<category_name>_run`, which is a path contains data specifically used for testing. You may also need to generate the data following step1. 

## Running Inference 

1. Organize the inference data according and save it inside the `datasets/inference` folder. 

2. Run the inference preprocessing script: 
    ```
    python preprocess_inference.py
    ```

3. Add `_show` or `_run` to input paths inside the `test` scripts. For example, `--dataroot datasets/stamp_giraffe_run/` in the `scripts/test_pretrained_box2mask_stamp.sh` and `scripts/test_pretrained_mask2image_stamp.sh` file. 

4. Run the scripts for joint_inference_model: 
    ```
    bash scripts/test_joint_inference_stamp.sh
    ```

5. You can find the result folder inside the current directory, e.g., `giraffe_run` or `giraffe_show`. 
