Setup

1. Clone and Prepare 
    1.1 git clone https://github.com/lone-surabhi/AdaFace-Pipeline
    1.2 cd AdaFace-Pipeline
    1.3 python -m venv venv
    1.4 source venv/bin/activate

2. Install Dependencies 
    pip install -r requirements.txt

3. Download agedb30 dataset

4. Download pretrained AdaFace checkpoint

5. Run the pipeline
    4.1 python -m src.pipeline samples/img1.jpeg samples/img2 jpeg \                                                   
    --arch ir_101 \
    --ckpt pretrained/adaface_ir101_ms1mv2.ckpt

    Output : [load] backbone=ir_101  ckpt=pretrained/adaface_ir101_ms1mv2.ckpt  device=cpu
    [warn] missing keys (first 10): ['body.3.shortcut.0.weight', 'body.3.shortcut.1.weight', 'body.3.shortcut.1.bias', 'body.3.shortcut.1.running_mean', 'body.3.shortcut.1.running_var', 'body.16.shortcut.0.weight', 'body.16.shortcut.1.weight', 'body.16.shortcut.1.bias', 'body.16.shortcut.1.running_mean', 'body.16.shortcut.1.running_var']

    [warn] unexpected keys (first 10): ['head.kernel', 'head.t', 'body.3.shortcut_layer.0.weight', 'body.3.shortcut_layer.1.weight', 'body.3.shortcut_layer.1.bias', 'body.3.shortcut_layer.1.running_mean', 'body.3.shortcut_layer.1.running_var', 'body.3.shortcut_layer.1.num_batches_tracked', 'body.16.shortcut_layer.0.weight', 'body.16.shortcut_layer.1.weight']

    img1.jpeg vs img2.jpeg  cosine: 0.9936

6. Evaluate agedb30 
    5.1 python -m src.eval_agedb \
    --data_root "/Users/surabhilone/Desktop/AdaFace-Pipeline/data/agedb30" \
    --pairs "/Users/surabhilone/Desktop/AdaFace-Pipeline/data/agedb30/agedb_30_112x112/agedb_30_ann.txt"

    Output : 6000/6000 [16:46<00:00,  5.96it/s]
    [stats] total pairs: 6000, used: 6000, missing_files: 0
    AgeDB-30 10-fold ACC: 98.25% ± 0.44% | AUC: 0.9918