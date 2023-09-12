python preparing/prepare_dataset.py
CUDA_VISIBLE_DEVICES=0,1 python predicting/predict_lightgbm.py
CUDA_VISIBLE_DEVICES=0 python predicting/predict_blend.py
