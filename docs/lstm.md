/usr/bin/python2.7 /home/zhangxulong/project/sing_voice_detection/lstm.py
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.so.8.0 locally
../data/sing_voice_detection/
start...
len Xtrain=================>>>>>>>>>>>>>>>>>> 32000
weight: models/lstm_X_dataset_model.h5
train...
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Train on 32000 samples, validate on 8000 samples
Epoch 1/10000
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:910] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 1070
major: 6 minor: 1 memoryClockRate (GHz) 1.8225
pciBusID 0000:01:00.0
Total memory: 7.92GiB
Free memory: 7.84GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0)
4s - loss: 0.5842 - acc: 0.7135 - precision: 0.7286 - recall: 0.6796 - fmeasure: 0.7017 - val_loss: 0.4640 - val_acc: 0.7712 - val_precision: 0.7786 - val_recall: 0.7610 - val_fmeasure: 0.7696
Epoch 2/10000
3s - loss: 0.4559 - acc: 0.7752 - precision: 0.7811 - recall: 0.7668 - fmeasure: 0.7738 - val_loss: 0.4270 - val_acc: 0.7936 - val_precision: 0.7978 - val_recall: 0.7875 - val_fmeasure: 0.7926
Epoch 3/10000
3s - loss: 0.4210 - acc: 0.7966 - precision: 0.8009 - recall: 0.7911 - fmeasure: 0.7960 - val_loss: 0.4176 - val_acc: 0.7974 - val_precision: 0.8007 - val_recall: 0.7931 - val_fmeasure: 0.7969
Epoch 4/10000
3s - loss: 0.3969 - acc: 0.8106 - precision: 0.8135 - recall: 0.8063 - fmeasure: 0.8099 - val_loss: 0.4043 - val_acc: 0.8065 - val_precision: 0.8089 - val_recall: 0.8030 - val_fmeasure: 0.8059
Epoch 5/10000
3s - loss: 0.3802 - acc: 0.8228 - precision: 0.8251 - recall: 0.8192 - fmeasure: 0.8222 - val_loss: 0.4014 - val_acc: 0.8094 - val_precision: 0.8126 - val_recall: 0.8060 - val_fmeasure: 0.8093
Epoch 6/10000
3s - loss: 0.3648 - acc: 0.8293 - precision: 0.8315 - recall: 0.8261 - fmeasure: 0.8288 - val_loss: 0.3855 - val_acc: 0.8161 - val_precision: 0.8189 - val_recall: 0.8145 - val_fmeasure: 0.8167
Epoch 7/10000
3s - loss: 0.3503 - acc: 0.8365 - precision: 0.8387 - recall: 0.8345 - fmeasure: 0.8366 - val_loss: 0.4014 - val_acc: 0.8119 - val_precision: 0.8150 - val_recall: 0.8100 - val_fmeasure: 0.8125
Epoch 8/10000
3s - loss: 0.3340 - acc: 0.8435 - precision: 0.8455 - recall: 0.8415 - fmeasure: 0.8435 - val_loss: 0.3893 - val_acc: 0.8226 - val_precision: 0.8244 - val_recall: 0.8199 - val_fmeasure: 0.8221
Epoch 9/10000
3s - loss: 0.3290 - acc: 0.8463 - precision: 0.8476 - recall: 0.8447 - fmeasure: 0.8461 - val_loss: 0.3964 - val_acc: 0.8155 - val_precision: 0.8173 - val_recall: 0.8142 - val_fmeasure: 0.8158
Epoch 10/10000
3s - loss: 0.3153 - acc: 0.8539 - precision: 0.8548 - recall: 0.8523 - fmeasure: 0.8536 - val_loss: 0.3838 - val_acc: 0.8229 - val_precision: 0.8251 - val_recall: 0.8205 - val_fmeasure: 0.8228
Epoch 11/10000
3s - loss: 0.3012 - acc: 0.8619 - precision: 0.8630 - recall: 0.8604 - fmeasure: 0.8617 - val_loss: 0.3961 - val_acc: 0.8257 - val_precision: 0.8273 - val_recall: 0.8242 - val_fmeasure: 0.8258
Epoch 12/10000
3s - loss: 0.2949 - acc: 0.8661 - precision: 0.8669 - recall: 0.8651 - fmeasure: 0.8660 - val_loss: 0.4084 - val_acc: 0.8131 - val_precision: 0.8145 - val_recall: 0.8114 - val_fmeasure: 0.8129
Epoch 13/10000
3s - loss: 0.2841 - acc: 0.8699 - precision: 0.8706 - recall: 0.8690 - fmeasure: 0.8698 - val_loss: 0.4070 - val_acc: 0.8181 - val_precision: 0.8194 - val_recall: 0.8170 - val_fmeasure: 0.8182
Epoch 14/10000
3s - loss: 0.2709 - acc: 0.8788 - precision: 0.8794 - recall: 0.8778 - fmeasure: 0.8786 - val_loss: 0.4355 - val_acc: 0.8071 - val_precision: 0.8078 - val_recall: 0.8054 - val_fmeasure: 0.8066
Epoch 15/10000
3s - loss: 0.2696 - acc: 0.8784 - precision: 0.8789 - recall: 0.8775 - fmeasure: 0.8782 - val_loss: 0.4123 - val_acc: 0.8197 - val_precision: 0.8205 - val_recall: 0.8180 - val_fmeasure: 0.8192
Epoch 16/10000
3s - loss: 0.2569 - acc: 0.8843 - precision: 0.8851 - recall: 0.8837 - fmeasure: 0.8844 - val_loss: 0.4208 - val_acc: 0.8201 - val_precision: 0.8206 - val_recall: 0.8192 - val_fmeasure: 0.8199
Saved model to disk
Saved model weights to disk
===============
loss_metrics:  [0.41723864525556564, 0.81409998774528503, 0.81579784631729124, 0.81309999883174899, 0.81443957567214964]
0.814099987745
takes time : 55(s)



# remove satefull and batchsize

/usr/bin/python2.7 /home/zhangxulong/project/sing_voice_detection/lstm.py
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.so.8.0 locally
../data/sing_voice_detection/
start...
len Xtrain=================>>>>>>>>>>>>>>>>>> 32000
weight: models/lstm_X_dataset_model.h5
train...
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Train on 32000 samples, validate on 8000 samples
Epoch 1/10000
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:910] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 1070
major: 6 minor: 1 memoryClockRate (GHz) 1.8225
pciBusID 0000:01:00.0
Total memory: 7.92GiB
Free memory: 7.84GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0)
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 2906 get requests, put_count=2662 evicted_count=1000 eviction_rate=0.375657 and unsatisfied allocation rate=0.462491
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 100 to 110
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 5166 get requests, put_count=5076 evicted_count=1000 eviction_rate=0.197006 and unsatisfied allocation rate=0.215447
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:259] Raising pool_size_limit_ from 256 to 281
11s - loss: 0.5187 - acc: 0.7411 - precision: 0.7518 - recall: 0.7221 - fmeasure: 0.7355 - val_loss: 0.4226 - val_acc: 0.7974 - val_precision: 0.8020 - val_recall: 0.7907 - val_fmeasure: 0.7963
Epoch 2/10000
10s - loss: 0.4187 - acc: 0.7962 - precision: 0.7993 - recall: 0.7913 - fmeasure: 0.7952 - val_loss: 0.4168 - val_acc: 0.8024 - val_precision: 0.8050 - val_recall: 0.7973 - val_fmeasure: 0.8011
Epoch 3/10000
10s - loss: 0.3886 - acc: 0.8141 - precision: 0.8166 - recall: 0.8105 - fmeasure: 0.8135 - val_loss: 0.3969 - val_acc: 0.8135 - val_precision: 0.8167 - val_recall: 0.8113 - val_fmeasure: 0.8139
Epoch 4/10000
10s - loss: 0.3667 - acc: 0.8260 - precision: 0.8279 - recall: 0.8229 - fmeasure: 0.8253 - val_loss: 0.3944 - val_acc: 0.8143 - val_precision: 0.8166 - val_recall: 0.8106 - val_fmeasure: 0.8136
Epoch 5/10000
10s - loss: 0.3503 - acc: 0.8344 - precision: 0.8364 - recall: 0.8321 - fmeasure: 0.8342 - val_loss: 0.3918 - val_acc: 0.8174 - val_precision: 0.8193 - val_recall: 0.8146 - val_fmeasure: 0.8169
Epoch 6/10000
10s - loss: 0.3303 - acc: 0.8440 - precision: 0.8453 - recall: 0.8421 - fmeasure: 0.8437 - val_loss: 0.4035 - val_acc: 0.8094 - val_precision: 0.8113 - val_recall: 0.8074 - val_fmeasure: 0.8093
Epoch 7/10000
10s - loss: 0.3085 - acc: 0.8550 - precision: 0.8560 - recall: 0.8536 - fmeasure: 0.8548 - val_loss: 0.4172 - val_acc: 0.8133 - val_precision: 0.8147 - val_recall: 0.8119 - val_fmeasure: 0.8133
Epoch 8/10000
10s - loss: 0.2917 - acc: 0.8639 - precision: 0.8649 - recall: 0.8626 - fmeasure: 0.8638 - val_loss: 0.4202 - val_acc: 0.8123 - val_precision: 0.8135 - val_recall: 0.8109 - val_fmeasure: 0.8122
Epoch 9/10000
10s - loss: 0.2756 - acc: 0.8719 - precision: 0.8729 - recall: 0.8709 - fmeasure: 0.8719 - val_loss: 0.4415 - val_acc: 0.8056 - val_precision: 0.8061 - val_recall: 0.8045 - val_fmeasure: 0.8053
Saved model to disk
Saved model weights to disk
===============
loss_metrics:  [0.44409195504188537, 0.7984, 0.79925483798980712, 0.79710000000000003, 0.79816025896072385]
0.7984
takes time : 95(s)

Process finished with exit code 0
