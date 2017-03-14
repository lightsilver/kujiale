# kujiale

# iaa 图像质量评价工程

```
iaa 
  ├── iaa_model.py  用Keras构建model并训练\n
  ├── train   存放训练图片\n
  └── validation  存放测试图片\n
```

# ImageSearch 以图搜图工程

```
ImageSearch
  ├── AutoEncoder.py  构建AutoEncoding方法并训练
  ├── AutoEncoder.pyc
  ├── getImage.py  从数据库中获取模型图片
  ├── img    存放从数据库中获取的图片
  ├── modelTest.py   输入图片路径，显示输入图片和重构出来的图片
  ├── modelTest.pyc
  ├── my_model_architecture.json   保存下来的训练好的autoEncoding模型
  ├── my_model_weights.h5   保存下来的训练好的autoEncoding模型权重
  └── s.py   使用VGG16 fc2层衡量两幅图片的相似度
  ```
