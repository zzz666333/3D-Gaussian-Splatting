# 神经网络和深度学习课程期末作业任务二

## 数据准备

请将 `mydata` 文件夹（在报告中的网盘链接中）放入本项目目录下。

---

## 训练

```bash
python train.py -s mydata --model_path ./output/new_model2 --sh_degree 3 --eval
```

---

## 渲染

```bash
python render.py -m ./output/new_model2 --skip_train
```

---

## 说明

本项目代码基于 3D Gaussian 官方代码稍作一点修改，  
原项目地址：[https://github.com/zzz666333/3D-Gaussian-Splatting](https://github.com/zzz666333/3D-Gaussian-Splatting)