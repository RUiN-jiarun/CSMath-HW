# 课程报告：张量分解在神经辐射场渲染中的应用

## 报告内容
详见pdf内容，`report`文件夹包含了tex工程，基于CVPR模板。

## code
`code`文件夹包含了一个基于numpy后端实现的简单的张量分解方法，主要实现了CP分解。  
### 项目依赖
+ python 3.9+
+ numpy>=1.24
+ scipy>=1.10
+ tensorly==0.8.1
### 运行方法
```sh
cd code
python test_cp.py
```
### 测试简单三阶CP分解性能与误差与tensorly对比
```text
Speed benchmark for 3 order tensor CP decomposition.

----------TensorNP----------
Iterations: [30] | Total: [3.08016s] | Avg.: [0.10267s]
----------Tensorly----------
Iterations: [30] | Total: [5.28228s] | Avg.: [0.17608s]

Correctness benchmark for 3 order tensor CP decomposition.

----------TensorNP----------
error (0.18030571017344435)
----------Tensorly----------
error (0.16475278160403942)
```

