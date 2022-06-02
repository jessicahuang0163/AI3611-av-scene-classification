baseline 用到 openl3 提取的 audio 和 visual 特征，此步骤过程较长，已预先提取好放于 `/dssg/home/acct-stu/stu464/ai3611/av_scene_classify/data/feature`，基于此跑 baseline (注意将环境初始化部分改成自己的设置):

```bash
python train.py -e configs/baseline.yaml (or other yamls)
```

注: eval_pred.py 用于计算指标，预测结果 `prediction.csv` 写成这样的形式 (分隔符为 `\t`):
```
aid     scene_pred      airport     bus     ......  tram
airport-lisbon-1175-44106   airport     0.9   0.000   ......  0.001
......
```
调用方法：
```bash
python evaluate.py -e experiments/baseline (or other folders)
```

