# B.N.House

Banknote House【纸钞屋】

#### Clone

```bash
git clone https://github.com/Rainweic/B.N.House.git
git submodule init
git submodule update
```


## 配置文件

存放路径: `configs`

## V0.0.1版本说明

### 运行demo 【DQN网络 登月游戏】

#### 训练

```bash
python train.py -c ./configs/dqn_lunarlander.json
```

会在 `log`文件夹下生成 `policy.pth`模型文件

#### 测试

```bash
python test.py -c ./configs/dqn_lunarlander.json
```
