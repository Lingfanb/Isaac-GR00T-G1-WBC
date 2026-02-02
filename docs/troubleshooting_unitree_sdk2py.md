# unitree_sdk2py 安装问题解决记录

**日期**: 2026-01-28  
**环境**: Isaac-GR00T 项目 (uv 管理的 Python 3.10 虚拟环境)

---

## 问题描述

在 Isaac-GR00T 项目的 `groot` 虚拟环境中安装 `unitree_sdk2py` 时失败。

### 错误信息

```
error: subprocess-exited-with-error
× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

---

## 根本原因

### 1. CycloneDDS C++ 库未找到

`cyclonedds==0.10.2` Python 包需要从源码编译，编译过程需要系统安装的 CycloneDDS C++ 库。

### 2. pip Build Isolation 问题

即使设置了 `CYCLONEDDS_HOME` 环境变量，pip 的 **build isolation** 机制会创建一个干净的临时环境来编译包，这个临时环境**无法访问**外部设置的环境变量。

### 3. 环境混淆

在同时激活 `conda base` 和 uv venv 时：
- `pip` 命令指向 conda 的 pip (`/home/lingfanb/miniconda3/bin/pip`)
- `python` 命令指向 venv 的 python (`.venv/bin/python`)
- 导致包被安装到错误的环境

---

## 解决方案

### 步骤 1: 手动复制 CycloneDDS

从已安装好 `cyclonedds` 的 conda 环境（如 `ros_teleop`）复制到 uv venv：

```bash
cd /home/lingfanb/Lingfan_Research_UCL/Isaac-GR00T

# ros_teleop 和 .venv 都是 Python 3.10，可以直接复制
cp -r ~/miniconda3/envs/ros_teleop/lib/python3.10/site-packages/cyclonedds* .venv/lib/python3.10/site-packages/
cp -r ~/miniconda3/envs/ros_teleop/lib/python3.10/site-packages/cyclonedds.libs .venv/lib/python3.10/site-packages/
```

### 步骤 2: 使用 uv 安装 unitree_sdk2py

由于 uv 管理的 venv 没有独立的 pip，需要使用 `uv pip` 命令：

```bash
uv pip install --no-deps -e unitree_sdk2_python/
```

> **注意**: `--no-deps` 跳过依赖检查，避免触发 `cyclonedds` 的重新编译。

### 步骤 3: 验证安装

```bash
python gr00t/eval/real_robot/G1/test_interface.py
```

如果看到 "G1 Interface Test" 界面而不是 `ModuleNotFoundError`，说明安装成功。

---

## 关键要点

| 问题 | 解决方法 |
|------|----------|
| CycloneDDS 编译失败 | 从已有环境复制预编译的包 |
| pip 安装到错误环境 | 使用 `uv pip` 而不是 `pip` |
| uv venv 没有 pip | 使用 `uv pip install` 命令 |

---

## 相关命令参考

```bash
# 检查当前 python 路径
which python

# 检查 pip 安装位置
pip show unitree_sdk2py | grep Location

# 查看 venv 中已安装的包
ls .venv/lib/python3.10/site-packages/ | grep -E "unitree|cyclone"

# 退出 conda base 环境
conda deactivate
```
