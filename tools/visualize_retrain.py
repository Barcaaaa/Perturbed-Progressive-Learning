import matplotlib.pyplot as plt
import numpy as np

# 创建画布和子图
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=180)

x = [1, 2, 3]
x_label = ['1/16', '1/8', '1/4']
one_stage = [68.02, 69.10, 69.93]
two_stage = [68.51, 69.43, 69.89]
progressive = [69.88, 71.17, 71.60]

# 绘制折线图
ax[0].plot(x, one_stage, label='one-stage re-training', color='blue', marker='o', linewidth=2, markersize=8)
ax[0].plot(x, two_stage, label='two-stage re-training', color='green', marker='^', linewidth=2, markersize=8)
ax[0].plot(x, progressive, label='progressive re-training', color='red', marker='*', linewidth=2, markersize=10)

# 设置标题和坐标轴标签
ax[0].set_title('MSDD-3',fontsize=15)
ax[0].set_xlabel('partition', fontsize=15)
ax[0].set_ylabel('mIoU (%)', fontsize=15)

ax[0].set_xticks(x)
ax[0].set_xticklabels(x_label)
ax[0].set_yticks(np.arange(67, 73, 1, dtype=int))

# 显示网格线
ax[0].grid(True)

# 加了才会显示label信息
ax[0].legend(loc="lower right")

# 设置刻度线的字体大小
ax[0].tick_params(axis='both', which='major', labelsize=12)

x_label = ['1/8', '1/4', '1/2']
one_stage = [51.85, 58.95, 68.62]
two_stage = [52.24, 59.16, 68.70]
progressive = [53.66, 60.51, 69.63]

# 绘制折线图
ax[1].plot(x, one_stage, label='one-stage re-training', color='blue', marker='o', linewidth=2, markersize=8)
ax[1].plot(x, two_stage, label='two-stage re-training', color='green', marker='^', linewidth=2, markersize=8)
ax[1].plot(x, progressive, label='progressive re-training', color='red', marker='*', linewidth=2, markersize=10)

# 设置标题和坐标轴标签
ax[1].set_title('MTD', fontsize=15)
ax[1].set_xlabel('partition', fontsize=15)
ax[1].set_ylabel('mIoU (%)', fontsize=15)

ax[1].set_xticks(x)
ax[1].set_xticklabels(x_label)
ax[1].set_yticks(np.arange(50, 75, 5, dtype=int))

# 显示网格线
ax[1].grid(True)

# 加了才会显示label信息
ax[1].legend(loc="lower right")

# 设置刻度线的字体大小
ax[1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('retrain.png')
plt.clf()
