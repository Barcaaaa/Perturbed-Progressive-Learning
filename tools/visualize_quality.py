import matplotlib.pyplot as plt
import numpy as np

# 创建画布和子图
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=180)

x = [1, 2, 3, 4]
x_label = ['1/16', '1/8', '1/4', '1/2']
easy = [67.74, 68.84, 70.93, 72.33]
hard = [53.28, 54.71, 55.92, 58.26]
boosted = [56.73, 57.50, 58.21, 60.04]

# 设置每个柱状图的宽度
bar_width = 0.4
dis = [bar_width / 2, bar_width / 2, bar_width / 2, bar_width / 2]

# 绘制折线图
ax[0].bar([a - b for a, b in zip(x, dis)], easy, width=bar_width,
          label='easy group', color='blue', alpha=0.5)
ax[0].bar([a + b for a, b in zip(x, dis)], hard, width=bar_width,
          label='hard group', color='green', alpha=0.5)
gap = [b - a for a, b in zip(hard, boosted)]
ax[0].bar([a + b for a, b in zip(x, dis)], gap, width=bar_width,
          label='boosted hard group', color='red', alpha=0.5, bottom=hard)

# 在子图的每个柱子的顶部添加数值标签（boosted的增幅）
for i, value in enumerate(boosted):
    ax[0].text(x[i] + bar_width/2, value - 1.5, '{:.2f}'.format(float(gap[i])), ha='center')
    # ax.annotate('{:.2f}'.format(float(gap[i])), xy=(x[i] + bar_width/2, value - 1.5), xytext=(x[i] + bar_width/2, value - 1.5),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', va='bottom')

# 设置标题和坐标轴标签
ax[0].set_title('MSDD-3',fontsize=15)
ax[0].set_xlabel('partition', fontsize=15)
ax[0].set_ylabel('mIoU (%)', fontsize=15)

ax[0].set_xticks(x)
ax[0].set_xticklabels(x_label)
ax[0].set_ylim(40, 75)

# 显示网格线
ax[0].grid(True)

# 加了才会显示label信息
ax[0].legend(loc="upper left")

# 设置刻度线的字体大小
ax[0].tick_params(axis='both', which='major', labelsize=12)


x_label = ['1/16', '1/8', '1/4', '1/2']
easy = [54.22, 65.18, 69.26, 71.12]
hard = [45.42, 55.85, 59.51, 60.77]
boosted = [50.66, 60.40, 62.68, 63.44]

# 绘制折线图
ax[1].bar([a - b for a, b in zip(x, dis)], easy, width=bar_width,
          label='easy group', color='blue', alpha=0.5)
ax[1].bar([a + b for a, b in zip(x, dis)], hard, width=bar_width,
          label='hard group', color='green', alpha=0.5)
gap = [b - a for a, b in zip(hard, boosted)]
ax[1].bar([a + b for a, b in zip(x, dis)], gap, width=bar_width,
          label='boosted hard group', color='red', alpha=0.5, bottom=hard)

# 在子图的每个柱子的顶部添加数值标签（boosted的增幅）
for i, value in enumerate(boosted):
    ax[1].text(x[i] + bar_width/2, value - 1.5, '{:.2f}'.format(float(gap[i])), ha='center')

# 设置标题和坐标轴标签
ax[1].set_title('DAGM2007',fontsize=15)
ax[1].set_xlabel('partition', fontsize=15)
ax[1].set_ylabel('mIoU (%)', fontsize=15)

ax[1].set_xticks(x)
ax[1].set_xticklabels(x_label)
ax[1].set_ylim(40, 75)

# 显示网格线
ax[1].grid(True)

# 加了才会显示label信息
ax[1].legend(loc="upper left")

# 设置刻度线的字体大小
ax[1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('quality.png')
plt.clf()
