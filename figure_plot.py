import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 数据
stages = ["Multi-Seed", "Algorithmic\nRefinement", "Parallel\nMapping", "Tensor\nTiling", "Memory\nOptimization"]
speedup = [1, 1.995576768, 2.373665778, 2.630549004, 2.809582724]

# 设置样式 - 针对双栏论文优化
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'lines.linewidth': 3.5,
    'lines.markersize': 12
})

# ICML 双栏格式 - 3.25英寸宽度
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制主线 - 更粗更明显
line = ax.plot(range(len(stages)), speedup,
               marker='o', markersize=14,
               linewidth=4,
               color='#2E86AB',  # 专业蓝色
               markerfacecolor='#2E86AB',
               markeredgewidth=3,
               markeredgecolor='white',
               label='Cumulative Speedup',
               zorder=3)

# 添加数据标签 - 更大字体
for i, (stage, value) in enumerate(zip(stages, speedup)):
    ax.annotate(f'{value:.2f}×',
                xy=(i, value),
                xytext=(0, 12),
                textcoords='offset points',
                ha='center',
                fontsize=15,
                fontweight='bold',
                color='#2E86AB')

# 添加基线参考线 - 更粗
ax.axhline(y=1, color='gray', linestyle='--', linewidth=2.5, alpha=0.6, label='Initial Kernel (1×)')

# 设置轴标签
ax.set_xlabel('Optimization Stages', fontweight='bold')
ax.set_ylabel('Speedup over Initial Kernel', fontweight='bold')

# 设置 x 轴刻度
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, rotation=0, ha='center')

# 设置 y 轴范围和刻度
ax.set_ylim(0.8, 3.2)
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# 美化网格 - 更明显的网格
ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
ax.set_axisbelow(True)

# 添加背景色
ax.set_facecolor('#F8F9FA')

# 美化边框 - 更粗的边框
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('#CCCCCC')

# 增加坐标轴刻度线粗细
ax.tick_params(width=2, length=6)

# 添加图例
ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)

# 紧凑布局
plt.tight_layout()

# 保存为 PDF
plt.savefig("result.pdf", format="pdf", bbox_inches="tight", dpi=300)
print("Figure saved as result.pdf")
plt.close()