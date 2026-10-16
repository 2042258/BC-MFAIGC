import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
num_malicious = [f"{int(v*100)}%" for v in values]
loss_values = [0.0002, 0.8842, 1.0073, 1.1504, 0.9166, 1.0448, 1.0847]  # TRA
esm_values  = [0.06, 1.34, 1.17, 1.42, 1.25, 1.32, 1.41]                # FL
avg_values  = [0.0002]*7                                                # BC-MFAIGC

plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']=['Times New Roman','DejaVu Serif']
plt.rcParams['font.size']=22
plt.figure(figsize=(14,8), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid')
ax=plt.gca()

def annot_line(xs, ys, dy=0.05, fmt='{:.3f}'):
    for x,y in zip(xs,ys):
        ax.annotate(fmt.format(y), (x,y), textcoords='offset points',
                    xytext=(0,10), ha='center', fontsize=12, color='#444444')

plt.plot(num_malicious, loss_values, '--', color='#2C6E91', marker='o', markersize=8, linewidth=2, label='TRA')
plt.plot(num_malicious, esm_values,  '--', color='#D1495B', marker='s', markersize=8, linewidth=2, label='FL')
plt.plot(num_malicious, avg_values,  '--', color='#2E8B57', marker='D', markersize=8, linewidth=2, label='BC-MFAIGC')

annot_line(num_malicious, loss_values, 0.05)
annot_line(num_malicious, esm_values, 0.05)
annot_line(num_malicious, avg_values, 0.05)

plt.xticks(num_malicious, fontsize=20)
plt.ylim(-0.2, 2.1)
plt.yticks(fontsize=20)
plt.xlabel('The proportion of malicious nodes', fontsize=26)
plt.ylabel('MSE', fontsize=26)
legend=plt.legend(loc='upper left', frameon=True)
plt.setp(legend.get_texts(), fontsize=20)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.grid(True, linestyle='--', alpha=0.6)
# TRA / FL note below the caption area (inside figure, upper-middle)
ax.text(0.5, 0.955, 'TRA = Traditional Algorithm;  FL = Federated Learning',
        transform=ax.transAxes, ha='center', fontsize=15, style='italic', color='#333333')
plt.tight_layout()
plt.savefig('./_plotfix/figure_10_modified.pdf', bbox_inches='tight')
print('saved figure_10_modified.pdf')
