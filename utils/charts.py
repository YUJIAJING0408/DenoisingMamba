import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# plt.rc("font",family='Sans Regular',weight="bold")
plt.rcParams['font.family'] = 'SimHei'

# def model(x, p):
#     return x ** (2 * p + 1) / (1 + x ** (2 * p))
#
# x = np.linspace(0.75, 1.25, 201)
# fig, ax = plt.subplots(figsize=(4,4),dpi=600)
# for p in [10, 15, 20, 30, 50, 100]:
#     ax.plot(x, model(x, p), label=p)
# ax.legend(title='Order')
# ax.set(xlabel='Voltage (mV)')
# ax.set(ylabel='Current ($\mu$A)')
# ax.autoscale(tight=True)
# fig.savefig(r'/home/yujiajing0408/PycharmProjects/MD/charts/fig1.png',dpi=600)


# PSNR
def PSNR():
    df = pd.read_csv(r'/home/yujiajing0408/PycharmProjects/MD/charts/datas/psnr/RMME layer and ks.csv')
    # print(df.shape)
    # print(df.head(0))
    # print(df.values[:200].shape)
    # 只取一部分 【】
    head = df.head(0).columns.values.tolist()[9:]
    datas = df.values[:200]
    data_x = datas[:, 0]
    data_y = datas[:, 9:]
    # plt
    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
    for i in range(len(head)):
        y = data_y[:, i]
        ax.plot(data_x, y, label=head[i], linewidth=0.7, antialiased=True)
    ax.legend(title='division size')
    ax.set(xlabel='step')
    ax.set(ylabel='psnr(db)')
    ax.relim()
    ax.set_ylim(25, 37)
    # ax.set_yticks([20,33,35,37])
    # ax.autoscale(tight=True)
    fig.savefig(r'/home/yujiajing0408/PycharmProjects/MD/charts/psnr.png', dpi=600)
    pass


def Aux():
    colors = ['#2cb247', '#5bdb9e', '#85ffe4', '#a2ffea', '#bbfff0', '#a6e3f8', '#8bc1fd']
    df = pd.read_csv(r'/home/yujiajing0408/桌面/论文/辅助信息.csv')
    head = df.head(0).columns.values.tolist()[1:]
    datas = df.values[:7]
    categories = datas[:, 0]
    rmse = datas[:, 1]
    psnr = datas[:, 2]
    ssim = datas[:, 3]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=300)
    bar0 = axs[0].bar(categories, rmse, edgecolor='black', color=colors)
    axs[0].set_title('RMSE', fontsize=20)
    axs[0].set_xlabel('aux combination', fontsize=10)
    axs[0].set_ylabel('rmse with 1e-3', fontsize=10)
    axs[0].set_ylim(1.5, 2.5)
    for bar in bar0:
        height = bar.get_height()  # 获取柱子的高度
        axs[0].annotate('{}'.format(height),  # 要显示的文本
                        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本的位置
                        xytext=(0, 3),  # 文本相对于指定位置的偏移量
                        textcoords="offset points",  # 文本坐标系的类型
                        ha='center', va='bottom')  # 文本的水平和垂直对齐方式

    bar1 = axs[1].bar(categories, psnr, edgecolor='black', color=colors)
    axs[1].set_title('PSNR', fontsize=20)
    axs[1].set_xlabel('aux combination')
    axs[1].set_ylabel('rmse(db)')
    axs[1].set_ylim(36, 38)
    for bar in bar1:
        height = bar.get_height()  # 获取柱子的高度
        axs[1].annotate('{}'.format(height),  # 要显示的文本
                        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本的位置
                        xytext=(0, 3),  # 文本相对于指定位置的偏移量
                        textcoords="offset points",  # 文本坐标系的类型
                        ha='center', va='bottom')  # 文本的水平和垂直对齐方式

    bar2 = axs[2].bar(categories, ssim, edgecolor='black', color=colors)
    axs[2].set_title('SSIM', fontsize=20)
    axs[2].set_xlabel('aux combination')
    axs[2].set_ylabel('ssim')
    axs[2].set_ylim(0.93, 0.97)
    for bar in bar2:
        height = bar.get_height()  # 获取柱子的高度
        axs[2].annotate('{}'.format(height),  # 要显示的文本
                        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本的位置
                        xytext=(0, 3),  # 文本相对于指定位置的偏移量
                        textcoords="offset points",  # 文本坐标系的类型
                        ha='center', va='bottom')  # 文本的水平和垂直对齐方式

    plt.tight_layout()
    fig.savefig(r'/home/yujiajing0408/PycharmProjects/MD/charts/aux_combination.png', dpi=300)
    return


# PSNR
def Loss():
    df = pd.read_csv(r'/home/yujiajing0408/桌面/论文/切块大小正序逆序的loss对比.csv')
    # print(df.shape)
    # print(df.head(0))
    # print(df.values[:200].shape)
    # 只取一部分 【】
    head = df.head(0).columns.values.tolist()
    datas = df.values[:40]
    data_x = [i for i in range(1,41)]

    # plt
    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
    for i in range(len(head)):
        y = datas[:, i]
        ax.plot(data_x, y, label=head[i], linewidth=0.7, antialiased=True)
    ax.grid(True, linestyle='--', color='grey', alpha=0.3)
    ax.legend(title='order of size',fontsize='12')
    ax.set(xlabel='epochs')
    ax.set(ylabel='loss')
    # ax.relim()
    # ax.set_ylim(25, 37)
    # ax.set_yticks([20,33,35,37])
    # ax.autoscale(tight=True)
    fig.savefig(r'/home/yujiajing0408/PycharmProjects/MD/charts/loss.png', dpi=600)
    pass

def Cost():
    colors = ['#003f5c','#8a508f','#ff6361','#ffa600']
    marks = ['o','*','+','D']
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=450)
    df_mem = pd.read_csv(r'/home/yujiajing0408/桌面/论文/显存开销.csv')
    df_time = pd.read_csv(r'/home/yujiajing0408/桌面/论文/时间开销.csv')
    head = df_mem.head(0).columns.values.tolist()[1:]
    labels = df_mem.values[:,0]
    datas_mem = df_mem.values[:,1:].transpose((0,1))
    datas_time = df_time.values[:,1:].transpose((0,1))
    for i in range(len(head)):
        y_label = []
        x_mem = []
        x_time = []
        if head[i] in ["RAE","DEMC"]:
            y_label = [l for l in labels if l%64 == 0]
            x_mem = [x for x in datas_mem[:,i] if not np.isnan(x)]
            x_time = [x * 1000 for x in datas_time[:,i] if not np.isnan(x)]
        elif head[i] in ["AFGSA","Ours"]:
            y_label = [l for l in labels if l % 120 == 0]
            x_mem = [x for x in datas_mem[:, i] if not np.isnan(x)]
            x_time = [x * 1000 for x in datas_time[:, i] if not np.isnan(x)]
        axs[0].plot(y_label, x_mem, label=head[i], linewidth=1.2,marker=marks[i],markersize=4, antialiased=True,color=colors[i])
        axs[1].plot(y_label, x_time, label=head[i], linewidth=1.2, marker=marks[i],markersize=4,antialiased=True,color=colors[i])
        for j in range(len(x_mem)):
            axs[0].text(y_label[j]+5,x_mem[j]-50,"{}".format(int(x_mem[j])),horizontalalignment='left', verticalalignment='top',fontsize=6,color=colors[i])
            axs[1].text(y_label[j]+5, x_time[j]-0.003 if head[i] != 'RAE' else x_time[j]+0.003,
                        "{}".format(x_time[j]),
                        horizontalalignment='left',
                        verticalalignment='top' if head[i] != 'RAE' else 'bottom',
                        fontsize=6,
                        color=colors[i])

    axs[0].grid(True, linestyle='--', color='grey', alpha=0.3)
    axs[0].legend(title='模型',fontsize='12')
    axs[0].set_title('显存开销', fontsize=20)
    axs[0].set_xlabel('图像大小',fontsize='12')
    axs[0].set_ylabel('显存（兆字节）',fontsize='12')
    axs[0].tick_params(axis='x', labelsize='8', rotation=90)
    axs[0].tick_params(axis='y', labelsize='8')
    axs[0].set_yticks([i for i in range(0,10000,2048)])
    axs[0].set_xticks([64, 120, 192, 256, 320, 360, 384, 448, 512, 576, 600,640])
    axs[1].grid(True, linestyle='--', color='grey', alpha=0.3)
    axs[1].legend(title='模型', fontsize='12')
    axs[1].set_xlabel('图像大小', fontsize='12')
    axs[1].set_ylabel('时间（毫秒）',fontsize='12')
    axs[1].set_title('时间开销', fontsize=20)
    axs[1].tick_params(axis='x',labelsize='8',rotation=90)
    axs[1].tick_params(axis='y',labelsize='8')
    axs[1].set_yticks([i for i in range(0,601,100)])
    axs[1].set_xticks([64,120,192,256,320,360,384,448,512,576,600,640])
    plt.tight_layout()
    fig.savefig(r'/home/yujiajing0408/PycharmProjects/MD/charts/mem-time_chinese.png', dpi=450)

if __name__ == '__main__':
    # Aux()
    # Loss()
    Cost()