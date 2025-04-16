import argparse  # 导入argparse模块，用于解析命令行参数
import pickle  # 导入pickle模块，用于序列化和反序列化数据

import numpy as np  # 导入numpy模块，用于科学计算
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

# 创建ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser()
# 添加命令行参数'datasets'，默认值为'ntu/xsub'，可选值为{'kinetics', 'ntu/xsub', 'ntu/xview'}
parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
# 添加命令行参数'alpha'，默认值为1，用于加权求和
parser.add_argument('--alpha', default=1, help='weighted summation')
# 解析命令行参数
arg = parser.parse_args()

# 获取'datasets'参数的值
dataset = arg.datasets
# 打开标签文件并加载数据
label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
# 打开第一个模型的预测结果文件并加载数据
r1 = open('./work_dir/' + dataset +
          '/agcn_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
# 打开第二个模型的预测结果文件并加载数据
r2 = open('./work_dir/' + dataset +
          '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
# 初始化正确预测数量、总数量和Top-5正确预测数量
right_num = total_num = right_num_5 = 0
# 遍历所有标签数据
for i in tqdm(range(len(label[0]))):
    # 获取当前样本的真实标签
    _, l = label[:, i]
    # 获取第一个模型的预测结果
    _, r11 = r1[i]
    # 获取第二个模型的预测结果
    _, r22 = r2[i]
    # 加权求和两个模型的预测结果
    r = r11 + r22 * arg.alpha
    # 获取Top-5预测结果
    rank_5 = r.argsort()[-5:]
    # 判断真实标签是否在Top-5预测结果中
    right_num_5 += int(int(l) in rank_5)
    # 获取预测结果中得分最高的类别
    r = np.argmax(r)
    # 判断预测结果是否正确
    right_num += int(r == int(l))
    # 更新总数量
    total_num += 1
# 计算Top-1准确率
acc = right_num / total_num
# 计算Top-5准确率
acc5 = right_num_5 / total_num
# 打印Top-1和Top-5准确率
print(acc, acc5)
