import pickle

# 指定文件路径
file_path = '/scripts/experiments/evaluation_logs/others/098/plots/F3_ours_NoBRS_20.pickle'

# 打开并读取pickle文件
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 查看加载的数据
print(data)
