import numpy as np

# 加载数据
statelist = np.load('statelist_square_2550.npy')
action_all = np.load('action_all_square_2550.npy')


# # 计算statelist在维度7上的均值和标准差
# statelist_reshaped = statelist.reshape(28*1400, 14)
# mean_statelist = np.mean(statelist_reshaped, axis=0)
# std_statelist = np.std(statelist_reshaped, axis=0)

# # # 计算action_all在维度7上的均值和标准差
action_all_reshaped = action_all.reshape(28*800, 7)
mean_action_all = np.mean(action_all_reshaped, axis=0)
std_action_all = np.std(action_all_reshaped, axis=0)

# 打印信息
# print("statelist的类型：", type(statelist))
# print("statelist的形状：", statelist.shape)
# print("action_all的类型：", type(action_all))
# print("action_all的形状：", action_all.shape)
print(mean_action_all)
print(std_action_all)
# print(mean_statelist)
# print(std_statelist)
np.save('action_all_square_transformer_std_2550.npy', std_action_all)
np.save('action_all_square_transformer_mean_2550.npy', mean_action_all)
