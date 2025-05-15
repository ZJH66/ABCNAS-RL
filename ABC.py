import numpy as np
from evaluation import MetricsDAG
from rl import RL
X = np.loadtxt("./datasets/traindataset.txt")
true_dag = np.loadtxt('./datasets/dbzstand.txt')

# 定义参数搜索空间
parameter_space = {
    'lr1_start': (0.001, 0.03, 0.005),#学习率
    'nb_epoch': (2000, 10000, 500), #遍历次数
    'n_layer' : (1,10,1)#神经网络层数
}

# 定义适应度函数
def fitness_function(parameters):
    # 打印新参数

    # 计算新参数的适应度
    # 使用参数训练强化学习模型
    print("Training with parameters:", parameters)  # 在训练前打印本次训练所选取的参数值
    rl_model = RL(lr1_start=parameters['lr1_start'],
                  nb_epoch=parameters['nb_epoch'],
                  n_layer=parameters['n_layer'])

    # 训练模型并计算召回率
    rl_model.learn(X)
    causal_matrix = rl_model.causal_matrix  # 假设 causal_matrix 是模型的因果矩阵

    print(causal_matrix)
    met = MetricsDAG(causal_matrix, true_dag)
    print(met.metrics)
    # 在这里计算损失函数，比如均方误差（MSE）、交叉熵等
    loss = np.mean((causal_matrix - true_dag) ** 2)  # 举例：均方误差
    print(loss)
    return loss

# 初始化蜜蜂群体
beenum = 20
population = []
Loss = []
for _ in range(beenum):
    parameters = {
        'nb_epoch': np.random.randint(parameter_space['nb_epoch'][0], parameter_space['nb_epoch'][1]+1) // 500 * 500,
        'lr1_start': np.random.uniform(parameter_space['lr1_start'][0], parameter_space['lr1_start'][1])//0.005 * 0.005,
        'n_layer': np.random.uniform(parameter_space['n_layer'][0], parameter_space['n_layer'][1])//1 * 1
        # 添加其他参数的随机初始化
    }
    population.append(parameters)
    Loss.append(1)
# current_loss = fitness_function(parameters[0])
best_loss = 1
# 执行蜂群优化算法
max_iter = 1
for iteration in range(max_iter):
    # 雇佣蜜蜂阶段
    for i in range(beenum):
        print("雇佣蜂" + str(i+1) + "开始工作")
        parameters = population[i]
        current_loss = fitness_function(parameters)
        if(current_loss<best_loss):
            best_loss = current_loss
            best_parameters = population[i]
            Loss[i] = current_loss

        # new_parameters = {
        #     'nb_epoch': parameters.get('nb_epoch', np.random.randint(parameter_space['nb_epoch'][0], parameter_space['nb_epoch'][1]+1) // 500 * 500),
        #     'lr1_start': parameters.get('lr1_start', np.random.uniform(parameter_space['lr1_start'][0], parameter_space['lr1_start'][1])),
        # }
        # current_loss = fitness_function(new_parameters)
        # fitness_new = fitness_function(new_parameters)
        # print("**************************************************")
        # 如果新参数更优，则替换原参数
        # if current_loss < best_loss:
        #     population[i] = new_parameters
    print("*****************************************************")
    # 观察蜜蜂阶段
    # 从雇佣蜜蜂中选择一半的蜜蜂作为观察蜜蜂，尝试在参数空间中随机搜索更好的解

    for i in range(beenum // 2):
        # parameters = population[i]
        print("观察蜂"+str(i+1)+"开始工作")
        new_parameters = {
            'nb_epoch': parameters.get('nb_epoch', np.random.randint(parameter_space['nb_epoch'][0], parameter_space['nb_epoch'][1] + 1) // 500 * 500),
            'lr1_start': parameters.get('lr1_start', np.random.uniform(parameter_space['lr1_start'][0], parameter_space['lr1_start'][1])//0.005 * 0.005),
            'n_layer': np.random.uniform(parameter_space['n_layer'][0], parameter_space['n_layer'][1])//1 * 1
        }  # 观察蜜蜂产生的新参数
        # for param_name, param_range in parameter_space.items():
        #     min_val, max_val = param_range
        #     new_value = np.random.uniform(min_val, max_val)
        #     new_parameters[param_name] = new_value
        # 计算新参数的适应度
        current_loss = fitness_function(new_parameters)
        # 如果新参数更优，则替换原参数
        if current_loss < Loss[i]:
            population[i] = new_parameters
            Loss[i] = current_loss
    # 侦查蜜蜂阶段
    # 从所有蜜蜂中选择一个最差的蜜蜂，并以随机方式生成新的参数
    print("*****************************************************")
    print("侦察蜂开始工作")
    worst_index = Loss.index(max(Loss))
    # worst_index = np.argmin([fitness_function(parameters) for parameters in population])
    parameters = population[worst_index]
    new_parameters = {
        'nb_epoch': parameters.get('nb_epoch', np.random.randint(parameter_space['nb_epoch'][0], parameter_space['nb_epoch'][1] + 1) // 500 * 500),
        'lr1_start': parameters.get('lr1_start', np.random.uniform(parameter_space['lr1_start'][0], parameter_space['lr1_start'][1])//0.005 * 0.005),
        'n_layer': np.random.uniform(parameter_space['n_layer'][0], parameter_space['n_layer'][1])//1 * 1
    }   # 侦查蜜蜂产生的新参数
    # for param_name, param_range in parameter_space.items():
    #     min_val, max_val = param_range
    #     new_value = np.random.uniform(min_val, max_val)
    #     new_parameters[param_name] = new_value
    # 计算新参数的适应度
    current_loss = fitness_function(new_parameters)
    # 如果新参数更优，则替换最差的蜜蜂
    if current_loss < Loss[worst_index]:
        population[worst_index] = new_parameters
        Loss[worst_index] = current_loss
    # 输出当前迭代的最佳参数和适应度
    best_index = Loss.index(min(Loss))
    best_parameters = population[best_index]
    best_fitness = Loss[best_index]
    print(f"Iteration {iteration}: Best parameters: {best_parameters}, Best fitness: {best_fitness}")

# # 评估最佳参数组合
# print("Best parameters:", best_parameters)
# print("Best fitness:", best_fitness)
