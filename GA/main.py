import sys
from GA_function import *

sys.path.append('.')

# J = [[[0, 1, 2, 3, 4], [1, 3]],  # 可选机器编号
#      [[0, 2, 4], [0, 1, 2], [1, 2, 3, 4]]]
#
# P = [[[2, 6, 5, 3, 4], [8, 4]],  # 可选机器的加工时间
#      [[3, 6, 5], [4, 6, 5], [7, 11, 5, 8]]]
#
# machine_num = 5
#
# while True:
#     print(np.random.rand())


data_name = 'MK01'
J, P, num_jobs, machine_num = load_data(f'Brandimarte_Data/Text/{data_name}')
pop = createInd(machine_num, J, P)
popsize = 40
fitness = [decode(machine_num, J, P, i)[1].max() for i in pop]
best_index = fitness.index(min(fitness))
best_ind = pop[best_index].copy()
best_T = decode(machine_num, J, P, best_ind)[0]
Cmax = fitness[best_index]
g, gen = 0, 100
history = [Cmax]
while g < gen:
    g += 1
    # 父代与选择的个体之间交叉
    choice_index = choice(fitness, 3, pool=popsize)
    new_pop = [pop[i] for i in choice_index]
    l = 0
    while l < popsize / 2:
        if np.random.rand() < 0.7:  # 交叉概率
            new_pop[l], new_pop[l + 1] = cross(new_pop[l], new_pop[l + 1])
        if np.random.rand() < 0.005:  # 变异概率
            new_pop[l] = mutation(new_pop[l], P)
        if np.random.rand() < 0.005:  # 变异概率
            new_pop[l + 1] = mutation(new_pop[l + 1], P)
        l += 2
    pop = new_pop
    # 更新种群信息
    fitness = [decode(machine_num, J, P, i)[1].max() for i in pop]

    best_index_temp = best_index = fitness.index(min(fitness))
    Cmax_temp = fitness[best_index]
    if Cmax_temp < Cmax:
        Cmax = Cmax_temp
        best_ind = pop[best_index_temp].copy()
        best_T = decode(machine_num, J, P, best_ind)[0]
    history.append(Cmax)


plt.plot(history)
drawGantt(best_T)
df = pd.DataFrame(best_T)
print('done')
