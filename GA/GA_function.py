import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd


# O11可用 0 1 2 3 4
# O12可用 1 3
# O21可用 0 2 4
# O22可用 0 1 2
# O23可用 1 2 3 4
# J = [[[0, 1, 2, 3, 4], [1, 3]],  # 可选机器编号
#      [[0, 2, 4], [0, 1, 2], [1, 2, 3, 4]]]
#
# P = [[[2, 6, 5, 3, 4], [8, 4]],  # 可选机器的加工时间
#      [[3, 6, 5], [4, 6, 5], [7, 11, 5, 8]]]
#
# machine_num = 5


def GS(n, J, P):  # global select
    """
     n:机器数量
     J:机器矩阵
     P:加工时间矩阵
     :return:
     """
    MS = []
    time_list = np.zeros(n)  # 机器时间数组
    job_num = len(J)  # 工件数量
    for i in range(job_num):  # i 工件编号
        job_list = J[i]  # 工序列表
        for j in range(len(job_list)):  # j为工序编号
            min_time = 999999
            min_time_index = 0
            min_time_machine = 0
            machine_list = job_list[j]  # i 工件的 j 工序可选的机器编号
            process_time = P[i][j]  # i 工件的 j 工序可选加工机器的加工时间
            if len(machine_list) == 1:
                MS.append(min_time_index)
                min_time_machine = machine_list[0]
                time_list[min_time_machine] += process_time[min_time_index]
            else:
                for m in range(len(machine_list)):
                    temp_time = time_list[machine_list[m]] + process_time[m]
                    if temp_time < min_time:
                        min_time = temp_time
                        min_time_idx = m
                        min_time_machine = machine_list[m]
                MS.append(min_time_idx)
                time_list[min_time_machine] += process_time[min_time_idx]

    return MS


def LS(n, J, P):  # local select
    """
    切换工件之后将机器时间数组归零
     n:机器数量
     J:机器矩阵
     P:加工时间矩阵
     :return:
     """
    MS = []
    job_num = len(J)  # 工件数量
    for i in range(job_num):  # i 工件编号
        time_list = np.zeros(n)  # 机器时间数组
        job_list = J[i]  # 工序列表
        for j in range(len(job_list)):  # j为工序编号
            min_time = 999999
            min_time_index = 0
            min_time_machine = 0
            machine_list = job_list[j]  # i 工件的 j 工序可选的机器编号
            process_time = P[i][j]  # i 工件的 j 工序可选加工机器的加工时间
            if len(machine_list) == 1:  # 若只能选择一个机器
                MS.append(min_time_index)
                min_time_machine = machine_list[0]
                time_list[min_time_machine] += process_time[min_time_index]
            else:
                for m in range(len(machine_list)):
                    temp_time = time_list[machine_list[m]] + process_time[m]
                    if temp_time < min_time:
                        min_time = temp_time
                        min_time_idx = m
                        min_time_machine = machine_list[m]
                MS.append(min_time_idx)
                time_list[min_time_machine] += process_time[min_time_idx]

    return MS


def RS(n, J, P):
    MS = []
    job_num = len(J)
    for i in range(job_num):
        job_list = J[i]
        for j in range(len(job_list)):
            machine_list = job_list[j]
            MS.append(np.random.randint(len(machine_list)))
    return MS


def createInd(n, J, P, split_list=None):
    """
     split_list 三种策略的数量矩阵
     GS LS RS
     """
    if split_list is None:
        split_list = [10, 10, 20]
    pop = []
    gs = GS(n, J, P)
    ls = LS(n, J, P)
    OS = []
    for i in range(len(J)):
        for _ in range(len(J[i])):
            OS.append(i)
    for _ in range(split_list[0]):
        pop.append(gs + np.random.permutation(OS).tolist())
    for _ in range(split_list[1]):
        pop.append(ls + np.random.permutation(OS).tolist())
    for _ in range(split_list[2]):
        pop.append(RS(n, J, P) + np.random.permutation(OS).tolist())
    return pop


def decode(n, J, P, s):
    """
    n:机器数量
    J:机器矩阵
    P:加工时间矩阵
    s:序列

    T 甘特图矩阵
    C 完工时间矩阵
    """
    job_num = len(J)  # 工件数量
    process_num = 0  # 总工序数
    max_process_num = 0  # 最大工序数
    machine_list = []
    for i in range(job_num):
        machine_list_by_process = s[:len(J[i])] if max_process_num == 0 else s[process_num:process_num + len(J[i])]
        max_process_num = max(max_process_num, len(J[i]))
        process_num += len(J[i])
        machine_list.append(machine_list_by_process)
    s = s[process_num:]  # 更新s为后半部分
    T = [[[0]] for _ in range(n)]
    C = np.zeros((job_num, max_process_num))
    k = np.zeros(job_num, dtype=int)  # 确定第几个工序的
    for job in s:
        machine_index = machine_list[job][k[job]]  # 机器编号
        machine = J[job][k[job]][machine_index]  # 机器号
        process_time = P[job][k[job]][machine_index]
        last_job_finish = C[job, k[job] - 1] if k[job] > 0 else 0  # 上一个工序的完工时间
        # 寻找机器上的第一个合适的空闲段
        start_time = max(last_job_finish, T[machine][-1][-1])
        insert_index = len(T[machine])  # 默认插入位置 在末尾
        for i in range(1, len(T[machine])):
            gap_start = max(T[machine][i - 1][-1], last_job_finish)
            gap_end = T[machine][i][0]
            if gap_end - gap_start >= process_time:
                insert_index = i  # 更新插入位置
                break  # 不需要找后面的gap了
        end_time = start_time + process_time
        C[job, k[job]] = end_time
        T[machine].insert(insert_index, [start_time, job, k[job], end_time])
        k[job] += 1
    return T, C


def drawGantt(timelist):
    T = timelist.copy()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色映射字典
    color_map = {}
    for machine_schedule in T:
        for task_data in machine_schedule[1:]:
            job_idx = task_data[1]
            if job_idx not in color_map:
                color_map[job_idx] = (random.random(), random.random(), random.random())

    # 遍历机器绘制
    for machine_idx, machine_schedule in enumerate(T):
        for task_data in machine_schedule[1:]:
            start_time, job_idx, op_idx, end_time = task_data
            color = color_map[job_idx]

            ax.barh(machine_idx, end_time - start_time, left=start_time, height=0.4, color=color, edgecolor='black')

            label = f'{job_idx + 1}-{op_idx + 1}'
            ax.text((start_time + end_time) / 2, machine_idx, label, ha='center', va='center', color='white',
                    fontsize=10)

    plt.xlabel('时间')

    ax.set_yticks(range(len(T)))
    ax.set_yticklabels([f'机器{i + 1}' for i in range(len(T))])

    legend_handles = []
    color_map = dict(sorted(color_map.items(), key=lambda item: item[0]))
    for job_idx, color in color_map.items():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'job{job_idx + 1}'))
    plt.legend(handles=legend_handles, title='工件', bbox_to_anchor=(1.0, 1.0))
    plt.title("FJSP_Solution_GA")
    plt.savefig('MK01.png')
    plt.show()


# 锦标赛法
def choice(fitness, k=3, pool=40):
    """
    :param fitness: 适应度
    :param k: 比较个数
    :param pool: 交叉池
    :return:
    """
    n = len(fitness)
    choice_index = []
    for _ in range(pool):
        random_indices = random.sample(list(range(n)), k)
        f_values = [fitness[i] for i in random_indices]
        min_f_values = min(f_values)
        choice_index.append(f_values.index(min_f_values))
    return choice_index


def cross_MS(A, B):
    """
    机器交叉
    :param A:
    :param B:
    :return:
    """
    job_num = len(A)
    random_numbers = random.sample(list(range(job_num)), 2)
    rl, rr = min(random_numbers), max(random_numbers)
    return A[:rl] + B[rl:rr + 1] + A[rr + 1:], B[:rl] + A[rl:rr + 1] + B[rr + 1:]


def cross_OS(A, B):
    """
    工序交叉
    :param A:
    :param B:
    :return:
    """
    job_id = list(set(A))
    job_num = len(A)
    # 确保抽取的数量不为0，同时不能等于列表长度
    while True:
        num_to_extract = random.randint(1, job_num - 1)
        if num_to_extract > 0 and num_to_extract < len(job_id):
            break
    # 随机抽取S1集合
    S1 = random.sample(job_id, num_to_extract)
    # A
    afinal = [None for _ in range(job_num)]
    temp_B = [item for item in B if item not in S1]
    k = 0
    for i in range(job_num):
        if A[i] in S1:
            afinal[i] = A[i]
        else:
            afinal[i] = temp_B[k]
            k += 1
    # B
    bfinal = [None for _ in range(job_num)]
    temp_A = [item for item in A if item not in S1]
    k = 0
    for i in range(job_num):
        if B[i] in S1:
            bfinal[i] = B[i]
        else:
            bfinal[i] = temp_A[k]
            k += 1
    return afinal, bfinal


def cross(A, B):
    """
    总交叉
    :param A:
    :param B:
    :return:
    """
    job_num_all = int(len(A) / 2)
    MS_A, OS_A = A[:job_num_all], A[job_num_all:]
    MS_B, OS_B = B[:job_num_all], B[job_num_all:]
    MS_A, MS_B = cross_MS(MS_A, MS_B)
    OS_A, OS_B = cross_OS(OS_A, OS_B)
    return MS_A + OS_A, MS_B + OS_B


def mutation_OS(Ind):
    """
    变异
    :param Ind:
    :return:
    """
    A = Ind.copy()
    n = len(A)
    idx1, idx2 = random.sample(range(n), 2)
    rl, rr = min(idx1, idx2), max(idx1, idx2)
    A[rl:rr] = A[rl:rr][::-1]
    return A


def mutation_MS(A, P):
    process_num = len(A)
    process_time = [item for sublist in P for item in sublist]
    r = np.random.randint(process_num)
    random_select = random.sample(range(process_num), r)
    for i in random_select:
        # A[i] = process_time[i].index(min(process_time[i]))
        A[i] = np.argmin(process_time[i])
    return A


def mutation(s, P):
    process_num = int(len(s) / 2)
    MS = s[:process_num]
    OS = s[process_num:]
    return mutation_MS(MS, P) + mutation_OS(OS)


def load_data(path):
    with open(path + ".fjs", 'r') as file:
        lines = file.readlines()

    first_line = lines[0].strip().split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    # 初始化加工机器矩阵 加工时间矩阵
    J, P = [], []
    for line in lines[1:]:
        data = line.strip().split()
        if not data:
            continue
        num_op = int(data[0])  # 该工序具有的操作数
        data = data[1:]
        job = []
        process_time = []
        l = 0
        while l < num_op:
            process_num = int(data[0])  # 有几个可选机器
            machine_list = []
            process_list = []
            data = data[1:]
            for i in range(process_num):
                machine_list.append(int(data[i * 2]) - 1)
                process_list.append(int(data[i * 2 + 1]))
            job.append(machine_list)
            process_time.append(process_list)
            data = data[i * 2 + 2:]
            l += 1
        J.append(job)
        P.append(process_time)
    return J, P, num_jobs, num_machines


