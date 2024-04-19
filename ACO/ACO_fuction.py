"""
柔性作业车间调度问题
(Flexible Job-shop Scheduling Problem, FJSP)
"""
import numpy as np
import random
from typing import List

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def parse_input(input_data):
    lines = input_data.split('\n')
    num_jobs, num_machines = map(int, lines[0].split()[:2])
    times = []

    current_line = 1
    for _ in range(num_jobs):
        job_data = lines[current_line].split()
        num_operations = int(job_data[0])
        job_times = []
        op_info_index = 1
        for _ in range(num_operations):
            op_count = int(job_data[op_info_index])
            machines_times = [None] * num_machines
            for i in range(op_count):
                machine = int(job_data[op_info_index + 1 + i * 2])
                time = int(job_data[op_info_index + 2 + i * 2])
                machines_times[machine - 1] = time
            job_times.append(machines_times)
            op_info_index += 1 + 2 * op_count
        times.append(job_times)
        current_line += 1

    return times


def return_times(filename):
    with open(filename, 'r') as file:
        input_data = file.read()
    times = parse_input(input_data)

    with open(filename, 'r') as file:
        data = file.readlines()
    first_line = data[0].strip().split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    data = data[1:]
    process_num = []
    for line in data:
        line = line.strip().split()
        if not line:
            continue
        process_num.append(int(line[0]))
    return num_jobs, num_machines, process_num, times


# 在这里修改想要读取的文件
job_num, machine_num, process_num, times = return_times('Brandimarte_Data/Text/Mk01.fjs')

# 拓扑序的信息素浓度，初始值100
topo_phs = [
    [100 for _ in range(job_num)]
    for num in range(sum(process_num))
]


def gen_topo_jobs() -> List[int]:
    """
    生成拓扑序
    Job在时空上处理的的拓扑序(Job索引)，这个序不能体现工序选择的机器
    :return 如[0,1,0,2,2,...]表示p11,p21,p12,p31,p32,...
    """
    # 按照每个位置的信息素浓度加权随机给出
    # 返回的序列长，是Job数量*工序的数量
    len = sum(process_num)
    # 返回的序列，最后这些-1都会被设置成0~job_num-1之间的索引
    ans = [-1 for _ in range(len)]
    # 记录每个job使用过的次数，用来防止job被使用超过process_num次
    job_use = [0 for _ in range(job_num)]
    # 记录现在还没超过process_num因此可用的job_id，每次满了就将其删除
    job_free = [job_id for job_id in range(job_num)]
    # 对于序列的每个位置
    for i in range(len):
        # 把这个位置可用的job的信息素浓度求和
        ph_sum = np.sum(list(map(lambda j: topo_phs[i][j], job_free)))
        # 接下来要随机在job_free中取一个job_id
        # 但是不能直接random.choice，要考虑每个job的信息素浓度
        test_val = .0
        rand_ph = random.uniform(0, ph_sum)
        for job_id in job_free:
            test_val += topo_phs[i][job_id]
            if rand_ph <= test_val:
                # 将序列的这个位置设置为job_id，并维护job_use和job_free
                ans[i] = job_id
                job_use[job_id] += 1
                if job_use[job_id] == process_num[job_id]:
                    job_free.remove(job_id)
                break
    return ans


# 每个Job的每个工序的信息素浓度，初始值100
machine_phs = [
    [
        [100 for _ in range(machine_num)]
        for _ in range(num)
    ]
    for num in process_num
]


def gen_process2machine() -> List[List[int]]:
    """
    生成每个Job的每个工序对应的机器索引号矩阵
    :return: 不规则二维列表
    """
    # 要返回的矩阵，取值0~machine_num-1
    ans = [
        [-1 for _ in range(num)]
        for num in process_num
    ]
    # 对于每个位置，也是用信息素加权随机出一个machine_id即可
    for job_id in range(job_num):
        for process_id in range(process_num[job_id]):
            # 获取该位置的所有可用机器号(times里不为None)
            machine_free = [machine_id for machine_id in range(machine_num)
                            if times[job_id][process_id][machine_id] is not None]
            # 计算该位置所有可用机器的信息素之和
            ph_sum = np.sum(list(map(lambda m: machine_phs[job_id][process_id][m], machine_free)))
            # 还是用随机数的方式选取
            test_val = .0
            rand_ph = random.uniform(0, ph_sum)
            for machine_id in machine_free:
                test_val += machine_phs[job_id][process_id][machine_id]
                if rand_ph <= test_val:
                    ans[job_id][process_id] = machine_id
                    break
    return ans


def cal_time(topo_jobs: List[int], process2machine: List[List[int]]):
    """
    给定拓扑序和机器索引号矩阵
    :return: 计算出的总时间花费
    """
    # 记录每个job在拓扑序中出现的次数，以确定是第几个工序
    job_use = [0 for _ in range(job_num)]
    # 循环中要不断查询和更新这两张表
    # (1)每个machine上一道工序的结束时间
    machine_end_times = [0 for _ in range(machine_num)]
    # (2)每个工件上一道工序的结束时间
    job_end_times = [0 for _ in range(job_num)]
    # 对拓扑序中的每个job_id
    for job_id in topo_jobs:
        # 在job_use中取出工序号
        process_id = job_use[job_id]
        # 在process2machine中取出机器号
        machine_id = process2machine[job_id][process_id]
        # 获取max(该job上一工序时间,该machine上一任务完成时间)
        now_start_time = max(job_end_times[job_id], machine_end_times[machine_id])
        # 计算当前结束时间，写入这两个表
        job_end_times[job_id] = machine_end_times[machine_id] = now_start_time + times[job_id][process_id][machine_id]
        # 维护job_use
        job_use[job_id] += 1
    return max(job_end_times)


def plot_gantt_chart(topo_jobs, process2machine, times):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(10, 6))

    job_use = [0 for _ in range(job_num)]
    tasks = []

    color_map = {}

    # (1)每个machine上一道工序的结束时间
    machine_end_times = [0 for _ in range(machine_num)]
    # (2)每个工件上一道工序的结束时间
    job_end_times = [0 for _ in range(job_num)]
    for job_id in topo_jobs:
        process_id = job_use[job_id]

        machine_id = process2machine[job_id][process_id]

        now_start_time = max(job_end_times[job_id], machine_end_times[machine_id])
        # 计算当前结束时间，写入这两个表
        job_end_times[job_id] = machine_end_times[machine_id] = now_start_time + times[job_id][process_id][machine_id]
        # 维护job_use
        job_use[job_id] += 1

        # Append the task to the list
        tasks.append((now_start_time, job_end_times[job_id], job_id, process_id, machine_id))
        if job_id not in color_map:
            color_map[job_id] = (random.random(), random.random(), random.random())

    for task in tasks:
        start, end, job_id, process_id, machine_id = task

        color = color_map[job_id]

        ax.barh(machine_id, end - start, left=start, height=0.4, color=color, edgecolor='black')

        label = f'{job_id + 1}-{process_id + 1}'
        ax.text((start + end) / 2, machine_id, label, ha='center', va='center', color='white')

    ax.set_yticks(range(machine_num))
    ax.set_yticklabels([f'机器{i + 1}' for i in range(machine_num)])

    plt.xlabel('时间')

    plt.title('FJSP_Solution_ACO')

    legend_handles = []
    for job_idx, color in color_map.items():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'job{job_idx + 1}'))
    plt.legend(handles=legend_handles, title='工件', bbox_to_anchor=(1.0, 1.0))

    plt.savefig('Mk01.png')
    plt.show()
