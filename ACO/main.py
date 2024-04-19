from ACO_fuction import *
import sys

sys.path.append('.')

# 迭代次数
iteration_num = 40

# 蚂蚁数量
ant_num = 30

# 绘图用
iter_list = range(iteration_num)
time_list = [0 for _ in iter_list]

best_topo_jobs = None
best_process2machine = None

# 对于每次迭代
for it in iter_list:
    # 每次迭代寻找最优的<拓扑序,机器分配>方式
    best_time = 9999999
    # 对于每只蚂蚁
    for ant_id in range(ant_num):
        # 生成拓扑序
        topo_jobs = gen_topo_jobs()
        # 生成每道工序的分配机器索引号矩阵
        process2machine = gen_process2machine()
        # 计算时间
        time = cal_time(topo_jobs, process2machine)
        # 如果时间更短，更新最优
        if time < best_time:
            best_topo_jobs = topo_jobs
            best_process2machine = process2machine
            best_time = time
    assert best_topo_jobs is not None and best_process2machine is not None
    # 更新拓扑序信息素浓度表
    for i in range(sum(process_num)):
        for j in range(job_num):
            if j == best_topo_jobs[i]:
                topo_phs[i][j] *= 1.1
            else:
                topo_phs[i][j] *= 0.9
    # 更新每个Job的每个工序的信息素浓度表
    for j in range(job_num):
        for p in range(process_num[j]):
            for m in range(machine_num):
                if m == best_process2machine[j][p]:
                    machine_phs[j][p][m] *= 1.1
                else:
                    machine_phs[j][p][m] *= 0.9
    # 记录时间
    time_list[it] = best_time

# 输出解
print("\t\t[工序分配给机器的情况]")
print("\t\t", end='')
for machine_id in range(machine_num):
    print("\tM{}".format(machine_id + 1), end='')
print()
for job_id in range(job_num):
    for process_id in range(process_num[job_id]):
        print("p{}-{}\t".format(job_id + 1, process_id + 1), end='')
        for machine_id in range(machine_num):
            if machine_id == best_process2machine[job_id][process_id]:
                print("\t√", end='')
            else:
                print("\t-", end='')
        print("")

print("\n\t\t[工序投给机器的顺序]")
job_use = [0 for _ in range(job_num)]
for job_id in best_topo_jobs:
    print("p{}-{} ".format(job_id + 1, job_use[job_id] + 1), end='')
    job_use[job_id] += 1

# 绘图
plt.plot(iter_list, time_list)
plt.xlabel("迭代轮次")
plt.ylabel("时间")
plt.title("FJSP-ACO")
plt.show()

# Call the function to plot the Gantt chart
plot_gantt_chart(best_topo_jobs, best_process2machine, times)