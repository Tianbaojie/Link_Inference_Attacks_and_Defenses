import os
import re
import time
import sys

# 获取显存使用情况
def parseGPUMem(str_content):
    lines = str_content.split("\n")
    target_line = lines[9]
    mem_part = target_line.split("|")[2]
    use_mem = mem_part.split("/")[0]
    total_mem = mem_part.split("/")[1]
    use_mem_int = int(re.sub("\D", "", use_mem))
    total_mem_int = int(re.sub("\D", "", total_mem))
    return use_mem_int, total_mem_int

# 获取GPU使用情况
def parseGPUUseage(str_content):
    lines = str_content.split("\n")
    target_line = lines[9]
    useage_part = int(target_line.split("|")[3].split("%")[0])
    return useage_part

# 获取监控进程显存使用情况
def parseProcessMem(str_content, process_name):
    part = str_content.split("|  GPU       PID   Type   Process name                             Usage      |")[1]
    lines = part.split("\n")
    for i in range(len(lines)):
        line = lines[i]
        if line.__contains__(process_name):
            mem_use = int(line[-10:-5])
            return mem_use


if __name__ == '__main__':
    str_command = "nvidia-smi"  # 需要执行的命令
    process_name = sys.argv[1]  # 待监控的进程名称
    out_path = "./time_memory_log/GPU_stat_"+process_name+".txt"
    time_interval = 0.5

    fout = open(out_path, "w")
    fout.write("Timestamp\tGPU_Usage_Percentage\tGPU_Total_Mem_Usage\tGPU_Total_Mem_Usage_Percentage\n")

    while True:
        out = os.popen(str_command)
        text_content = out.read()
        out.close()
        usage_percentage = parseGPUUseage(text_content)
        use_mem, total_mem = parseGPUMem(text_content)
        use_percent = round(use_mem * 100.0 / total_mem, 2)

        str_outline = str(time.time()) + "\t" + str(usage_percentage) + "\t" + str(use_mem) + "\t" + str(use_percent) + "\n"
        fout.write(str_outline)
        fout.flush()
        time.sleep(time_interval)