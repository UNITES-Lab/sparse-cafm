#!/bin/bash
while true; do
    clear
    echo -e "\e[1;32m$(date)\e[0m"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F ', ' '
    {printf "\033[1;34mGPU: %s\033[0m | Temp: \033[1;31m%s°C\033[0m | Util: \033[1;32m%s%%\033[0m | Mem: \033[1;33m%s/%s MiB\033[0m\n", $1, $2, $3, $4, $5}'
    sleep 1
done

