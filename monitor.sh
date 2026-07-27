#!/bin/bash
# LiveTalking 资源监控脚本
# 每 5 秒采样一次，记录 CPU/内存/GPU/Session 变化

LOG="/tmp/livetalking_monitor.log"
SAMPLE_INTERVAL=5

echo "===============================================" >> "$LOG"
echo "  LiveTalking Monitor — $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
echo "  采样间隔: ${SAMPLE_INTERVAL}s" >> "$LOG"
echo "===============================================" >> "$LOG"
printf "%-20s %10s %10s %8s %8s %8s %10s %10s\n" \
    "TIME" "RAM(MB)" "GPU(MB)" "CPU%" "SESS" "CONN" "FPS" "PUSHED" >> "$LOG"
echo "-------------------------------------------------------------------------------" >> "$LOG"

while true; do
    TS=$(date '+%H:%M:%S')
    
    # ── 进程资源 ──
    PROC=$(ps aux | grep "python.*app.py" | grep -v grep | head -1)
    if [ -z "$PROC" ]; then
        printf "%-20s %10s %10s %8s %8s %8s %10s\n" \
            "$TS" "DEAD" "-" "-" "-" "-" >> "$LOG"
        echo "[$(date)] ⚠️ 进程已停止" >> "$LOG"
        exit 1
    fi
    RSS=$(echo "$PROC" | awk '{print int($6/1024)}')
    CPU=$(echo "$PROC" | awk '{printf "%.0f", $3}')
    
    # ── GPU 显存 ──
    GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | grep -oP '\d+')
    [ -z "$GPU" ] && GPU="-"
    
    # ── 连接数 ──
    CONN=$(ss -tn state established '( sport = :5001 )' 2>/dev/null | tail -n +2 | wc -l)
    
    # ── Session 数 ──
    SESS=$(curl -sk --connect-timeout 2 https://localhost:5001/api/admin/sessions 2>/dev/null | \
           python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('data',{}).get('sessions',[])))" 2>/dev/null)
    [ -z "$SESS" ] && SESS="-"
    
    # ── FPS（从日志最近一行 pacing 提取） ──
    FPS=$(tail -100 /tmp/livetalking.log 2>/dev/null | grep "pacing.*pushed" | tail -1 | grep -oP '[\d.]+(?= fps)')
    FRAMES=$(tail -100 /tmp/livetalking.log 2>/dev/null | grep "pacing.*pushed" | tail -1 | grep -oP '\d+(?= frames)')
    [ -z "$FPS" ] && FPS="-"
    [ -z "$FRAMES" ] && FRAMES="-"
    
    printf "%-20s %10s %10s %8s %8s %8s %10s %10s\n" \
        "$TS" "${RSS}" "${GPU}" "${CPU}%" "${SESS}" "${CONN}" "${FPS}" "${FRAMES}" >> "$LOG"
    
    sleep "$SAMPLE_INTERVAL"
done
