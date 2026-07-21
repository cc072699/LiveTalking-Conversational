import os
import time
import subprocess
import re
from datetime import datetime

log_file = 'livetalking.log'
file_position = 0
sessions = {}

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def get_livetalking_pid():
    cmd = "ps aux | grep 'venv/bin/python app.py' | grep -v grep | grep -v 'bash -c' | head -1"
    stdout, _ = run_cmd(cmd)
    if stdout:
        parts = stdout.split()
        return parts[1]
    return None

def get_process_info(pid):
    if not pid:
        return None
    cmd = f"ps aux | grep {pid} | grep -v grep"
    stdout, _ = run_cmd(cmd)
    if stdout:
        parts = stdout.split()
        return {
            'pid': parts[1],
            'user': parts[0],
            'cpu': parts[2],
            'mem': parts[3],
            'vsz': parts[4],
            'rss': parts[5],
            'status': parts[7],
            'time': parts[10],
            'cmd': ' '.join(parts[11:])
        }
    return None

def get_gpu_info():
    stdout, _ = run_cmd("nvidia-smi")
    if not stdout:
        return {"error": "nvidia-smi not available"}
    
    gpus = []
    lines = stdout.split('\n')
    for i, line in enumerate(lines):
        if 'MiB /' in line and '%' in line and i > 0:
            gpu_info = {}
            try:
                info_parts = line.split('|')
                name_line = lines[i-1]
                name_parts = name_line.split('|')
                
                if len(name_parts) >= 2:
                    name_info = name_parts[1].strip().split()
                    gpu_info['id'] = name_info[0] if name_info else 'N/A'
                    gpu_info['name'] = ' '.join(name_info[1:]) if len(name_info) > 1 else 'Unknown'
                
                if len(info_parts) >= 2:
                    temp_power = info_parts[1].strip().split()
                    gpu_info['temp'] = temp_power[1].replace('C', '') if len(temp_power) > 1 else 'N/A'
                    gpu_info['power'] = temp_power[3] if len(temp_power) > 3 else 'N/A'
                
                if len(info_parts) >= 3:
                    mem_info = info_parts[2].strip().split()
                    gpu_info['memory'] = mem_info[0] if mem_info else 'N/A'
                    gpu_info['memory_total'] = mem_info[2] if len(mem_info) > 2 else 'N/A'
                
                if len(info_parts) >= 4:
                    util_info = info_parts[3].strip().split()
                    gpu_info['util'] = util_info[0] if util_info else 'N/A'
                
                gpus.append(gpu_info)
            except:
                pass
    return {"gpus": gpus}

def get_memory_info():
    stdout, _ = run_cmd("free -h")
    lines = stdout.split('\n')
    result = {
        'total': 'N/A', 'used': 'N/A', 'free': 'N/A', 'available': 'N/A',
        'swap_total': 'N/A', 'swap_used': 'N/A', 'swap_free': 'N/A'
    }
    try:
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 7:
                result['total'] = parts[1]
                result['used'] = parts[2]
                result['free'] = parts[3]
                result['available'] = parts[6]
        if len(lines) > 3:
            swap_parts = lines[3].split()
            if len(swap_parts) >= 4:
                result['swap_total'] = swap_parts[1]
                result['swap_used'] = swap_parts[2]
                result['swap_free'] = swap_parts[3]
    except:
        pass
    return result

def get_cpu_info():
    stdout, _ = run_cmd("top -bn1 | head -5")
    result = {'user': 'N/A', 'system': 'N/A', 'idle': 'N/A'}
    try:
        lines = stdout.split('\n')
        for line in lines:
            if 'Cpu(s):' in line or '%Cpu(s):' in line:
                cpu_line = line
                parts = cpu_line.split(',')
                if len(parts) >= 4:
                    result['user'] = parts[0].split(':')[1].strip().split()[0] if ':' in parts[0] else 'N/A'
                    result['system'] = parts[1].strip().split()[0] if len(parts) > 1 else 'N/A'
                    result['idle'] = parts[3].strip().split()[0] if len(parts) > 3 else 'N/A'
                break
    except:
        pass
    return result

def get_new_logs():
    global file_position
    new_logs = []
    try:
        with open(log_file, 'r') as f:
            f.seek(file_position)
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    new_logs.append(line)
            file_position = f.tell()
    except:
        pass
    return new_logs

def get_latest_logs(lines=100):
    stdout, _ = run_cmd(f"tail -{lines} {log_file}")
    logs = []
    if stdout:
        for line in stdout.split('\n'):
            if line:
                logs.append(line)
    return logs

def update_dialog_state(new_logs):
    tts_status = '未知'
    tts_session = None
    inference_active = False
    errors = []
    recent_dialogs = []
    
    for log in new_logs:
        if 'Creating sessionid=' in log:
            match = re.search(r'sessionid=([^,]+)', log)
            if match:
                sid = match.group(1)
                sessions[sid] = {'status': 'created', 'time': log[:23], 'messages': []}
        
        if 'offer sessionid=' in log:
            match = re.search(r'sessionid=(\S+)', log)
            if match and match.group(1) in sessions:
                sessions[match.group(1)]['status'] = 'offered'
        
        if 'Connection state is connecting' in log:
            for sid in sessions:
                if sessions[sid]['status'] in ['created', 'offered']:
                    sessions[sid]['status'] = 'connecting'
        
        if 'Connection state is connected' in log:
            for sid in sessions:
                if sessions[sid]['status'] == 'connecting':
                    sessions[sid]['status'] = 'connected'
        
        if 'Connection state is failed' in log:
            for sid in sessions:
                if sessions[sid]['status'] == 'connecting':
                    sessions[sid]['status'] = 'failed'
        
        if 'Connection state is closed' in log:
            for sid in sessions:
                if sessions[sid]['status'] == 'connected':
                    sessions[sid]['status'] = 'closed'
        
        if 'Removing session' in log:
            match = re.search(r'session (\S+)', log)
            if match and match.group(1) in sessions:
                del sessions[match.group(1)]
        
        if 'QwenTTS WebSocket 连接已建立' in log:
            tts_status = '✅ 已连接'
        if 'QwenTTS 初始化完成' in log:
            tts_status = '✅ 初始化完成'
        if 'QwenTTS WebSocket 关闭' in log:
            tts_status = '❌ 已关闭'
        if 'QwenTTS reconnected' in log:
            tts_status = '✅ 已重连'
        if 'QwenTTS session:' in log:
            tts_session = log.split(':')[-1].strip()
        
        if 'start inference' in log:
            inference_active = True
        if 'inference thread stop' in log:
            inference_active = False
        
        if 'QwenTTS 合成完成' in log:
            recent_dialogs.append(log)
            for sid in sessions:
                if sessions[sid]['status'] == 'connected':
                    sessions[sid]['messages'].append(log)
        
        if 'inference 状态切换' in log:
            recent_dialogs.append(log)
            for sid in sessions:
                if sessions[sid]['status'] == 'connected':
                    sessions[sid]['messages'].append(log)
        
        if '状态切换' in log and ('静音' in log or '说话' in log):
            recent_dialogs.append(log)
        
        if 'notify:' in log:
            recent_dialogs.append(log)
            for sid in sessions:
                if sessions[sid]['status'] == 'connected':
                    sessions[sid]['messages'].append(log)
        
        if 'ERROR' in log:
            errors.append(log)
    
    active_count = sum(1 for s in sessions.values() if s['status'] == 'connected')
    
    return {
        'total_sessions': len(sessions),
        'active_sessions': active_count,
        'tts_status': tts_status,
        'tts_session': tts_session,
        'inference_active': inference_active,
        'errors': errors[-5:],
        'recent_dialogs': recent_dialogs[-5:]
    }

def get_fps_from_logs(logs):
    fps_values = []
    for log in logs:
        if 'actual avg final fps:' in log:
            match = re.search(r'fps:([\d.]+)', log)
            if match:
                fps_values.append(float(match.group(1)))
    
    if fps_values:
        avg_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
    else:
        avg_fps = min_fps = max_fps = 0
    
    return {
        'avg_fps': round(avg_fps, 2),
        'min_fps': round(min_fps, 2),
        'max_fps': round(max_fps, 2)
    }

def get_port_status(port):
    stdout, _ = run_cmd(f"ss -tlnp | grep {port}")
    if stdout:
        parts = stdout.split()
        return {
            'port': parts[3].split(':')[-1],
            'state': parts[0],
            'process': parts[-1]
        }
    return {'port': port, 'state': 'not listening'}

def print_status():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=" * 70)
    print(f"LiveTalking 实时监控 - {now}")
    print("=" * 70)
    
    pid = get_livetalking_pid()
    process_info = get_process_info(pid)
    if process_info:
        print("\n【进程状态】")
        print(f"  PID: {process_info['pid']}")
        print(f"  用户: {process_info['user']}")
        print(f"  CPU: {process_info['cpu']}%")
        print(f"  内存: {process_info['mem']}%")
        print(f"  状态: {process_info['status']}")
        print(f"  运行时间: {process_info['time']}")
        print(f"  命令: {process_info['cmd']}")
    else:
        print("\n【进程状态】")
        print("  ❌ 进程未运行!")
    
    print("\n【端口状态】")
    logs = get_latest_logs(50)
    listen_port = 8051
    for log in logs:
        if 'start http server' in log.lower():
            match = re.search(r':(\d+)', log)
            if match:
                listen_port = int(match.group(1))
                break
    port_status = get_port_status(listen_port)
    print(f"  端口 {listen_port}: {port_status['state']}")
    if 'listening' in port_status['state'] and 'process' in port_status:
        print(f"  进程: {port_status['process']}")
    
    print("\n【GPU 使用】")
    gpu_info = get_gpu_info()
    if 'error' in gpu_info:
        print(f"  {gpu_info['error']}")
    else:
        for gpu in gpu_info['gpus']:
            try:
                util_value = float(gpu['util'].replace('%', '')) if gpu['util'] != 'N/A' else 0
                status = "✅" if util_value > 0 else "⏸️"
            except:
                status = "⚠️"
            print(f"  {status} GPU{gpu['id']}: {gpu['name']}")
            print(f"     温度: {gpu['temp']}C | 功耗: {gpu['power']}")
            print(f"     显存: {gpu['memory']} / {gpu['memory_total']} | 利用率: {gpu['util']}")
    
    print("\n【内存使用】")
    mem_info = get_memory_info()
    print(f"  总内存: {mem_info['total']} | 已用: {mem_info['used']} | 可用: {mem_info['available']}")
    print(f"  Swap: {mem_info['swap_used']} / {mem_info['swap_total']}")
    
    print("\n【CPU 使用】")
    cpu_info = get_cpu_info()
    print(f"  用户态: {cpu_info['user']}% | 系统态: {cpu_info['system']}% | 空闲: {cpu_info['idle']}%")
    
    print("\n【FPS 统计】")
    fps_info = get_fps_from_logs(logs)
    fps_status = "✅" if fps_info['avg_fps'] >= 25 else "⚠️" if fps_info['avg_fps'] >= 20 else "❌"
    print(f"  {fps_status} 平均 FPS: {fps_info['avg_fps']}")
    print(f"  最小 FPS: {fps_info['min_fps']} | 最大 FPS: {fps_info['max_fps']}")
    
    print("\n【对话状态】")
    new_logs = get_new_logs()
    dialog_info = update_dialog_state(new_logs)
    print(f"  总会话数: {dialog_info['total_sessions']}")
    print(f"  活跃会话: {'✅' if dialog_info['active_sessions'] > 0 else '❌'} {dialog_info['active_sessions']}")
    print(f"  TTS 状态: {dialog_info['tts_status']}")
    print(f"  推理状态: {'✅ 进行中' if dialog_info['inference_active'] else '⏸️ 停止'}")
    if dialog_info['tts_session']:
        print(f"  TTS Session: {dialog_info['tts_session']}")
    
    if sessions:
        print("\n【会话详情】")
        for sid, info in sessions.items():
            status_icon = {
                'created': '🔄', 'offered': '📤', 'connecting': '🔗',
                'connected': '✅', 'failed': '❌', 'closed': '⏹️'
            }.get(info['status'], '❓')
            print(f"  {status_icon} Session: {sid[:12]}...")
            print(f"     状态: {info['status']} | 创建时间: {info['time']}")
            if info['messages']:
                print(f"     消息数: {len(info['messages'])}")
                for msg in info['messages'][-3:]:
                    if 'notify:' in msg:
                        msg_text = msg
                    elif '合成完成' in msg:
                        msg_text = msg
                    elif '状态切换' in msg:
                        msg_text = msg
                    else:
                        msg_text = msg[:60]
                    print(f"        - {msg_text}")
    
    if dialog_info['recent_dialogs']:
        print("\n【最近对话】")
        for dialog in dialog_info['recent_dialogs']:
            if 'notify:' in dialog:
                print(f"  🗣️ {dialog}")
            elif '合成完成' in dialog:
                print(f"  🎤 {dialog}")
            elif '状态切换' in dialog:
                print(f"  🔄 {dialog}")
            else:
                print(f"  {dialog}")
    
    if dialog_info['errors']:
        print("\n【错误告警】")
        for error in dialog_info['errors']:
            print(f"  ❌ {error}")
    
    print("\n【最新日志】")
    print("-" * 70)
    for log in logs[-15:]:
        if 'ERROR' in log:
            print(f"  ❌ {log}")
        elif 'WARNING' in log and 'Subtitle' in log:
            pass
        elif 'INFO' in log and ('fps' in log.lower() or 'session' in log.lower() or 'connection' in log.lower() or 'TTS' in log or 'inference' in log.lower() or 'notify' in log.lower()):
            print(f"  {log}")
    
    print("\n" + "=" * 70)
    print("按 Ctrl+C 停止监控")
    print("=" * 70)

if __name__ == '__main__':
    get_latest_logs(200)
    try:
        while True:
            print_status()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n监控已停止")