
import psutil


def is_process_running(pid_name):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    
    if isinstance(pid_name, int):
        return is_process_running_from_pid(pid_name)
    elif isinstance(pid_name, str):
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if pid_name.lower() in proc.name().lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    return False

def is_process_running_from_pid(pid):
    for proc in psutil.process_iter():
        if proc.pid == pid:
            return True
    return False