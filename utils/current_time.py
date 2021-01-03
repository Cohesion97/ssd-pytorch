import time

def time_to_hour(sec):
    hour = sec // 3600
    min = (sec % 3600)//60
    sec = (sec%60)
    return int(hour),int(min),int(sec)

def get_current_time():
    time_stamp = time.time()  # 当前时间的时间戳
    local_time = time.localtime(time_stamp)  #
    str_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    return str_time