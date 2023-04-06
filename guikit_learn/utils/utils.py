import os
import psutil
import shutil
import time


def delayed_start(threshold_cpu=50):
    while True:
        cpu_p = psutil.cpu_percent(interval=1)
        print(f"CPU使用率({cpu_p}%)が{threshold_cpu}%になるまで待機します")
        if cpu_p < threshold_cpu:
            break
        time.sleep(20)
    return

# 解析結果のアーカイブ化
def make_archive_then_delete(target_folder, zip_path=None):
    if os.path.exists(target_folder) == False:
        print(f"{target_folder} does not exits.")
        time.sleep(0.2)
        return

    if zip_path is None:
        zip_path = target_folder

    shutil.make_archive(target_folder, format="zip", root_dir=target_folder)
    shutil.rmtree(target_folder)