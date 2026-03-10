import igneous.task_creation as tc
import os
from taskqueue import totask
import igneous.tasks


def process_task(msg):
    task = totask(msg)
    task.execute()
    return None

def submit_tasks():
    paths = [
        "gs://ng_scratch_ranl_7/make_cv_happy/seg/20250403024338",]
    all_tasks = []
    for img_path in paths:
        mip = 0
        num_mips = 2
        tasks = tc.create_downsampling_tasks(img_path,
                fill_missing=False,
                delete_black_uploads=True,
                mip=mip, num_mips=num_mips)
        all_tasks += list(tasks)
    return all_tasks

if __name__ == "__main__":
    tasks = submit_tasks()
    process_task(tasks[0])
