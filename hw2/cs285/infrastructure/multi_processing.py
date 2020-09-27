import json
import math
import os
import time
from pprint import pprint
import numpy as np
import multiprocessing as mp
import argparse
import sys

import concurrent.futures

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def run_multiprocessing_tasks(
    tasks,
    thread_func,
    func_args=(),
    num_cores=4,
    verbose=False,
    join_results=False,
    use_threading=False
):
    # execute pipeline in a parallel way
    last_time = time.time()

    # get parallel_arguments
    if tasks:
        parallel_arguments = []
        num_tasks_per_core = math.ceil(len(tasks)/num_cores)
        for i in range(num_cores):
            parallel_arguments.append(
                (tasks[i*num_tasks_per_core: (i+1)*num_tasks_per_core], ) + func_args
            )
    else:
        parallel_arguments = [func_args] * num_cores

    if not use_threading:
        # running using mp
        p = mp.Pool(processes=num_cores)
        all_summary = p.starmap(thread_func, parallel_arguments)
        p.close()
        p.join()

    else:
        # running using threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures =[
                executor.submit(thread_func, *(parallel_arguments[i]))
                for i in range(num_cores)
            ]
            all_summary = [f.result() for f in futures]

    # # running using torch.mp
    # processes = []
    # for i in range(num_cores):
    #     p = mp.Process(
    #         target=thread_func,
    #         args=parallel_arguments[i]
    #     )
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    if verbose:
        # reading results
        print('time used:', time.time()-last_time)
        if isinstance(all_summary[0], dict) and 'success_tasks' in all_summary[0]:
            # combine all results
            all_success_tasks = sum([tmp_summary['success_tasks'] for tmp_summary in all_summary], [])
            print('len(all_success_tasks)', len(all_success_tasks))

        if isinstance(all_summary[0], dict) and 'error_tasks' in all_summary[0]:
            # combine all error tasks
            all_error_tasks = sum([tmp_summary['error_tasks'] for tmp_summary in all_summary], [])
            print('len(all_error_tasks)', len(all_error_tasks))

    if join_results and isinstance(all_summary[0], tuple):
        # when output is multiple variables (a tuple), the mp output is a tuple,
        #   where each variable is a list of results from each processor.
        #   Therefore, need to sum to combine results in each variable
        last_results = []
        for i in range(len(all_summary[0])):
            last_results.append([x[i] for x in all_summary])
    else:
        last_results = all_summary

    return last_results

def save_results(results, dir_path='../generated/results', prefix='results'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = '{}_{}.json'.format(prefix, str(hash(str(results))))
    with open(os.path.join(dir_path, file_name), 'w') as fw:
        json.dump(results, fw, indent=2)
    return file_name

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

def use_file_as_stdout(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    sys.stdout = open(file_path, 'w')
    sys.stdout = Unbuffered(sys.stdout)
    print('this is printed in the console')
