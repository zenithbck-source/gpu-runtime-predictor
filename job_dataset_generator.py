import pandas as pd
import random

def generate_training_data(n, seed=42):
    data = []

    for i in range(0, n):
        ncpus = random.choice([1,2,4,8,16])
        mem_gb = random.choice([1,2,4,8,16,32])
        submit_hour = random.randint(0, 23)
        queue_type = random.choice(['short', 'normal', 'long'])
        data.append({'ncpus':ncpus, 'mem_gb':mem_gb, 'submit_hour':submit_hour, 'queue_type':queue_type})

    return data

if __name__ == '__main__':
    data = generate_training_data(2)
    print(data)