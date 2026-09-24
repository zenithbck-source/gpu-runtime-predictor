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

def mapping_data(dataset):
    queue_mapping = {'short': 0, 'normal' : 1, 'long':2}
    dataset = pd.DataFrame(dataset)
    dataset['queue_type'] = dataset['queue_type'].map(queue_mapping)

    dataset['actual_runtime'] = dataset['ncpus'] * 2 + dataset['mem_gb'] * 1.5 + (dataset['queue_type'] * 20) + 15

    return dataset


if __name__ == '__main__':
    fresh_data = generate_training_data(1000)
    final_data = mapping_data(fresh_data)
    final_data.to_csv('training_data.csv')