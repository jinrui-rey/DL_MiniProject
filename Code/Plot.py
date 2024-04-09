import os
import matplotlib.pyplot as plt
import argparse


def filter_para(path, para):
    assert para.lower() in ['optimizer', 'lr', 'batch', 'decay_step', 'lr_decay_rate'], \
        "Variable must be in ['optimizer','lr','batch','decay_step','lr_decay_rate']"

    stragegy = ['SGD', 'Adam', 'ASGD', 'Adagrad']

    file_list = os.listdir(path)
    log_list = [file for file in file_list if file.endswith('.txt') and file.startswith('log_')]
    all_args = []
    for i in range(len(log_list)):
        file = os.path.join(path, log_list[i])
        with open(file, 'r') as f:
            NameSpace = f.readlines()[0]
            content = NameSpace.split("Namespace(")[1].strip(")\n").split(", ")
            args_dict = {}
            for item in content:
                key, value = item.split('=')
                value = eval(value)  # Evaluate the string representation of the value
                args_dict[key] = value
            all_args.append(args_dict)

    timestamp_list = []

    if para.lower() == 'optimizer':
        for args in all_args:
            if args['optimizer'] in stragegy and args['lr'] == 0.1:
                timestamp_list.append(args['timestamp'])

    if para.lower() == 'lr':
        for args in all_args:
            if args['optimizer'] == 'SGD' and args['batch'] == 64:
                timestamp_list.append(args['timestamp'])

    if para.lower() == 'batch':
        for args in all_args:
            if args['optimizer'] == 'SGD' and args['lr'] == 0.3 and args['decay_step'] == 5000:
                timestamp_list.append(args['timestamp'])

    if para.lower() == 'decay_step':
        for args in all_args:
            if args['optimizer'] == 'SGD' and args['batch'] == 128 and args['lr_decay_rate'] == 0.2:
                timestamp_list.append(args['timestamp'])

    filtered_list = []
    for log in log_list:
        if log[4:-4] in timestamp_list:
            filtered_list.append(log)

    return filtered_list


def draw(path, files, name):
    All_loss = []
    for i in range(len(files)):
        loss = []
        file_add = os.path.join(path, files[i])
        with open(file_add, 'r') as f:
            for line in f:
                if line.startswith('Namespace'):
                    content = line.split("Namespace(")[1].strip(")\n").split(", ")
                    args_dict = {}
                    for item in content:
                        key, value = item.split('=')
                        value = eval(value)  # Evaluate the string representation of the value
                        args_dict[key] = value

                columns = line.split(',')
                if columns[-1].startswith(' loss:'):
                    loss.append(float(columns[-1][7:]))

        loss.append(args_dict)
        All_loss.append(loss)

    for i in range(len(All_loss)):
        id = (str(All_loss[i][-1]['optimizer']) + ',lr_' + str(All_loss[i][-1]['lr']) + ',bs_' + str(
            All_loss[i][-1]['batch']) + ',ds_' + str(All_loss[i][-1]['decay_step']))
        plt.plot(All_loss[i][:-1], '-', linewidth=3, label=id)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.title('Experiment on '+str(name))
    plt.savefig(str(name)+'.png')
    plt.show()


def plot_best(path):
    file_list = os.listdir(path)
    log_list = [file for file in file_list if file.endswith('.txt') and file.startswith('log_')]
    id = []
    for i in range(len(log_list)):
        file = os.path.join(path, log_list[i])
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('The final accuracy is:'):
                    acc = eval(line.split(':')[-1])
                    id.append((acc,log_list[i][4:-4]))

    max_tuple = max(id, key=lambda x: x[0])

    best_id = max_tuple[1]

    file_name = 'log_' + str(best_id) + '.txt'
    file_add = os.path.join(path, file_name)
    loss =[]
    with open(file_add, 'r') as f:
        for line in f:
            if line.startswith('Namespace'):
                content = line.split("Namespace(")[1].strip(")\n").split(", ")
                args_dict = {}
                for item in content:
                    key, value = item.split('=')
                    value = eval(value)  # Evaluate the string representation of the value
                    args_dict[key] = value

            columns = line.split(',')
            if columns[-1].startswith(' loss:'):
                loss.append(float(columns[-1][7:]))
    loss.append(args_dict)
    id = (str(loss[-1]['optimizer']) + ',lr_' + str(loss[-1]['lr']) + ',bs_' + str(
        loss[-1]['batch']) + ',ds_' + str(loss[-1]['decay_step'])) + ',dr_'+str(loss[-1]['lr_decay_rate'])

    plt.plot(loss[:-1], '-', linewidth=3, label=id)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Final Acc='+str(max_tuple[0])+'%')
    plt.legend()
    plt.grid(True)
    plt.savefig('best.png')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path of the log of experiments', default='./exp_result')
    parser.add_argument('--parameter', help='experiment variable', default='decay_step', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    result = filter_para(args.path, args.parameter)
    draw(args.path, result, args.parameter)
    plot_best(args.path)

