import os
import sys
import json
# import codecs

# turn on when debug
debug = False


infer_arguments = {
    # beam num for beam search
    'beam': '5',
    # remove bpe
    'remove-bpe': 'True',
}
file_names = ['mt02_u8.', 'mt03_u8.', 'mt04_u8.', 'mt05_u8.', 'mt06_u8.', 'mt08_u8.']
valid_pref = ['.low0', '.low1', '.low2', '.low3']


# Group 1
group_1 = {
    'infer_code': 'fairseq-interactive /home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/bin',
    'valid_code': 'perl /home2/zhangzhuocheng/lab/torch15/fairseq_z/fairseq/scripts/multi-bleu.perl',
    # infer args
    'infer_args': infer_arguments,
    'files': file_names,
    'input_path': '/home2/zhangzhuocheng/lab/translation/datasets/zh_en/std/source/',
    'valid_pref': valid_pref,
    'source_lang': '30kbpe.zh',
    'target_lang': 'en',
    'print_args': 'False',
    # average epoch or update
    'average': None,
    # path to model
    'path': None,
}


# Group List
arg_group = {
    'zh_en_1': group_1,
    # 'zh_en_2': group_2,
}


def parser():
    """
    Parse sys.argv into dictionary

    Returns:
        args: dictionary
    """
    sys_args = sys.argv
    args = {}
    key = None
    for arg in sys_args:
        if(key is not None):
            # key = key.replace('-', '_')
            args[key] = arg
            key = None
            continue
        if arg[:2] == '--':
            key = arg[2:]
        else:
            key = None
    return args


def update(args):
    group = args['group']
    new_args = arg_group[group]
    global debug
    # update args for inference
    for name in args:
        if name in new_args['infer_args']:
            new_args['infer_args'][name] = args[name]
    # update args for global
    for name in args:
        if name in new_args:
            new_args[name] = args[name]
    if('debug' in args):
        debug = eval(args['debug'])
    # check args
    for name in new_args['infer_args']:
        assert new_args['infer_args'][name] is not None
    if(eval(new_args['print_args'])):
        print(json.dumps(new_args, indent=4))
    return new_args


def average(args, debug):
    print('Averaging' + '.' * 50)
    exe_code = 'python /home2/zhangzhuocheng/lab/torch15/fairseq_z/fairseq/scripts/average_checkpoints.py'
    exe_code += ' --inputs '
    exe_code += args['path']
    assert (args['average'] == 'epoch') or (args['average'] == 'update')
    if(args['average'] == 'epoch'):
        exe_code += ' --num-epoch-checkpoints %d' % 5
        exe_code += ' --output '
        exe_code += args['path'] + '/average-epoch.pt'
        args['infer_args']['path'] = args['path'] + '/average-epoch.pt'
    else:
        exe_code += ' --num-update-checkpoints %d' % 5
        exe_code += ' --output '
        exe_code += args['path'] + '/average-update.pt'
        args['infer_args']['path'] = args['path'] + '/average-update.pt'
    if(debug):
        print(exe_code, end='\n\n')
    else:
        os.system(exe_code)
    return args


def infer(args, debug):
    print('Infering' + '.' * 50)
    exe_code = args['infer_code']
    infer_args = args['infer_args']
    if(not os.path.exists(args['path'] + '/infer')):
        os.mkdir(args['path'] + '/infer')
    for arg_name in infer_args:
        exe_code += ' --' + arg_name
        exe_code += ' ' + infer_args[arg_name]
    for file_name in args['files']:
        exec_code = exe_code + ' --input %s' % args['input_path'] + file_name + args['source_lang']
        exec_code = exec_code + ' --raw-output %s/infer/' % args['path'] + file_name + args['average'] + '.out'
        if(not debug):
            os.system(exec_code)
        else:
            print(exec_code)
    print()
    return args


def valid(args, debug):
    print('Validing' + '.' * 50)
    exe_code = args['valid_code']
    for file_name in args['files']:
        file_path_base = args['input_path'] + file_name + args['target_lang']
        output_path = args['path'] + '/infer/' + file_name + args['average'] + '.out'
        file_paths = [file_path_base + pref for pref in args['valid_pref']]
        exec_code = exe_code + ' ' + ' '.join(file_paths)
        exec_code += ' < %s' % output_path
        exec_code += ' | tee %s/infer/%s_bleu.%s.txt' % (args['path'], file_name[:-1], args['average'])
        if(not debug):
            os.system(exec_code)
        else:
            print(exec_code)
    print()
    return args


if __name__ == "__main__":
    args = parser()
    args = update(args)
    args = average(args, debug)
    args = infer(args, debug)
    args = valid(args, debug)
