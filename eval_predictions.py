"""
Применение:
    python3 eval_predictions.py predictions_file target_file
    содержание predictions_file и target_file:
    
    ...
    path/to/image/imagename\\tgrountruthOrPrediction
    ...
    
    флаги:
    --long_len -- длина, начиная с которой слово считается длинным (default=10)
    --mode -- full или general
    --punctuations partial -- '.-, full -- все, empty -- без пунктационных или свой набор знаков препинания в словаре
"""



import argparse
import copy
import Levenshtein
import os
import pathlib
import sys

def get_dict(filename):
    with open(filename) as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        l = line.strip('\n')
        if len(l) == 0:
            continue
        img_path, label = l.split('\t')[:2]
        word_dict[pathlib.PurePath(img_path).name] = label
    return word_dict

def detailed_stat(predictions_file, target_file, alphabet, long_word_min_len=10):
    pred_dict = get_dict(predictions_file)
    target_dict = get_dict(target_file)

    cases = ('LC + UC, all words', 'LC + UC, long words', 'MC, all words', 'MC, long words')
    matrix1 = {'del': {char: 0 for char in alphabet}, 'ins': {char: 0 for char in alphabet}}
    matrix1 = dict(**matrix1, **{sym: {char: 0 for char in alphabet} for sym in alphabet})
    matrix2 = {'del': {char: 0 for char in alphabet[:-26]}, 'ins': {char: 0 for char in alphabet[:-26]}}
    matrix2 = dict(**matrix2, **{sym: {char: 0 for char in alphabet[:-26]} for sym in alphabet[:-26]})
    matrices = {case: copy.deepcopy(matrix1) if case[0] == 'L' else copy.deepcopy(matrix2) for case in cases}

    for name, target in target_dict.items():
        pred = pred_dict[name]
        pred_target = {'LC + UC': (pred, target), 'MC': (pred.lower(), target.lower())}

        is_long = (len(target) >= long_word_min_len)

        for case, (pred, target) in pred_target.items():
            ops = Levenshtein.editops(pred, target)
            matrices_names = [case + ', all words',]
            if is_long:
                matrices_names.append(case + ', long words')
            for matrix_name in matrices_names:
                for op in ops:
                    if op[0] == 'insert':
                        matrices[matrix_name]['ins'][target[op[2]]] += 1
                    elif op[0] == 'delete':
                        matrices[matrix_name]['del'][pred[op[1]]] += 1
                    elif op[0] == 'replace':
                        matrices[matrix_name][pred[op[1]]][target[op[2]]] += 1

    for case in cases:
        print(case + ':')
        matrix = matrices[case]
        first = '\t' + '\t'.join(matrix['ins'].keys())
        print(first)
        for key in matrix.keys():
            row = matrix[key]
            line = key + '\t'
            for sym in matrix['ins'].keys():
                line += '{}\t'.format(row[sym])
            print(line)
        print()
        print()

def general_stat(predictions_file, target_file, long_word_min_len=10):
    pred_dict = get_dict(predictions_file)
    target_dict = get_dict(target_file)

    cases = ('LC + UC, all words', 'LC + UC, long words', 'MC, all words', 'MC, long words')
    correct_num = {case: 0 for case in cases}
    edit_dis = {case: 0.0 for case in cases}
    norm_edit_dis = {case: 0.0 for case in cases}
    long_num = 0
    all_num = 0
    long_char = 0
    all_char = 0

    for name, target in target_dict.items():
        pred = pred_dict[name]
        pred_target = {'LC + UC': (pred, target), 'MC': (pred.lower(), target.lower())}

        is_long = (len(target) >= long_word_min_len)

        for case, (pred, target) in pred_target.items():
            cur_edit_dis = Levenshtein.distance(pred, target)
            is_correct = int(pred == target)

            correct_num[case + ', all words'] += is_correct
            edit_dis[case + ', all words'] += cur_edit_dis
            norm_edit_dis[case + ', all words'] += cur_edit_dis / len(target)

            if is_long:
                correct_num[case + ', long words'] += is_correct
                edit_dis[case + ', long words'] += cur_edit_dis
                norm_edit_dis[case + ', long words'] += cur_edit_dis / len(target)

        all_char += len(target)
        all_num += 1

        if is_long:
            long_char += len(target)
            long_num += 1

    acc = {'Accuracy ({})'.format(case): 1.0 * correct_num / max(all_num if (case[-9:-6] == 'all') else long_num, 1) 
            for case, correct_num in correct_num.items()}
    norm_edit_dis = {'Normalized edit distance 1 ({})'.format(case): dis / max(all_num if (case[-9:-6] == 'all') else long_num, 1) 
            for case, dis in norm_edit_dis.items()}
    edit_dis = {'Normalized edit distance 2 ({})'.format(case): dis / max(all_char if (case[-9:-6] == 'all') else long_char, 1) 
            for case, dis in edit_dis.items()}
    quality_dict = dict(**acc, **norm_edit_dis, **edit_dis)

    for key, value in quality_dict.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--long_len', default=10, help='Minimum length of a long word', type=int)
    parser.add_argument('-m', '--mode', default='general', help='general or full')
    parser.add_argument('-p', '--punctuations', default='partial', help="full or partial '.- or empty or custom punctuations")
    parser.add_argument('predictions_file')
    parser.add_argument('target_file')
    args = parser.parse_args()
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    partial_punc = "'.-"
    full_punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    if args.punctuations == 'partial':
        alphabet = partial_punc + alphabet
    elif args.punctuations == 'full':
        alphabet = full_punc + alphabet
    elif args.punctuations == 'empty':
        alphabet = '' + alphabet
    else:
        alphabet = args.punctuations + alphabet

    if args.mode == 'full':
        detailed_stat(args.predictions_file, args.target_file, alphabet, args.long_len)
    else:
        general_stat(args.predictions_file, args.target_file, args.long_len)

