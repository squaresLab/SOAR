import re
import os, sys
import pandas
from commons.library_api import *
import json
import types


def freeze(d):
    if isinstance(d, list):
        return tuple(freeze(value) for value in d)
    elif isinstance(d, tuple):
        return tuple(freeze(value) for value in d)
    elif isinstance(d, int):
        return d
    elif isinstance(d, float):
        return d
    elif isinstance(d, str):
        return d
    else:
        return 0


def extract_arguments(api_name, line):
    idx = line.find(api_name + '(')
    if idx == -1:
        return False
    line = line[idx + len(api_name) + 1:]
    arguments = []
    argument = ''
    stack = 0
    if line[0] == ')':
        return False
    for c in line:
        if stack == 0:
            if c == ',':
                arguments += [argument.strip()]
                argument = ''
            elif c == ')':
                return arguments + [argument.strip()]
            elif c == '(':
                stack += 1
                argument += c
            else:
                argument += c
        elif stack >= 1:
            if c == ')':
                argument += c
                stack -= 1
            elif c == '(':
                stack += 1
                argument += c
            else:
                argument += c
    return False


def maybe_eval(value):
    try:
        val = eval(value)
        val = freeze(val)
        return val
    except:
        return None


path = "../github_crawler/tf_pyfiles"
file_names = list(map(lambda x: path + '/' + x, os.listdir(path)))
csv_files = {}
for file_name in file_names:
    csv_files[file_name] = pandas.read_csv(file_name, encoding='utf-8')
    try:
        csv_files[file_name]['code'] = csv_files[file_name]['code'].map(lambda x: eval(x).decode('utf-8'))
    except UnicodeDecodeError as e:
        print(e)


lib = load_library('tf')
lib.apis = list(filter(lambda x: x.id.find('tf.keras') != -1, lib.apis))

arguments = {}
for api in lib.apis:
    arguments[api.name] = {}
    ordering = {}
    count = 0
    for arg in api.arguments:
        arguments[api.name][arg.name] = set()
        ordering[count] = arg.name
        count += 1

    arguments_api = []
    for file_name in csv_files:
        for line in csv_files[file_name]['code']:
            match = extract_arguments(api.name, line)
            if match:
                arguments_api += [match]

    for arg_set in arguments_api:
        for i in range(len(arg_set)):
            if arg_set[i].find('=') != -1:
                string = arg_set[i].split('=')
                if string[0].strip() in arguments[api.name] and len(string) > 1:
                    arguments[api.name][string[0].strip()].add(maybe_eval(string[1].strip()))
            elif i in ordering and ordering[i] in arguments[api.name]:
                try:
                    arguments[api.name][ordering[i]].add(maybe_eval(arg_set[i]))
                except:
                    print('hmmm')

    for i in range(len(api.arguments)):
        arguments[api.name][ordering[i]] = list(arguments[api.name][ordering[i]])

with open('./tf_args.json', 'w+') as f:
    json.dump(arguments, f)
