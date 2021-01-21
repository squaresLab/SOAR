import scrapy
import json

from typing import Dict


class APIItem(scrapy.Item):
    # TODO: extract more information from the code (e.g. function name, arguments, return types)
    item_id = scrapy.Field()
    item_type = scrapy.Field()  # this could be 'class', 'method' or 'function'
    code = scrapy.Field()
    # TODO: use separate fields for different part of the description
    summary = scrapy.Field()

    # Aidan's new fields
    description = scrapy.Field()
    parameters = scrapy.Field()
    returns = scrapy.Field()
    return_type = scrapy.Field()
    example = scrapy.Field()
    library = scrapy.Field()
    shape = scrapy.Field()
    signature = scrapy.Field()


def process_code_info(code: str) -> dict:
    result = dict()

    result['name'] = code.split('(')[0]

    # resolve the parameters
    result['parameters'] = []
    try:
        parameters_fields = code.split('(')[1][:-1].split(',')
    except:
        return result
    if len(parameters_fields) == 1 and parameters_fields[0] == '':
        return result

    for parameter_str in parameters_fields:
        parameter = dict()

        parameter['name'] = parameter_str.split('=')[0]
        parameter['is_optional'] = '=' in parameter_str
        parameter['type'] = ''

        if parameter['is_optional']:
            default_value = parameter_str.split('=')[1]
            parameter['default_value'] = default_value
            # TODO: figure out how to get the type in more ways
            # get some type information from default values
            if default_value.isdecimal():
                parameter['type'] = 'int'
            elif default_value.isnumeric():
                parameter['type'] = 'float'
            elif default_value[0] == default_value[-1] == "'":
                parameter['type'] = 'string'
            elif default_value.lower() == 'true' or default_value.lower() == 'false':
                parameter['type'] = 'bool'

        result['parameters'].append(parameter)

    return result


def nice_dump(filename, json_list):
    with open(filename, 'w') as fp:
        fp.write(
            '[' +
            ',\n'.join(json.dumps(i) for i in json_list) +
            ']\n')


if __name__ == '__main__':
    # print(get_keywords_from_code('Word2Vec'))
    # print(get_keywords_from_code('torch.distributions.distribution.Distribution.arg_constraints'))
    # print(get_keywords_from_code('torch.distributed.get_rank(group= & lt;objectobject & gt;)'))
    # print(get_keywords_from_code('torch.distributed.rpc.init_rpc(name, backend=BackendType.PROCESS_GROUP, rank=-1, world_size=None, rpc_backend_options=None)¶'))
    pass
