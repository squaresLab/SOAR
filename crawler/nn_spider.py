import re
import json
import os
from scrapy.spiders import Rule, CrawlSpider
from scrapy.crawler import Crawler, CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from w3lib.html import remove_tags
from crawler.utils import APIItem, nice_dump, process_code_info


class TorchSpider(CrawlSpider):
    name = "torch"
    version = "1.4.0"
    allowed_domains = ['pytorch.org']
    start_urls = [f'https://pytorch.org/docs/master/nn.html']
    split_def = re.compile(r'^([\w\.]+)\(([\w\,\s=\*\.]*)\)')

    rules = (
        Rule(LinkExtractor(
            allow='master/generated'),
            callback='parse_api',
            follow=False,
        ),
    )

    def parse_item(self, item, item_type, selector, response):
        # Initialize local variables
        item_id = 'NA'
        code = 'NA'
        description = 'NA'
        returns = 'NA'
        return_type = 'NA'
        shape = ''

        raw_code = selector.css('dt').get()
        try:
            id_css = selector.css('dt')
            item_id = id_css.attrib['id']
        except:
            try:
                item_id = selector.css('dt')[1].attrib['id']

            except:
                return

        # if 'torch.nn.Conv2d' not in item_id:
        #     return

        code = remove_tags(raw_code).replace('\n', '').replace(' ', '').replace('[source]', '').replace('¶', '')
        description = remove_tags(selector.css('dd').get()).replace('[source]', ' ').replace('\n', ' ') \
            .replace('¶', '').replace('\t', ' ')

        # Parameters always appears under a "list" tag,
        # so we scrape all lists and record only when "Parameters" keyword appears
        list_of_items = selector.css('dl.field-list')
        is_param_list = list_of_items.xpath('//dt[contains(text(), "Parameters")]').get()

        parameters = []
        if is_param_list:
            for p in list_of_items.css('li').getall():
                parameters.append(remove_tags(p).replace('\n', '').replace('\u2013', ':'))
            if not parameters:
                for p in list_of_items.css('dd').getall():
                    parameters.append(remove_tags(p).replace('\n', '').replace('\u2013', ':'))

        returns = selector.xpath('//dt[contains(text(), "Returns")]/following-sibling::dd[1]')
        if returns:
            returns = remove_tags(returns.get()).replace('\n', '')

        example_x = selector.xpath('//p[contains(text(), "Example:")]/following-sibling::div[1]').getall()

        example = 'NA'
        if example_x:
            for x in example_x:
                example = remove_tags(x)
                if code in example or item_id in example:
                    example = example.replace('&gt;', '')
                    break

        dls = selector.css('dl')
        for dl in dls:
            if dl.attrib.__len__() == 0:
                is_shape = dl.xpath('//dt[contains(text(), "Shape:")]').get()
                if is_shape:
                    shape = dl.css('dd').get()
                    shape = remove_tags(shape)
                    break

        item['item_id'] = item_id
        item['item_type'] = item_type
        item['code'] = code
        item['description'] = description
        item['parameters'] = parameters
        item['returns'] = returns
        item['example'] = example
        item['shape'] = shape

    def parse_api(self, response):

        self.logger.info(f'Scraping {response.url}')
        # dealing with functions (methods without too much information)
        fselectors = response.css('dl.function')
        if fselectors:
            for fselector in fselectors:
                dt = fselector.css('dt')
                item = APIItem()
                item['library'] = 'torch'
                try:
                    self.parse_item(item, 'function', fselector, response)
                    yield item
                except:
                    try:
                        bad_id = fselector.css('dt').attrib['id']
                        print('######################### BAD FUNCTION: ' + bad_id + ' ######################')
                    except:
                        pass

        item = APIItem()
        # dealing with methods (with more information on the parameters, etc)
        mselectors = response.css('dl.method')
        if mselectors:
            for mselector in mselectors:
                item = APIItem()
                item['library'] = 'torch'
                try:
                    self.parse_item(item, 'method', mselector, response)
                    yield item
                except:
                    try:
                        bad_id = mselector.css('dt').attrib['id']
                        print('######################### BAD METHOD: ' + bad_id + ' ######################')
                    except:
                        pass

        # dealing with classes (could be an NN layer or some other class)
        cselectors = response.css('dl.class')
        # if 'torch.nn' in response.url:
        #     print(response.url)
        if cselectors:
            for cselector in cselectors:
                item = APIItem()
                item['library'] = 'torch'
                dt = cselector.css('dt')
                aselectors = cselector.css('dl.attribute')
                item_id = cselector.css('dt').attrib['id']

                if 'nn.Conv2d' in item_id:
                    print('hiiiii')
                length_dt = len(dt)
                try:
                    self.parse_item(item, 'class', cselector, response)
                    yield item
                except:
                    try:
                        bad_id = cselector.css('dt').attrib['id']
                        print('######################### BAD CLASS: ' + bad_id + ' ######################')
                    except:
                        pass

                for aselector in aselectors:
                    try:
                        self.parse_item(item, 'attribute', aselector, response)
                        yield item
                    except:
                        try:
                            bad_id = aselector.css('dt').attrib['id']
                            print('######################### BAD ATTRIBUTE: ' + bad_id + ' ######################')
                        except:
                            pass


def code_to_params(code_string):
    # all type of brackets
    square_ctr = 0
    code_params = code_string.split('(', 1)[1].split(')')[0]
    split_code = list(code_params)
    for i, char in enumerate(split_code):
        if char == '[':
            square_ctr += 1
        elif char == ']':
            square_ctr -= 1
        elif char == ',':
            if square_ctr != 0:
                # if not 0 then we are inside a bracket, replace with ... instead
                split_code[i] = '{.!.}'
    s = ''
    return_str = s.join(split_code)
    return_list = return_str.split(',')
    for i, char in enumerate(return_list):
        char = char.replace('{.!.}', ',')
        return_list[i] = char

    return return_list


def preprocess_torch_data(raw_data_file):
    # load the raw data
    data = None
    with open(raw_data_file) as f:
        data = json.load(f)
    processed_data = []

    for item in data:

        processed_item = dict()

        try:
            processed_item['id'] = item['item_id']
            if 'torch.nn.Conv2d' in item['item_id']:
                print('here')
        except:
            continue
        processed_item['type'] = item['item_type']

        raw_code = item['code']
        code = item['item_id'] + raw_code[raw_code.find('('):raw_code.find(')') + 1]
        processed_item['code'] = code

        # extract the summary
        description = item['description']

        summary = description.split('. ')[0]
        processed_item['example'] = item['example']

        if 'Example:' in item['description']:
            example = item['description'].split('Example:')[1]
            example = example.replace('&gt;', '')
            processed_item['example'] = example

        processed_item['summary'] = summary
        processed_item['returns'] = item['returns']
        processed_item['shape'] = item['shape']

        code = processed_item['code']

        processed_item['code-info'] = dict()

        processed_item['code-info']['name'] = code.split('(')[0]

        processed_item['code-info']['parameters'] = []

        result = dict()

        result['name'] = code.split('(')[0]

        ## Using only signature

        if '(' in code and ')' in code:
            param_dict = code_to_params(code)
            # parameters_fields = code.split('(')[1][:-1].split(',')
            if param_dict:
                # for arg in arg_json:
                for param in param_dict:

                    arg = dict()
                    param_split = param.split(':')
                    if len(param_split) > 1:
                        param_name = param_split[0]
                        # if param_name in arg['name']:
                        arg['name'] = param_split[0]
                        arg['type'] = param_split[1]

                        if '=' in param_split[1]:
                            type_def = param_split[1].split('=')
                            arg['type'] = type_def[0]
                            arg['default_value'] = type_def[1]

                        arg['is_optional'] = False
                        arg['description'] = ''
                        for param in item['parameters']:
                            des_list = param.split(' : ')
                            des_name = des_list[0].split('(')[0].replace(' ', '')

                            if arg['name'] == des_name:
                                # param_bracket = param[param.find("(") + 1:param.find(")")].split(',')
                                if 'optional' in param:
                                    arg['is_optional'] = True

                                if len(des_list) > 1:
                                    arg['description'] = des_list[1]
                                try:
                                    des_words = des_list[1].split(' ')
                                    for idx, word in enumerate(des_words):
                                        if 'Default:' in word:
                                            arg['default_value'] = des_words[idx + 1].replace('.', '')
                                            # print(arg['default_value'])
                                except:
                                    continue

                        processed_item['code-info']['parameters'].append(arg)

        processed_data.append(processed_item)

    preprocessed_json_file_name = 'preprocessed_' + raw_data_file
    if os.path.exists(preprocessed_json_file_name):
        os.remove(preprocessed_json_file_name)

    nice_dump(preprocessed_json_file_name, processed_data)


if __name__ == '__main__':
    json_file_name = 'nn_docs.json'

    # if os.path.exists(json_file_name):
    #     os.remove(json_file_name)
    #
    # spider = TorchSpider()
    # process = CrawlerProcess({
    #     'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    #     'FEED_FORMAT': 'json',
    #     'FEED_URI': json_file_name
    # })
    #
    # process.crawl(TorchSpider)
    # process.start()
    # process.join()
    # print("crawling completes, starts preprocessing...")

    preprocess_torch_data(json_file_name)

    with open('C:/projects/api-representation-learning/crawler/preprocessed_nn_docs.json') as f:
        nn_json = json.load(f)
    with open('C:/projects/api-representation-learning/crawler/preprocessed_torch_docs.json') as f:
        torch_json = json.load(f)

    processed_data = []

    for idx, torch in enumerate(torch_json):
        for nn in nn_json:
            if nn['id'] == torch['id']:
                torch_json[idx] = nn
        processed_data.append(torch_json[idx])

    nice_dump('C:/projects/api-representation-learning/crawler/preprocessed_torch_docs.json', processed_data)



