import re
import json
import os
from scrapy.spiders import Rule, CrawlSpider
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from w3lib.html import remove_tags
from crawler.utils import APIItem, nice_dump, process_code_info


class NumpySpider(CrawlSpider):
    name = "numpy"
    version = "1.18"
    allowed_domains = ['numpy.org']
    start_urls = [f'https://numpy.org/doc/{version}/genindex.html']
    split_def = re.compile(r'^([\w\.]+)\(([\w\,\s=\*\.]*)\)')

    rules = (
        Rule(LinkExtractor(
            allow=(re.compile(r'.+\.html')),
            restrict_xpaths='//li'),
            callback='parse_api', ),
    )

    def parse_api(self, response):
        # Initiating item struct
        item = APIItem()
        item['library'] = 'numpy'

        self.logger.info(f'Scraping {response.url}')

        item_id = 'None'
        code = 'None'
        description = 'None'
        returns = 'None'
        examples = []

        dt = response.css('dt')
        if dt:
            item_id = dt.attrib['id']
            code = remove_tags(dt.get())

        description = response.css('dd')
        if description:
            description = remove_tags(description.get())

        params_tr = response.xpath('//tr[contains(text(), "Parameters")]')
        parameters = []
        if params_tr:
            for p in params_tr.css('dd').getall():
                if "Parameters" not in p:
                    parameters.append(remove_tags(p).replace('\n', ''))
        if not params_tr:
            list_of_items = response.css('dl.field-list')
            is_param_list = list_of_items.xpath('//dt[contains(text(), "Parameters")]').get()
            if is_param_list:
                for p in list_of_items.css('dt').getall():
                    if "Parameters" not in p:
                        parameters.append(remove_tags(p))

        return_tr = response.xpath('//tr[contains(text(), "Returns")]')
        if return_tr:
            returns = return_tr.css('dd').get()
            if returns:
                returns = remove_tags(returns)
        if not return_tr:
            returns = response.xpath('//dt[.="Returns"]/following-sibling::dd[1]')
            if returns:
                returns = remove_tags(returns.get()).replace('\n', '')

        examples = []
        example_p = response.xpath('//p[contains(text(), "Examples")]/following::div').getall()
        if example_p:
            for e in example_p:
                example = remove_tags(e)
                if '&gt;&gt;&gt' in example:
                    examples.append(example.replace('&gt;&gt;&gt', ''))

        item['item_id'] = item_id
        item['code'] = code
        item['description'] = description
        item['parameters'] = parameters
        item['returns'] = returns
        item['examples'] = examples
        yield item


def preprocess_torch_data(raw_data_file):
    # load the raw data
    data = None
    with open(raw_data_file) as f:
        data = json.load(f)

    processed_data = []

    for item in data:
        # TODO: find better ways to exclude non-functions
        if '(' not in item['code']:
            continue

        processed_item = dict()

        # unify the notation for the code
        raw_code = item['code']
        code = item['item_id'] + raw_code[raw_code.find('('):raw_code.find(')') + 1]
        processed_item['code'] = code

        # extract the summary
        description = item['description']
        # if 'Parameters' in description:
        #   summary = description.split('Parameters')[0]
        # else:
        #  summary = description.split('.')[0]

        summary = description.split('. ')[0]

        processed_item['item_id'] = item['item_id']
        processed_item['summary'] = summary
        processed_item['description'] = ''
        processed_item['example'] = item['examples']
        processed_item['returns'] = item['returns']
        processed_item['code-info'] = process_code_info(processed_item['code'])

        # add description to all the arguments
        arg_json: list = processed_item['code-info']['parameters']
        arg_names = list(map(lambda arg: arg['name'], arg_json))
        matching_result = dict()
        for i in range(len(arg_names)):
            arg_name = arg_names[i]
            start_mark = '\n' + arg_name + ' ('
            if i != len(arg_names) - 1:
                end_mark = '\n' + arg_names[i + 1] + ' ('
            else:
                end_mark = '\n\n'

            if not (start_mark in description and end_mark in description):
                continue
            matching_result[arg_name] = '(' + description.split(start_mark)[1].split(end_mark)[0]

        for arg_dict in arg_json:
            name = arg_dict['name']
            if name in matching_result:
                arg_dict['description'] = matching_result[name]
            else:
                arg_dict['description'] = ''

        # augment the types of arguments with NL description
        for arg in arg_json:
            if arg['type'] == '':
                # TODO: figure out why it fails sometimes
                try:
                    description_types = arg['description'].split('(')[1].split(')')[0]
                    if 'int' in description_types:
                        arg['type'] = 'int'
                    elif 'float' in description_types:
                        arg['type'] = 'float'
                    elif 'bool' in description_types:
                        arg['type'] = 'bool'
                    elif 'Tensor' in description_types:
                        arg['type'] = 'tensor'
                    elif 'string' in description_types:
                        arg['type'] = 'string'
                    else:
                        arg['type'] = 'others'
                except IndexError:
                    arg['type'] = 'others'

        processed_data.append(processed_item)

    preprocessed_json_file_name = 'preprocessed_' + raw_data_file
    if os.path.exists(preprocessed_json_file_name):
        os.remove(preprocessed_json_file_name)

    nice_dump(preprocessed_json_file_name, processed_data)


if __name__ == '__main__':
    json_file_name = 'numpy_docs.json'
    # '''
    if os.path.exists(json_file_name):
        os.remove(json_file_name)

    spider = NumpySpider()
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
        'FEED_FORMAT': 'json',
        'FEED_URI': json_file_name
    })

    process.crawl(NumpySpider)
    process.start()

    process.join()
    print("crawling completes, starts preprocessing...")
    # '''
    preprocess_torch_data(json_file_name)
