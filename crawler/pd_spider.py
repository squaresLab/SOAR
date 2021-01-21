import re
import json
import os
from scrapy.spiders import Rule, CrawlSpider
from scrapy.crawler import Crawler, CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from w3lib.html import remove_tags
from crawler.utils import APIItem, nice_dump, process_code_info


class PdSpider(CrawlSpider):
    name = "pandas"
    version = "1.4.0"
    allowed_domains = ['pandas.pydata.org']
    start_urls = [f'https://pandas.pydata.org/docs/reference/index.html']
    split_def = re.compile(r'^([\w\.]+)\(([\w\,\s=\*\.]*)\)')

    rules = (
        Rule(LinkExtractor(
            allow=re.compile(r'.+\.html')),
            callback='parse_api',
            follow=True),
    )

    def parse_item(self, item, item_type, selector):
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

        # if 'torch.nn.MaxPool2d' in item_id:
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

        # returns = selector.xpath('//dt[contains(text(), "Returns")]/following-sibling::dd[1]')
        # if returns:
        #     returns = remove_tags(returns.get()).replace('\n', '')

        example_x = selector.xpath('//p[contains(text(), "Example")]/following-sibling::div[1]').getall()

        example = 'NA'
        if example_x:
            for x in example_x:
                example = remove_tags(x)
                if code in example or item_id in example:
                    example = example.replace('&gt;', '')
                    break

        item['item_id'] = item_id
        # item['item_type'] = item_type
        item['code'] = code
        item['description'] = description
        item['parameters'] = parameters
        # item['returns'] = returns
        item['example'] = example
        # item['shape'] = shape

    def parse_api(self, response):

        # test specific web
        ##############
        # if '.nn' not in response.url:
        #     return
        ##############

        self.logger.info(f'Scraping {response.url}')
        # dealing with functions (methods without too much information)
        fselectors = response.css('dl')
        if fselectors:
            for fselector in fselectors:
                dt = fselector.css('dt')
                item = APIItem()
                item['library'] = 'pandas'
                try:
                    self.parse_item(item, 'function', fselector)
                    yield item
                except:
                    try:
                        bad_id = fselector.css('dt').attrib['id']
                        print('######################### BAD FUNCTION: ' + bad_id + ' ######################')
                    except:
                        pass


# class InnerSpider(CrawlSpider):


def preprocess_pd_data(raw_data_file):
    # load the raw data
    data = None
    with open(raw_data_file) as f:
        data = json.load(f)

    processed_data = []
    item_tracker = []

    for item in data:
        # TODO: find better ways to exclude non-functions
        # if '(' not in item['code']:
        #     continue

        processed_item = dict()

        try:
            processed_item['id'] = item['item_id']
        except:
            continue
        if 'pandas.DataFrame' not in item['item_id']:
            continue
        if item['item_id'] not in item_tracker:
            item_tracker.append(item['item_id'])
        else:
            continue

        # processed_item['type'] = item['item_type']
        # unify the notation for the code
        raw_code = item['code']
        code = item['item_id'] + raw_code[raw_code.find('('):raw_code.find(')') + 1]
        processed_item['code'] = code

        # extract the summary
        description = item['description']
        if 'Parameters' in description:
            summary = description.split('Parameters')[0]
        else:
            summary = description.split('.')[0]

        summary = description.split('. ')[0]
        processed_item['example'] = item['example']

        if 'Example:' in item['description']:
            example = item['description'].split('Example:')[1]
            example = example.replace('&gt;', '')
            processed_item['example'] = example

        processed_item['summary'] = summary
        # processed_item['returns'] = item['returns']
        # processed_item['shape'] = item['shape']
        # if processed_item['type'] != 'class' and processed_item['type'] != 'attribute':

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

            if (start_mark in description and end_mark in description):
                matching_result[arg_name] = '(' + description.split(start_mark)[1].split(end_mark)[0]

            else:
                if item['parameters']:
                    for item_param in item['parameters']:
                        item_param = item_param.replace('python:', '')
                        if ':' in item_param and arg_name in item_param:
                            item_param_name = item_param.split(':')[0].replace(' ', '')
                            if '(' in item_param_name:
                                item_param_name = item_param_name.split('(')[0]
                            if arg_name == item_param_name:
                                param_description = item_param.split(':')[1].replace(')', '')
                                matching_result[arg_name] = param_description

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

            if arg['type'] == 'others':
                if item['parameters']:
                    for item_param in item['parameters']:
                        if arg['name'] in item_param:
                            arg['type'] = item_param[item_param.find("(") + 1:item_param.find(")")]
                            arg['type'] = arg['type'].replace('python:', '')
                            break

        processed_data.append(processed_item)

    preprocessed_json_file_name = 'preprocessed_' + raw_data_file
    if os.path.exists(preprocessed_json_file_name):
        os.remove(preprocessed_json_file_name)

    nice_dump(preprocessed_json_file_name, processed_data)


if __name__ == '__main__':
    json_file_name = 'pd_docs.json'

    # if os.path.exists(json_file_name):
    #     os.remove(json_file_name)
    #
    # spider = PdSpider()
    # process = CrawlerProcess({
    #     'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    #     'FEED_FORMAT': 'json',
    #     'FEED_URI': json_file_name
    # })
    #
    # process.crawl(PdSpider)
    # process.start()
    # process.join()
    # print("crawling completes, starts preprocessing...")

    preprocess_pd_data(json_file_name)

