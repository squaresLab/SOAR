import re
import json
import os

from scrapy.spiders import Rule, CrawlSpider
from scrapy.crawler import Crawler, CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from w3lib.html import remove_tags

from crawler.utils import APIItem, nice_dump, process_code_info


class TfSpider(CrawlSpider):
    name = "tf"
    version = "2.1"
    allowed_domains = ['tensorflow.org']
    start_urls = [
        f'https://www.tensorflow.org/versions/r{version}/api_docs/python/tf'
    ]
    split_def = re.compile(r'^([\w\.]+)\((.*)\)$')

    rules = (
        Rule(LinkExtractor(
            allow=(re.compile(r'.+api_docs\/python\/tf')),
            restrict_css='.devsite-nav-title'),
            callback='parse_api', ),
    )

    # def parse_item(self, response):

    def parse_api(self, response):
        self.logger.info(f'Scraping {response.url}')
        function_header = response.css('.lang-python')
        if len(function_header) == 0:
            return

        ID = remove_tags(response.css('h1.devsite-page-title').get()).replace('\n', '').replace(' ', '')
        # if 'tf.keras.layers.GRU' in ID:
        #     print(ID)
        # else:
        #     return
        code = remove_tags(function_header.get()).replace('\n', '').replace(' ', '')

        split = self.split_def.match(code)
        if split is None:
            return

        # description = remove_tags(response.xpath('//img[contains(@src,"GitHub-Mark-32px.png")]/following::p[1]').get())
        description = remove_tags(response.css('div.devsite-article-body').get()).replace('\n', ' ').replace('\t', ' ')

        # Have to treat page differently if there exists an h3 methods tag
        methods = response.xpath('//h2[@id="methods"]/following::h3')
        parameters = []
        returns = 'NA'
        example = 'NA'
        # Before Methods logic:
        # If there's a methods tag, look for returns before methods, only if there is returns above methods,
        # we record that return. If there isn't, we record returns as none.
        # If there is no methods tag, record first returns.

        if methods:
            returns = response.xpath('//h2[@id="methods"]/preceding-sibling::h4[@id="returns"]')
            params = response.xpath('//h2[@id="methods"]/preceding-sibling::h4[@id="args"]')
            example_h4 = response.xpath('//h2[@id="methods"]/preceding-sibling::h4[@id="for_example"]')
            if not example_h4:
                example_h4 = response.xpath('//h2[@id="methods"]/preceding-sibling::h4[@id="example"]')
            if not example_h4:
                example_h4 = response.xpath('//h2[@id="methods"]/preceding-sibling::h4[@id="examples"]')

            # the if statements ensure that we only take fields if fields are before methods tag
            if returns:
                returns = remove_tags(response.xpath('//h4[@id="returns"]/following-sibling::p[1]').get()).replace('\n',
                                                                                                                   '')
            if params:
                parameters = []
                param_list = response.xpath('//h4[@id="args"]/following-sibling::ul[1]')
                for p in param_list.xpath('.//li'):
                    parameters.append(remove_tags(p.get()).replace('\n', ''))




            if example_h4:
                example = response.xpath('//h4[@id="for_example"]/following-sibling::pre[1]').get()
                if not example:
                    example = response.xpath('//h4[@id="example"]/following-sibling::pre[1]').get()
                if not example:
                    example = response.xpath('//h4[@id="examples"]/following-sibling::pre[1]').get()
            if example:
                example = remove_tags(example).replace("\n", "").replace("&gt", "")
            item_type = 'class'

        else:
            returns = response.xpath('//h4[@id="returns"]/following-sibling::p[1]').get()
            if returns:
                returns = remove_tags(returns).replace('\n', '')
            else:
                returns = response.xpath('//h4[@id="output_shape"]/following-sibling::p[1]').get()
                if returns:
                    returns = remove_tags(returns).replace('\n', '')

            parameters = []
            param_list = response.xpath('//h4[@id="args"]/following-sibling::ul[1]')
            if param_list:
                for p in param_list.xpath('.//li'):
                    parameters.append(remove_tags(p.get()).replace('\n', ''))
            else:
                param_list = response.xpath('//h4[@id="arguments"]/following-sibling::ul[1]')
                if param_list:
                    for p in param_list.xpath('.//li'):
                        parameters.append(remove_tags(p.get()).replace('\n', ''))

            example = response.xpath('//h4[@id="for_example"]/following-sibling::pre[1]')
            if not example:
                example = response.xpath('//h4[@id="example"]/following-sibling::pre[1]')
            if not example:
                example = response.xpath('//h4[@id="examples"]/following-sibling::pre[1]')
            if example:
                example = remove_tags(example.get()).replace("\n", "").replace("&gt", "")
            item_type = 'class'

        item = APIItem()
        item['item_id'] = ID
        item['item_type'] = item_type
        item['code'] = code
        item['description'] = description
        item['parameters'] = parameters
        item['returns'] = returns
        item['example'] = example
        yield item

        # Each element of methods is a h3 tag
        if methods:
            returns = 'NA'
            parameters = []
            example = 'NA'
            # if there is a h2 tag "methods" on the page, then we loop through each method (h3 tag)
            item = APIItem()
            ret_param_ids = []
            for idx, method in enumerate(methods):

                try:
                    method_name = str(method.attrib['id'])
                except:
                    continue

                # If code snippet under "example" contains method id, take
                example = method.xpath('//h4[@id="for_example"]/following-sibling::pre[1]')
                if not example:
                    example = method.xpath('//h4[@id="example"]/following-sibling::pre[1]')
                if not example:
                    example = method.xpath('//h4[@id="examples"]/following-sibling::pre[1]')
                if example:
                    example = remove_tags(example.get())
                if method_name not in example:
                    example = 'NA'

                try:
                    method_name_next = methods[idx + 1].attrib['id']
                except:
                    returns_temp = response.xpath(
                        '//h4[contains(text(),"Returns:")][last()]')
                    ret_id = returns_temp.attrib['id']
                    ret_xpath = 'h4[@id="' + ret_id + '"]'
                    if ret_id not in ret_param_ids:
                        returns = remove_tags(
                            method.xpath('//' + ret_xpath + '/following-sibling::p[1]').get().replace('\n', ''))

                    param = method.xpath(
                        '//h4[contains(text(),"Args:")][last()]')
                    param_id = param.attrib['id']
                    # param_list = response.xpath('//h4[@id="args"]/following-sibling::ul[1]')
                    param_xpath = '//h4[@id="' + param_id + '"]'
                    if param_id not in ret_param_ids:
                        param_list = response.xpath(param_xpath + '/following-sibling::ul[1]')
                        for p in param_list.xpath('.//li'):
                            parameters.append(remove_tags(p.get()).replace('\n', ''))

                if method_name_next:
                    method_name_next = str(method_name_next)
                    code_xpath = '//h3[@id="' + method_name + '"]/following-sibling::pre[1]'
                    code = remove_tags(response.xpath(code_xpath).get()).replace('\n', '').replace(' ', '')

                    # Note: can't just use following after method tag, because it might skip to next method's args
                    # But can use preceding before next method, because I use h3 "method" as root for each iteration,
                    # So skipping forward is impossible.
                    ret_id = 'returns'
                    ret = method.xpath(
                        '//h3[@id="' + method_name_next + '"]/preceding-sibling::h4[contains(text(),"Returns:")][1]')
                    if ret:
                        ret_id = ret.attrib['id']
                        ret_param_ids.append(ret_id)
                        ret_xpath = 'h4[@id="' + ret_id + '"]'
                        returns = remove_tags(
                            method.xpath('//' + ret_xpath + '/following-sibling::p[1]').get().replace('\n', ''))

                    else:
                        returns = 'None'
                        ret_p = method.xpath(
                            '//h3[@id="' + method_name_next + '"]/preceding-sibling::p[contains(text(),"Returns a")][1]')
                        if ret_p:
                            ret_p = remove_tags(ret_p.get())
                            returns = ret_p

                    param_id = 'args'
                    param = method.xpath(
                        '//h3[@id="' + method_name_next + '"]/preceding-sibling::h4[contains(text(),"Args:")][1]')
                    if param:
                        param_id = param.attrib['id']
                        ret_param_ids.append(param_id)
                        parameters = []
                        param_xpath = '//h4[@id="' + param_id + '"]'
                        param_list = method.xpath(param_xpath + '/following-sibling::ul[1]')
                        for p in param_list.xpath('.//li'):
                            parameters.append(remove_tags(p.get()).replace('\n', ''))
                    else:
                        parameters = []

                method_id = ID + ': ' + method_name
                example = example.replace("&gt", "")
                # filling item struct again
                item = APIItem()
                item['item_id'] = method_id
                item['item_type'] = 'method'
                item['code'] = code
                item['returns'] = returns
                item['parameters'] = parameters
                item['example'] = example
                yield item


def preprocess_tf_data(raw_data_file: str) -> None:
    # load the raw data
    data = None
    with open(raw_data_file) as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        processed_item = dict()
        if 'description' not in item:
            item['description'] = ''

        # do some cleaning for the description
        description = item['description'].replace('TensorFlow 1 version', '')
        description = description.replace('View source on GitHub', '')

        # extract the summary
        summary = description.split('View aliases')[0].split(item['item_id'])[0].strip()

        processed_item['id'] = item['item_id']
        if '\n\n\nMethods:\n\n' in item['description'] or '\n\n\nAttributes:\n\n\n' in item['description']:
            processed_item['type'] = 'class'
        else:
            processed_item['type'] = 'function'
        processed_item['type'] = item['item_type']

        processed_item['code'] = item['code']
        processed_item['summary'] = summary
        processed_item['example'] = item['example']

        if not item['example']:
            processed_item['example'] = 'NA'
        processed_item['returns'] = item['returns']

        processed_item['code-info'] = process_code_info(processed_item['code'])

        # add description to all the arguments
        arg_json: dict = processed_item['code-info']['parameters']
        arg_names = list(map(lambda arg: arg['name'], arg_json))
        matching_result = dict()
        for i in range(len(arg_names)):
            arg_name = arg_names[i]
            start_mark = '\n' + arg_name + ': '
            if i != len(arg_names) - 1:
                end_mark = '\n' + arg_names[i + 1] + ': '
            else:
                end_mark = '\n\n'

            if (start_mark in description and end_mark in description):
                matching_result[arg_name] = description.split(start_mark)[1].split(end_mark)[0]

            else:
                if item['parameters']:
                    for item_param in item['parameters']:
                        if ':' in item_param and arg_name in item_param:
                            item_param_name = item_param.split(':')[0].replace(' ', '')
                            if '(' in item_param_name:
                                item_param_name = item_param_name.split('(')[0]
                            if arg_name == item_param_name:
                                matching_result[arg_name] = item_param.split(':')[1].replace(')', '')

        for arg_dict in arg_json:
            name = arg_dict['name']
            if name in matching_result:
                arg_dict['description'] = matching_result[name]
            else:
                arg_dict['description'] = ''

        # augment the types of arguments with NL description
        for arg in arg_json:
            if arg['type'] == '':
                description_types = arg['description'][:50].lower()
                if 'tensor' in description_types:
                    arg['type'] = 'tensor'
                elif 'integer' in description_types:
                    arg['type'] = 'int'
                elif 'scalar' in description_types or 'float' in description_types:
                    arg['type'] = 'float'
                elif 'bool' in description_types:
                    arg['type'] = 'bool'
                elif 'str' in description_types or 'name' in description_types:
                    arg['type'] = 'string'
                else:
                    arg['type'] = 'others'

        processed_data.append(processed_item)

    preprocessed_json_file_name = 'preprocessed_' + raw_data_file
    if os.path.exists(preprocessed_json_file_name):
        os.remove(preprocessed_json_file_name)

    nice_dump(preprocessed_json_file_name, processed_data)


if __name__ == '__main__':
    json_file_name = 'tf_docs.json'

    # if os.path.exists(json_file_name):
    #     os.remove(json_file_name)
    #
    # spider = TfSpider()
    # process = CrawlerProcess({
    #     'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    #     'FEED_FORMAT': 'json',
    #     'FEED_URI': json_file_name
    # })
    #
    # process.crawl(TfSpider)
    # process.start()
    # process.join()

    print("crawling completes, starts preprocessing...")
    preprocess_tf_data(json_file_name)
