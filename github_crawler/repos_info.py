import pandas as pd
from github import Github
import datetime
import time

tf_repos = []
torch_repos = []

access_token_a = '6eadab012828f3b3650c02bfa23618e8860a28b2'
access_token_b = '3671d57dd2fe024e6b6b1a233890f9f832eb2292'
access_token_c = '8f2e0a31ec4bec6614356b354183e69782d6011a'
access_token_d = '79f0a1bbbf85d29885454a597a70458222aa3129'
ga = Github(access_token_a)
gb = Github(access_token_b)
gc = Github(access_token_c)
gd = Github(access_token_d)


def repos_info(g, rname):
    print('cloning repos: ' + rname)
    repo = g.get_repo(str(rname))
    stars = repo.stargazers_count
    forks = repo.forks_count
    months = (datetime.datetime.now().year - repo.created_at.year) * 12

    torch_v = 'none'
    tf_v = 'none'
    np_v = 'none'

    try:
        env = repo.get_contents("environment.yml")
        contents = str(env.decoded_content).split('\\n')

        for content in contents:
            if 'torch' in content:
                torch_v = content
            if 'tensorflow' in content:
                tf_v = content
            if 'numpy' in content:
                np_v = content
    except:
        pass

    try:
        req = repo.get_contents("requirements.txt")
        contents = str(req.decoded_content).split('\\')
        for content in contents:
            if 'torch' in content and '=' in content:
                torch_v = content
            if 'tensorflow' in content and '=' in content:
                tf_v = content
            if 'numpy' in content and '=' in content:
                np_v = content
    except:
        pass

    np_v = np_v.replace("b\'", "")
    return_info = [rname, stars, forks, months, tf_v, torch_v, np_v]
    return return_info


def api_info(df_api):
    count = 0
    resTable = [[]]
    for rname in df_api[df_api.columns[0]]:

        if count == 0:
            print('######################## USING API A ################################')
            try:
                return_info = repos_info(ga, rname)
                resTable.append(return_info)
            except:
                count = 1
        elif count == 1:
            print('######################## USING API B ################################')
            try:
                return_info = repos_info(gb, rname)
                resTable.append(return_info)
            except:
                count = 2
        elif count == 2:
            print('######################## USING API C ################################')
            try:
                return_info = repos_info(gc, rname)
                resTable.append(return_info)
            except:
                count = 3
        elif count == 3:
            print('######################## USING API D ################################')
            try:
                return_info = repos_info(gd, rname)
                resTable.append(return_info)
            except:
                print('sleeping for hour')
                time.sleep(3400)
                count = 0
    df_out = pd.DataFrame(resTable,
                          columns=['Name', 'Stars', 'Forks', 'Months', 'TF_version', 'Torch_version', 'Np_version'])
    return df_out


def repos_stats(g, rname, api):
    if 'tensorflow' in api:
        apiCall = 'tf'
    elif 'torch' in api:
        apiCall = 'torch'
    elif 'numpy' in api:
        apiCall = 'np'

    print('cloning repos: ' + rname)
    api_counter = 0
    repo = g.get_repo(str(rname))
    contents = repo.get_contents("")
    pycode_container = []
    while contents:
        try:
            file_content = contents.pop(0)
            fpath = file_content.path
        except:
            continue
        if file_content.type == "dir":
            contents.extend(repo.get_contents(fpath))
        else:
            if ".py" in fpath[-3:]:
                # Retrieve, decode, and split file contents by newline
                try:
                    code_contents = str(repo.get_contents(file_content.path).decoded_content)
                except:
                    continue

                nline_split = code_contents.split('\\n')
                # isAPI = False
                # for idx, line in enumerate(nline_split):
                #     if idx > 15:
                #         break
                #     if api in line:
                #         isAPI = True

                file_api_counter = 0
                # Loop through each line searching for api call
                for nline in nline_split:
                    if apiCall + '.' in nline:
                        file_api_counter += 1
                        api_counter += 1
                code_out = [file_content.path, file_api_counter, code_contents]
                pycode_container.append(code_out)

    dirName = 'C:\\Projects\\api-representation-learning\\github_crawler\\' + apiCall + '_pyfiles\\' + rname.replace(
        '/', '__')
    csvName = dirName + '.csv'
    df_out = pd.DataFrame(pycode_container, columns=['file_path', 'api_count', 'code'])
    if len(df_out) > 0:
        df_out.to_csv(csvName, index=False)


def api_stats(df_api, api):
    count = 0
    for idx, rname in enumerate(df_api['repos']):
        if idx < 915:
            continue

        if count == 0:
            print('######################## USING API A ################################')
            try:
                repos_stats(ga, rname, api)
            except:
                count = 1

        if count == 1:
            print('######################## USING API B ################################')
            try:
                repos_stats(gb, rname, api)
            except:
                count = 2
        if count == 2:
            print('######################## USING API C ################################')
            try:
                repos_stats(gc, rname, api)
            except:
                count = 3
        elif count == 3:
            print('######################## USING API D ################################')
            try:
                repos_stats(gd, rname, api)
            except:
                print('sleeping for hour')
                time.sleep(3400)
                count = 0


df_tf = pd.read_csv('C:\\Projects\\api-representation-learning\\github_crawler\\repos_tf.csv')
df_torch = pd.read_csv('C:\\Projects\\api-representation-learning\\github_crawler\\repos_torch.csv')
df_np = pd.read_csv('C:\\Projects\\api-representation-learning\\github_crawler\\repos_np.csv')

# TODO this is for basic repos info
# df_out_tf = api_info(df_tf)
# df_out_tf.to_csv('C:\\Projects\\api-representation-learning\\github_crawler\\repos_info_tf.csv', index=False)

# df_out_torch = api_info(df_torch)
# df_out_torch.to_csv('C:\\Projects\\api-representation-learning\\github_crawler\\repos_info_torch.csv', index=False)

# df_out_np = api_info(df_np)
# df_out_np.to_csv('C:\\Projects\\api-representation-learning\\github_crawler\\repos_info_np.csv', index=False)

# TODO this is for code clone
# api_stats(df_tf, 'tensorflow')
api_stats(df_np, 'numpy')
# api_stats(df_torch, 'torch')
