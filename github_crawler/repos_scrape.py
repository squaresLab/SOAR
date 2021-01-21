from github import Github
import time
import pandas as pd


access_token_a = '6eadab012828f3b3650c02bfa23618e8860a28b2'
access_token_b = '3671d57dd2fe024e6b6b1a233890f9f832eb2292'
g = Github(access_token_a)
counter = 0

# # Crawl tf
# tf_repos = []
# with open('C:\\Projects\\api-representation-learning\\github_crawler\\repos_tf.csv','w') as f:
#     # bigc = 0
#     tf = g.search_repositories(query='topic:tensorflow', sort='stars', order='desc')
#     for r in tf:
#         counter += 1
#         rname = r.full_name
#         tf_repos.append(rname)
#         print(rname)
#         if counter > 29:
#             time.sleep(60)
#             counter = 0
# df_tf = pd.DataFrame(tf_repos, columns=['repos'])
# tf_path = 'C:\\Projects\\api-representation-learning\\github_crawler\\repos_tf.csv'
# df_tf.to_csv(torch_path, index=False)
#
# # Crawl torch
# torch_repos = []
# torch = g.search_repositories(query='topic:pytorch', sort='stars', order='desc')
# for r in torch:
#     counter += 1
#     rname = r.full_name
#     torch_repos.append(rname)
#     print(rname)
#     if counter > 29:
#         time.sleep(60)
#         counter = 0
# df_torch = pd.DataFrame(torch_repos, columns=['repos'])
# torch_path = 'C:\\Projects\\api-representation-learning\\github_crawler\\repos_torch.csv'
# df_torch.to_csv(torch_path, index=False)

np_repos = []
numpy = g.search_repositories(query='topic:numpy', sort='stars', order='desc')
for r in numpy:
    counter += 1
    rname = r.full_name
    np_repos.append(rname)
    print(rname)
    if counter > 29:
        time.sleep(60)
        counter = 0
df_np = pd.DataFrame(np_repos, columns=['repos'])
np_path = 'C:\\Projects\\api-representation-learning\\github_crawler\\repos_np.csv'
df_np.to_csv(np_path, index=False)

# with open('C:\\Projects\\api-representation-learning\\github_crawler\\repos_tf.csv','w') as f:
#     writer = csv.writer(f, delimiter='\n')
#     writer.writerow(tf_repos)
#
# with open('C:\\Projects\\api-representation-learning\\github_crawler\\repos_torch.csv','w') as f:
#     writer = csv.writer(f, delimiter='\n')
#     writer.writerow(torch_repos)

