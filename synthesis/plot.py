import matplotlib.pyplot as plt
import numpy as np

from typing import List
from functools import reduce


def plot_packages(package_sizes: List[List[float]], large_group_names: List[str],
                  small_group_names: List[List[str]], package_vals: List[List[float]]=None, max_groups=10):
  
  assert len(small_group_names) == len(large_group_names)
  
  outer_vals = list(map(lambda x: sum(x), package_sizes))
  inner_vals = list(reduce(lambda x, y: x+y, package_sizes))
  
  # set color depending on if the color should mean something (i.e. package_vals)
  if package_vals is None:
    # TODO: plot the package size percentages without extra information
    cmap = plt.get_cmap('tab10')
    outer_colors = cmap(np.arange(len(outer_vals)))
    inner_colors = None
  else:
    # calculate outer color on weighted average of inner package_vals
    weighted_sum = map(lambda sizes, vals:
                       map(lambda size, val: size*val, sizes, vals),
                       package_sizes, package_vals)
    weighted_sum = list(map(lambda x, y: sum(x) / y, weighted_sum, outer_vals))
      
    cmap = plt.get_cmap('bwr')
    outer_colors = cmap(weighted_sum)
    inner_colors = cmap(list(reduce(lambda x,y: x+y, package_vals)))

    # add values to the labels
    combine_name_percentage_func = (lambda name, percentage, size:
                                    name+'[{:.0f}%]'.format(percentage*100) if size > 1.0 else '')
    large_group_names = list(map(combine_name_percentage_func, large_group_names, weighted_sum, outer_vals))
    small_group_names = list(map(lambda names, percentages, sizes:
                                 list(map(combine_name_percentage_func, names, percentages, sizes)),
                                 small_group_names, package_vals, package_sizes))
  
  # flatten the smaller group names
  small_group_names = list(reduce(lambda x, y: x+y, small_group_names))
  nested_pie_plot(outer_vals, inner_vals, large_group_names, small_group_names, outer_colors, inner_colors)


def nested_pie_plot(outer_vals: List[float], inner_vals: List[float],
                    outer_names, inner_names, outer_colors, inner_colors):
  fig, ax = plt.subplots(figsize=(20, 20))
  
  size = 0.3
  
  ax.pie(outer_vals, radius=1, labels=outer_names, colors=outer_colors,
         wedgeprops=dict(width=size, edgecolor='w'))
  
  ax.pie(inner_vals, radius=1 - size, labels=inner_names, labeldistance=0.7, colors=inner_colors,
         wedgeprops=dict(width=size, edgecolor='w'))
  
  ax.set(aspect='equal')
  # plt.show()
  
  plt.savefig('nested_pie_plot.png')
  
  
def pie_plot(vals: List[float], names: List[str], show_percentage=False, low_percentage_cutoff=0.5):
  # do some calculation of percentage before getting to generate the plot
  percentages = list(map(lambda x: x * 100.0 / sum(vals), vals))
  if show_percentage:
    names = list(map(lambda name, percent: name+'[{:.1f}%]'.format(percent), names, percentages))
  names = list(map(lambda name, percent: name if percent > low_percentage_cutoff else '', names, percentages))
  
  # now we generate the plot
  fig, ax = plt.subplots(figsize=(20, 10))
  
  size = 0.3

  cmap = plt.get_cmap('tab10')
  colors = cmap(np.arange(len(vals)) % 10)
  ax.pie(vals, radius=1, labels=names, colors=colors, wedgeprops=dict(width=size, edgecolor='w'))
  
  ax.set(aspect='equal')
  # plt.show()
  
  plt.savefig('pie_plot.png')


if __name__ == '__main__':
   package_size = [[2,5,6], [10,25,2], [1,20,5,7]]
   package_vals = [[0.1,0.9,0.4], [0.8, 0.8, 0.3], [0.1, 0.2, 0.4, 0.9]]
   large_group_names = ['A', 'B', 'C']
   small_group_names = [['a.1', 'a.2', 'a.3'], ['b.1', 'b.2', 'b.3'], ['c.1', 'c.2', 'c.3', 'c.4']]

   plot_packages(package_size, large_group_names, small_group_names, package_vals)
