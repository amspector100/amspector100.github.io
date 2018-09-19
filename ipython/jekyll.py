# I apologize for the fact that this file is not very generalizable. If you try to run this on a different computer it will sadly fail.

# modification of config created here: https://gist.github.com/cscorley/9144544
try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys

# Get root directory
ipython_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(ipython_dir)

f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        f = arg.split('.ipynb')[0]
        break
print(f)

c = get_config()
c.NbConvertApp.export_format = 'markdown'
c.MarkdownExporter.template_path = [ipython_dir] # point this to your jekyll template file
c.MarkdownExporter.template_file = 'jekyll.tpl'
#c.Application.verbose_crash=True

# modify this function to point your images to a custom path
# by default this saves all images to a directory 'images' in the root of the blog directory
def path2support(path):
    """Turn a file path into a URL"""
    ipython_subdirect = os.path.basename(path).split('_')[0] + '_files'
    return '\\assets\\images\\ipython\\' + ipython_subdirect + '\\' + os.path.basename(path)

c.MarkdownExporter.filters = {'path2support': path2support}


if f:
    c.NbConvertApp.output_base = f.lower().replace(' ', '-')
    c.FilesWriter.build_directory = parent_dir + '/_posts' # point this to your build directory