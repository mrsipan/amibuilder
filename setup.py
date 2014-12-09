from setuptools import setup, find_packages

extras_require={'test': ['mock', 'zope.testing']}

entry_points = '''
[console_scripts]
amibuilder = amibuilder:main

[cliff.amibuilder]
split = amibuilder:SplitFile
upload = amibuilder:Upload
bundle = amibuilder:Bundle
import = amibuilder:Import
run = amibuilder:Run
mkfs = amibuilder:MakeFs
extract = amibuilder:Extract
'''

setup(
    name='amibuilder',
    version='0.1',
    install_requires=[
        'boto', 'logutils', 'zc.thread', 'docker-py', 'cliff'],
    package_dir={'': '.'},
    py_modules=['amibuilder'],
    description='kickstart current machine',
    zip_safe=False,
    extras_require=extras_require,
    tests_require=extras_require['test'],
    entry_points=entry_points
    )
