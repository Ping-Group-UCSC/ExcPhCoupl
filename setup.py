from setuptools import setup, find_packages

setup(
	name='ExcPhCoupl',
	version='0.0.1',
	license='LICENSE',
	long_description='README.md',
	packages=find_packages(),
	install_requires=[
		# List your project's dependencies here
	],
	entry_points={'console_scripts' : ['excph=src.run:main']}
)
