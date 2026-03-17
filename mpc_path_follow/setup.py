from setuptools import setup

package_name = 'mpc_path_follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marian',
    maintainer_email='marian@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_path_follow_node = mpc_path_follow.mpc_path_follow_node:main',
            'mpc_path_follow_node_opp = mpc_path_follow.mpc_path_follow_node_opp:main'
        ],
    },
)
