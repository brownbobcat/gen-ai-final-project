import os

project_root = os.path.dirname(os.path.dirname(__file__))
adapter_path = os.path.join(project_root, 'model', 'adapter_model')
print(f'Current working directory: {os.getcwd()}')
print(f'Script location: {__file__}')
print(f'Project root: {project_root}')
print(f'Adapter path: {adapter_path}')
print(f'Adapter exists: {os.path.exists(adapter_path)}')