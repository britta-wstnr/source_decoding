""" Project settings for the source decoding project."""
import getpass

# get user (server or local machine)
user = getpass.getuser()

# base paths are different on server and local machine
if user == 'we':
    code_base_path = '/path/to/python_code_base/'
    script_path = '/path/to/code/scripts/'
    dir_out = '/path/to/output/'
elif user == 'britta':
    code_base_path = 'path/to/python_code_base/'
    script_path = 'path/to/code/scripts/'
    dir_out = 'path/to/output/'
else:
    raise ValueError('Unknown user %s.' % user)
