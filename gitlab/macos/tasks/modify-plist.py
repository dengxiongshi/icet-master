import sys
from pathlib import Path

file_path = f'{Path.home()}/Library/LaunchAgents/gitlab-runner.plist'

file_object = open(file_path, 'r')
content = file_object.readlines()
file_object.close()

host = str(sys.argv[1])
port = str(sys.argv[2])


for i, line in enumerate(content):
    if line.find('EnvironmentVariables') > 0:
        indent_ends = content[i + 2].find('<')
        indents = content[i + 2][:indent_ends]
        content.insert(i + 2, f'{indents}<key>HTTP_PROXY</key>\n')
        content.insert(i + 3, f'{indents}<string>{host}:{port}</string>\n')
        content.insert(i + 4, f'{indents}<key>HTTPS_PROXY</key>\n')
        content.insert(i + 5, f'{indents}<string>{host}:{port}</string>\n')
        break

file_object = open(file_path, 'w')
contents = ''.join(content)
file_object.write(contents)
file_object.close()
