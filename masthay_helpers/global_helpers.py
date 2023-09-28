from subprocess import check_output as co
import os
import numpy as np
import subprocess
import re
from functools import wraps
import sys
import argparse


# global helpers tyler
def sco(s, split=True):
    u = co(s, shell=True).decode('utf-8')
    if split:
        u = u.split('\n')[:-1]
    return u


def conda_include_everything():
    inc_paths = ':'.join(sco('find $CONDA_PREFIX/include -type d'))
    c_path = os.environ.get("C_INCLUDE_PATH")
    cmd = "echo 'export C_INCLUDE_PATH=%s:%s'" % (c_path, inc_paths)
    cmd += ' | pbcopy'
    os.system(cmd)


def inside_quad(p, f):
    face = np.array(f)

    def inside_triangle(p, A, B, C):
        AB = B - A
        BC = C - B
        AP = p - A
        BP = p - B
        magAB = np.linalg.norm(AB) ** 2
        magBC = np.linalg.norm(BC) ** 2
        dotAP = np.dot(AP, AB)
        dotBP = np.dot(BC, BP)
        testA = 0 <= dotAP and dotAP <= magAB
        testB = 0 <= dotBP and dotBP <= magBC
        return testA and testB

    A = face[0]
    B = face[1]
    C = face[2]
    D = face[3]
    return inside_triangle(p, A, B, C) or inside_triangle(p, A, C, D)


def segment_quad_intersect(a, b, face):
    v = np.array(b - a)
    p1 = face[1] - face[0]
    p2 = face[2] - face[0]
    w = np.cross(p1, p2)
    beta = np.dot(w, v)
    input(w)
    input(v)
    input(beta)
    if not np.isclose(beta, 0.0):
        gamma = np.dot(w, np.array(face[0] - a))
        candidate = a + gamma / beta * v
        inside_seg = 0 <= gamma and gamma <= beta
        input(gamma)
        input(a)
        input(v)
        input(gamma / beta)
        input('candidate = %s' % candidate)
        input(inside_seg)
        return inside_quad(candidate, face) and inside_seg, candidate
    else:
        return inside_quad(a, face), a


def get_dependencies():  # Run the grep command using subprocess
    grep_command = "grep -rE '^(import|from .* import)' --include='*.py' ."
    result = sco(grep_command)

    final = []
    for line in result:
        l = line.split(':')[1].split(' ')[1]
        if l not in final:
            final.append(l)
    return final


def prettify_dict(d, jsonify=True):
    s = str(d)
    s = re.sub(r'<function (\w+) at 0x[\da-f]+>', r'\1', s)
    s = s.replace('{', '{\n')
    s = s.replace('}', '\n}')
    s = s.replace(', ', ',\n')
    lines = s.split('\n')
    idt = 4 * ' '
    idt_level = 0
    for i, l in enumerate(lines):
        if l in ['}', '},', ',']:
            idt_level -= 1
            if idt_level < 0:
                idt_level = 0
        lines[i] = idt_level * idt + l
        if l[-1] == '{':
            idt_level += 1
    res = '\n'.join(lines)
    if jsonify:
        res = res.replace("'", '"')
    return res


def save_metadata(*, path=None, cli=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import os

            meta = func(*args, **kwargs)

            if cli:
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    '--store_path', type=str, help='Path for storing metadata'
                )
                args = parser.parse_args()
                save_path = args.store_path
            else:
                save_path = (
                    path
                    if path is not None
                    else os.path.dirname(os.path.abspath(__file__))
                )
            print(
                f'save_metadata attempting create folder {save_path}...', end=''
            )
            os.makedirs(save_path, exist_ok=True)
            print('SUCCESS')
            full_path = os.path.join(save_path, 'metadata.pydict')
            print(
                f'save_metadata attempting to save metadata to {full_path}...',
                end='',
            )
            with open(full_path, 'w') as f:
                f.write(prettify_dict(meta))
            print('SUCCESS')

            return meta

        return wrapper

    return decorator


def add_root_package_path(*, path, pkg):
    path_tokens = path.split(os.sep)
    global_root = os.sep.join(path_tokens[: path_tokens.index(pkg)])
    sys.path.append(global_root)


def path_up(path, n=1):
    if path[-1] == os.sep:
        path = path[:-1]
    path_tokens = path.split(os.sep)
    return os.sep.join(path_tokens[:-n])
